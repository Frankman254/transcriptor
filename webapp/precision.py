"""The ultra-precise transcription pipeline.

Ties the pieces together:

    separar voz → VAD → ventanas → N pasadas en paralelo → voto → ¿convergió?

The loop is the point.  Instead of trusting one decoding of the audio, it keeps
adding decorrelated passes and re-voting until an extra pass stops changing the
transcript.  When that happens, the remaining disagreement is in the audio, not
in the decoder, and more compute would buy nothing.

Everything is local and restartable: the caller gets an ``emit`` callback for
progress and a plain dict back.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "PrecisionConfig", "PrecisionResult", "engine_availability",
    "rescue_from_original", "run",
]

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable

import numpy as np

import audio_analysis as aa
import consensus as cs
import transcribe_engines as te
from translation import normalize_code

# Where the convergence loop stops.  0.99 means an extra pass changed less than
# one word in a hundred; past that the votes are chasing punctuation.
DEFAULT_STABILITY_TARGET = 0.99
DEFAULT_MIN_PASSES = 3
DEFAULT_MAX_PASSES = 5

# A one-hour DJ set is not one language.  Detection runs on every window
# instead of on a handful of samples: one encoder pass per window is cheap next
# to three full transcription passes, and sampling eight spots over an hour is
# how a mix that opens in Japanese gets transcribed as English.
#
# When a single language covers this much of the speech, it is treated as the
# language of the whole track — forcing it is more stable than letting an
# ad-lib window drift. Below it, each window keeps its own.
DOMINANT_LANGUAGE_SHARE = 0.97
MIN_LANGUAGE_PROBABILITY = 0.60

# Cuánto tiene que sumar un idioma que no sea inglés dentro de un tramo con
# pista de idioma para quedarse con el tramo entero.  El umbral es bajo a
# propósito y la asimetría está justificada: sobre voz con instrumental colado
# Whisper responde "inglés" por defecto, pero casi nunca inventa japonés — para
# decir "ja" tiene que oír fonética japonesa.  Una minoría en japonés dentro de
# una canción pesa más que una mayoría en inglés, porque dentro de una misma
# canción no cambia quién canta ni en qué idioma.
PRIOR_MINORITY_SHARE = 0.15

# Gate del rescate: una zona donde el consenso tiene letra sostenida por dos o
# más pasadas no se reemplaza jamás contra UNA sola hipótesis del mix. Una
# pasada sobre audio con música nunca aporta evidencia más fuerte que varios
# votos sobre el stem limpio, y la acústica del mix está correlacionada con la
# de cualquier pasada solitaria. Lo que sí gana siempre: huecos sin letra
# (count-ins borrados por el separador) y texto que ninguna mayoría sostiene.
RESCUE_MIN_VOTED_WORDS = 3


@dataclass
class PrecisionConfig:
    model: str = "large-v3"
    language: str | None = None
    min_passes: int = DEFAULT_MIN_PASSES
    max_passes: int = DEFAULT_MAX_PASSES
    stability_target: float = DEFAULT_STABILITY_TARGET
    engine_preference: str = "auto"
    workers: int | None = None
    initial_prompt: str | None = None
    prompt_spans: list[dict] = field(default_factory=list)
    language_priors: list[dict] = field(default_factory=list)
    segment_refiner: Callable[[list[dict]], list[dict]] | None = None


@dataclass
class PrecisionResult:
    segments: list[dict] = field(default_factory=list)
    language: str | None = None
    language_probability: float = 0.0
    engine: str = ""
    passes: int = 0
    converged: bool = False
    rounds: list[dict] = field(default_factory=list)
    windows: int = 0
    speech_seconds: float = 0.0
    languages: dict | None = None
    dropped: int = 0
    looped: int = 0
    stem: dict | None = None
    reference: dict | None = None
    rescued_windows: int = 0
    rescue_model: str | None = None
    consensus_fallback: bool = False

    @property
    def text(self) -> str:
        return "\n".join(segment["text"] for segment in self.segments)

    def as_report(self) -> dict:
        return {
            "engine": self.engine,
            "passes": self.passes,
            "converged": self.converged,
            # A single round has nothing to compare against, so the UI must not
            # report "did not converge" when convergence was never checked.
            "checked": len(self.rounds) > 1,
            "rounds": self.rounds,
            "windows": self.windows,
            "speechSeconds": round(self.speech_seconds, 2),
            "droppedSlots": self.dropped,
            "loopedWords": self.looped,
            "language": self.language,
            "languageProbability": round(self.language_probability, 4),
            "languages": self.languages,
            "stem": self.stem,
            "reference": self.reference,
            "rescuedWindows": self.rescued_windows,
            "rescueModel": self.rescue_model,
            "consensusFallback": self.consensus_fallback,
        }


# --------------------------------------------------------------------------- #
# Recursos
# --------------------------------------------------------------------------- #
def plan_resources(engine: str, workers: int | None = None) -> tuple[int, int]:
    """(worker processes, threads each) for this machine and runtime.

    Oversubscription is the classic way to make a parallel transcription slower
    than a serial one: every worker's BLAS grabbing all ten cores means ten
    workers fighting over ten cores.  Total threads stay at the core count.
    """
    cores = os.cpu_count() or 4
    # MLX comparte la GPU y memoria unificada. Varias copias del modelo pelean
    # por el mismo dispositivo y suelen ser más lentas que una cola única.
    if engine == "mlx-whisper":
        return 1, cores
    if workers is None:
        # int8 weights are small enough for three copies of large-v3; the
        # float32 reference implementation only fits two comfortably.
        cap = 3 if engine == "faster-whisper" else 2
        workers = max(1, min(cap, cores // 3))
    workers = max(1, workers)
    return workers, max(1, cores // workers)


def engine_availability(models) -> dict:
    """Qué motores de ASR existen en esta máquina y qué pesos tiene cada uno.

    Una sola pregunta en lugar de seis.  Quien pinta el catálogo de motores
    quiere saber qué puede ofrecer; no tiene por qué saber que debajo hay un
    `transcribe_engines`, ni que cada runtime guarda un artefacto distinto
    (el `.pt` de OpenAI, la conversión CTranslate2, los pesos MLX).  Esa
    correspondencia vive aquí, junto a `choose_engine`, que es quien la usa
    para decidir de verdad.

    `auto` responde con el mismo criterio que `choose_engine`: lo que la
    interfaz anuncia y lo que el trabajo acaba usando no pueden divergir.
    """
    mlx = te.mlx_whisper_available()
    faster = te.faster_whisper_available()
    auto = ("mlx-whisper" if mlx else
            "faster-whisper" if faster else "openai-whisper")
    cached = {}
    for name in models:
        by_engine = {
            "mlx-whisper": te.mlx_model_cached(name),
            "faster-whisper": te.faster_model_cached(name),
        }
        cached[name] = {
            **by_engine,
            # El modo ultra corre con el motor que elegiría `auto`, así que la
            # interfaz debe decir si *ese* tiene los pesos, no otro.
            "auto": by_engine.get(auto, True),
        }
    return {
        "auto": auto,
        "available": {"mlx-whisper": mlx, "faster-whisper": faster,
                      "openai-whisper": True},
        "cached": cached,
    }


def choose_engine(model: str, emit=None, preferred: str = "auto") -> str:
    """Prefer the fast runtime, but never fail the job over it.

    The CTranslate2 weights are a different artifact from the ``.pt`` files in
    ``~/.cache/whisper``, so the first ultra run may need to fetch them.  If
    that is not possible (no network, no disk), the reference implementation is
    already on this machine and produces the same schema.
    """
    preferred = preferred if preferred in {
        "auto", "mlx-whisper", "faster-whisper", "openai-whisper"
    } else "auto"
    if preferred in {"auto", "mlx-whisper"} and te.mlx_whisper_available():
        if emit:
            emit("status", stage="loading_model",
                 message=f"Preparando '{model}' con MLX en Apple Silicon…")
        return "mlx-whisper"
    if preferred == "mlx-whisper" and emit:
        emit("status", stage="loading_model",
             message="MLX no está disponible; usaré faster-whisper como respaldo.")
    if preferred == "openai-whisper":
        return "openai-whisper"
    if not te.faster_whisper_available():
        return "openai-whisper"
    if te.faster_model_cached(model):
        return "faster-whisper"
    if emit:
        emit("status", stage="downloading_model",
             message=f"Descargando el modelo rápido '{model}' (sólo esta vez)…")
    try:
        from faster_whisper import WhisperModel

        probe = WhisperModel(
            te.faster_model_name(model), device="cpu",
            compute_type="int8_float32", cpu_threads=1,
            download_root=str(te.FASTER_CACHE),
        )
        del probe
        return "faster-whisper"
    except Exception:  # noqa: BLE001
        if emit:
            emit("status", stage="transcribing",
                 message="No se pudo bajar el motor rápido; uso el modelo local.")
        return "openai-whisper"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@contextmanager
def shared_audio(pcm: np.ndarray):
    """Publish the decoded audio once, for every worker to read in place.

    Yields ``(name, sample_count)``.  Only the name escapes, never a view: a
    live numpy array over the block would keep the buffer exported and make
    `close()` fail, so the single view stays local and is dropped here.
    """
    block = shared_memory.SharedMemory(create=True, size=max(4, pcm.nbytes))
    view = np.ndarray(pcm.shape, dtype="<f4", buffer=block.buf)
    view[:] = pcm
    try:
        yield block.name, pcm.size
    finally:
        del view
        block.close()
        block.unlink()


def _task(name: str, window: aa.Window, total: int,
          sample_rate: int = aa.SAMPLE_RATE) -> dict:
    """Where in the shared block this window lives."""
    return {
        "shm": name,
        "first": max(0, int(window.start * sample_rate)),
        "last": min(total, int(window.end * sample_rate)),
    }


@dataclass
class LanguageMap:
    """Which language is sung where.

    A single-language track collapses to one span and behaves exactly as
    before; a multilingual mix keeps one span per window so each part is
    transcribed — and later translated — in the language it was actually sung
    in.
    """

    dominant: str | None = None
    share: float = 0.0
    spans: list[tuple[float, float, str]] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def multilingual(self) -> bool:
        return len({span[2] for span in self.spans}) > 1

    def at(self, start: float, end: float) -> str | None:
        """The language sung over this window, by the time it covers most of."""
        if not self.multilingual:
            return self.dominant
        overlap: dict[str, float] = {}
        for span_start, span_end, language in self.spans:
            shared = min(end, span_end) - max(start, span_start)
            if shared > 0:
                overlap[language] = overlap.get(language, 0.0) + shared
        if not overlap:
            return self.dominant
        return max(overlap, key=lambda key: overlap[key])

    def as_dict(self) -> dict:
        return {
            "dominant": self.dominant,
            "share": round(self.share, 4),
            "multilingual": self.multilingual,
            "seconds": {k: round(v, 1) for k, v in sorted(
                self.seconds.items(), key=lambda kv: -kv[1])},
        }


def _smooth(spans: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Drop single-window flips.

    One window of "oh oh oh" can be detected as anything.  A language that
    appears for exactly one window between two windows that agree with each
    other is noise, not a language change.
    """
    if len(spans) < 3:
        return spans
    out = list(spans)
    for i in range(1, len(out) - 1):
        before, current, after = out[i - 1][2], out[i][2], out[i + 1][2]
        if current != before and before == after:
            out[i] = (out[i][0], out[i][1], before)
    return out


def _by_region(detected: list[tuple[float, float, str]],
               regions: list[aa.SpeechRegion]) -> list[tuple[float, float, str]]:
    """Collapse per-window detections to one language per speech region.

    Every pass cuts the timeline at a different phase, so a window near a
    language change would get a different answer depending on the pass — and
    two passes transcribing the same audio in two languages produce text that
    cannot be voted on at all.  Regions are shared by every pass and windows
    never cross them, so pinning the language to the region makes the answer
    independent of where a pass happened to cut.

    It also matches reality: in a DJ set the language changes when the track
    does, and a track change is a silence, which is a region boundary.
    """
    spans: list[tuple[float, float, str]] = []
    for region in regions:
        seconds: dict[str, float] = {}
        for start, end, language in detected:
            shared = min(region.end, end) - max(region.start, start)
            if shared > 0:
                seconds[language] = seconds.get(language, 0.0) + shared
        if seconds:
            winner = max(seconds, key=lambda key: seconds[key])
            spans.append((region.start, region.end, winner))
    return spans


def _apply_priors(spans: list[tuple[float, float, str]],
                  priors: list[dict]) -> list[tuple[float, float, str]]:
    """Reescribe los tramos que el tracklist dice que no están en inglés.

    Un tramo con pista de idioma se queda entero con ese idioma si el detector
    lo apoya aunque sea en minoría, o si el detector solo vio inglés — que es
    justo el fallo que la pista está para corregir.  Si el detector vio otro
    idioma distinto del inglés y distinto de la pista, manda el detector: oyó
    el audio y la pista solo leyó un nombre.
    """
    if not priors or not spans:
        return spans
    out = list(spans)
    for prior in priors:
        start, end = float(prior["start"]), float(prior["end"])
        expected = prior["language"]
        inside = [index for index, (a, b, _) in enumerate(out)
                  if min(b, end) - max(a, start) > 0]
        if not inside:
            continue
        seconds: dict[str, float] = {}
        for index in inside:
            a, b, language = out[index]
            seconds[language] = seconds.get(language, 0.0) + (b - a)
        total = sum(seconds.values()) or 1.0
        others = {k: v for k, v in seconds.items() if k not in ("en", expected)}
        if others and max(others.values()) / total >= PRIOR_MINORITY_SHARE:
            continue                    # el detector oyó otra cosa: le creemos
        if seconds.get(expected, 0.0) / total >= PRIOR_MINORITY_SHARE or (
                set(seconds) <= {"en"}):
            for index in inside:
                a, b, _ = out[index]
                out[index] = (a, b, expected)
    return out


def _detect_language(executor: ProcessPoolExecutor, name: str, total: int,
                     windows: list[aa.Window], regions: list[aa.SpeechRegion],
                     emit, priors: list[dict] | None = None) -> LanguageMap:
    """Detect the language of every window, then decide if one rules them all."""
    if not windows:
        return LanguageMap()
    emit("status", stage="detecting_language",
         message=f"Detectando el idioma en {len(windows)} tramos con voz…")

    futures = {
        executor.submit(te.detect_language_task, _task(name, window, total)): window
        for window in windows
    }
    detected: list[tuple[float, float, str]] = []
    for future in as_completed(futures):
        window = futures[future]
        answer = future.result()
        language = answer.get("language")
        probability = float(answer.get("probability") or 0.0)
        if language and probability >= MIN_LANGUAGE_PROBABILITY:
            detected.append((window.start, window.end, language))

    if not detected:
        return LanguageMap()
    detected.sort()
    # Region first, then smoothing: a whole region detected as something odd
    # between two regions that agree is still noise.
    spans = _apply_priors(_smooth(_by_region(detected, regions)), priors or [])

    seconds: dict[str, float] = {}
    for start, end, language in spans:
        seconds[language] = seconds.get(language, 0.0) + (end - start)
    total_seconds = sum(seconds.values())
    dominant = max(seconds, key=lambda key: seconds[key])
    share = seconds[dominant] / total_seconds if total_seconds else 0.0

    if share >= DOMINANT_LANGUAGE_SHARE:
        # One language for the whole track: collapse the spans so downstream
        # code takes the simple path.
        spans = [(spans[0][0], spans[-1][1], dominant)]
    return LanguageMap(dominant=dominant, share=share, spans=spans,
                       seconds=seconds)


def _run_round(executor: ProcessPoolExecutor, name: str, samples: int,
               regions: list[aa.SpeechRegion], specs: list[te.PassSpec],
               languages: LanguageMap, config: PrecisionConfig,
               emit, done_base: int, total: int,
               ) -> tuple[dict[int, list[dict]], int, dict[int, list[str]]]:
    """Submit every (pass, window) unit of this round and collect the words."""
    tasks = []
    for spec in specs:
        # Each pass cuts the timeline differently, so a window's language is
        # looked up by time rather than carried from the detection pass.
        for window in aa.plan_windows(regions, phase=spec.phase):
            tasks.append({
                **_task(name, window, samples),
                "spec": vars(spec),
                "window_index": window.index,
                "offset": window.start,
                "language": languages.at(window.start, window.end),
                "initial_prompt": _prompt_at(
                    config.prompt_spans, window.start, window.end)
                    or config.initial_prompt,
            })

    collected: dict[int, list[dict]] = {spec.index: [] for spec in specs}
    failures: dict[int, list[str]] = {spec.index: [] for spec in specs}
    done = done_base
    futures = [executor.submit(te.transcribe_task, task) for task in tasks]
    for future in as_completed(futures):
        result = future.result()
        collected[result["pass_index"]].extend(result["words"])
        if result.get("error"):
            failures[result["pass_index"]].append(result["error"])
        done += 1
        emit("progress", stage="transcribing", progress=min(done / max(total, 1), 1.0))

    for words in collected.values():
        words.sort(key=lambda word: (word["start"], word["end"]))
    return collected, done, failures


def _prompt_at(spans: list[dict], start: float, end: float) -> str | None:
    """Choose the song reference that overlaps most of an ASR window."""
    if not spans:
        return None

    def overlap(span: dict) -> float:
        return max(
            0.0,
            min(end, float(span.get("end", 0))) -
            max(start, float(span.get("start", 0))),
        )

    winner = max(spans, key=overlap)
    return winner.get("prompt") if overlap(winner) > 0 else None


def run(audio: Path, config: PrecisionConfig, emit,
        stem_report: aa.StemReport | None = None) -> PrecisionResult:
    """Transcribe `audio` by voting several passes until the answer stops moving."""
    emit("status", stage="analyzing",
         message="Analizando el audio y buscando dónde hay voz…")
    pcm = aa.decode_pcm(audio)
    regions = aa.detect_speech_regions(pcm)
    speech = sum(region.duration for region in regions)
    if not regions:
        return PrecisionResult(
            language=config.language, engine="", passes=0, converged=True,
            stem=stem_report.as_dict() if stem_report else None,
        )

    base_windows = aa.plan_windows(regions)
    emit("status", stage="analyzing",
         message=(f"{len(base_windows)} ventanas con voz · "
                  f"{speech / 60:.1f} min de canto sobre "
                  f"{pcm.size / aa.SAMPLE_RATE / 60:.1f} min de audio."))

    engine = choose_engine(config.model, emit, config.engine_preference)
    workers, threads = plan_resources(engine)
    max_passes = max(config.min_passes, config.max_passes)
    # Upper bound for the progress bar: the convergence loop usually stops
    # earlier, and finishing "ahead of schedule" reads better than a bar that
    # jumps backwards when a round is added.
    total_tasks = len(base_windows) * max_passes

    result = PrecisionResult(
        engine=engine, windows=len(base_windows), speech_seconds=speech,
        stem=stem_report.as_dict() if stem_report else None,
    )

    context = multiprocessing.get_context("spawn")
    # Exit order matters: the executor shuts down (and its workers unmap the
    # block) before `shared_audio` unlinks it.
    with shared_audio(pcm) as (name, samples), ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=te.init_worker, initargs=(config.model, engine, threads),
    ) as executor:
        del pcm  # the shared block is now the only copy of the audio
        # Whisper wants an ISO code; the UI may hand us "Spanish".
        forced = normalize_code(config.language)
        if forced:
            # An explicit choice is an instruction, not a hint: one span, no
            # detection, no per-window drift.
            languages = LanguageMap(
                dominant=forced, share=1.0,
                spans=[(0.0, float("inf"), forced)], seconds={forced: speech})
        else:
            languages = _detect_language(
                executor, name, samples, base_windows, regions, emit,
                priors=config.language_priors)
            if languages.multilingual:
                mix = " · ".join(
                    f"{code} {seconds / 60:.0f} min"
                    for code, seconds in sorted(
                        languages.seconds.items(), key=lambda kv: -kv[1])[:4])
                emit("status", stage="detecting_language",
                     message=(f"Mezcla multilingüe: {mix}. Cada tramo se "
                              f"transcribe y se traduce en su propio idioma."))
            elif languages.dominant:
                emit("status", stage="detecting_language",
                     message=(f"Idioma detectado: {languages.dominant} "
                              f"({languages.share * 100:.0f}% del canto)."))
        result.language = languages.dominant
        result.language_probability = languages.share
        result.languages = languages.as_dict()

        emit("status", stage="transcribing",
             message=(f"Transcribiendo con {engine} · {workers} procesos × "
                      f"{threads} hilos · hasta {max_passes} pasadas."),
             engine=engine)

        hypotheses: dict[int, list[dict]] = {}
        previous: list[cs.ConsensusWord] | None = None
        words: list[cs.ConsensusWord] = []
        stats = cs.ConsensusStats()
        done = 0
        used = 0

        while used < max_passes:
            # First round runs the minimum needed for a majority; after that,
            # one pass at a time so we stop the moment it stops helping.
            batch = config.min_passes if used == 0 else 1
            specs = te.pass_specs(used + batch)[used:used + batch]
            emit("status", stage="transcribing",
                 message=(f"Pasada {used + 1}–{used + batch} de hasta "
                          f"{max_passes} · {', '.join(s.label for s in specs)}"))
            collected, done, failures = _run_round(
                executor, name, samples, regions, specs, languages, config,
                emit, done, total_tasks)
            for spec in specs:
                # Una canción puede producir una ventana vacía; una pasada
                # completa vacía *con errores* es un motor roto, no evidencia
                # acústica de silencio. Nunca debe participar en la votación.
                expected = len(aa.plan_windows(regions, phase=spec.phase))
                if (not collected[spec.index]
                        and len(failures[spec.index]) >= expected):
                    first = failures[spec.index][0]
                    raise RuntimeError(
                        f"La pasada {spec.index + 1} falló en todas sus ventanas: "
                        f"{first}"
                    )
            hypotheses.update(collected)
            used += batch

            words, stats = cs.consensus_words(
                [hypotheses[i] for i in sorted(hypotheses)],
                region_bounds=[(r.start, r.end) for r in regions],
            )
            primary = hypotheses.get(0) or max(
                hypotheses.values(), key=len, default=[])
            retention = len(words) / len(primary) if primary else 1.0
            fallback = bool(primary and retention < 0.08)
            if fallback:
                # Última barrera: un desacuerdo temporal catastrófico nunca
                # puede convertir miles de palabras oídas en una pista
                # supuestamente instrumental. Conserva la pasada determinista
                # y marca sus líneas como dudosas para revisión.
                words = cs.single_pass_words(primary, passes=used)
                result.consensus_fallback = True
                emit("status", stage="transcribing",
                     message=("El consenso retuvo menos del 8% de la pasada "
                              "principal; conservaré esa lectura en vez de "
                              "borrar toda la letra."))
            report = cs.summarize(words, stats, used, previous)
            report_data = report.as_dict()
            report_data.update({"fallback": fallback,
                                "retention": round(retention, 4)})
            result.rounds.append(report_data)
            emit("consensus", **report_data)
            previous = words

            if report.stability is not None and report.stability >= config.stability_target:
                result.converged = True
                emit("status", stage="transcribing",
                     message=(f"Convergió con {used} pasadas "
                              f"({report.stability * 100:.2f}% estable)."))
                break

    result.passes = used
    result.dropped = stats.dropped
    result.looped = stats.looped
    result.segments = cs.to_segments(words, language_at=languages.at)
    if config.segment_refiner:
        result.segments = config.segment_refiner(result.segments)
    for segment in result.segments:
        emit("segment", start=segment["start"], end=segment["end"],
             text=segment["text"], agreement=segment.get("agreement"),
             referenceScore=segment.get("referenceScore"),
             track=segment.get("track"),
             progress=1.0)
    return result


def _rescue_intervals(segments: list[dict], duration: float,
                      limit: int = 24) -> list[tuple[float, float]]:
    """Lowest-confidence lyrics plus short holes where a shout may be missing."""
    suspects = sorted(
        (segment for segment in segments
         if float(segment.get("agreement", 1.0)) < 0.70
         or float(segment.get("confidence", 1.0)) < 0.48),
        key=lambda segment: (
            float(segment.get("agreement", 1.0))
            * float(segment.get("confidence", 1.0))),
    )[:max(1, limit * 2 // 3)]
    intervals = [
        (max(0.0, float(item["start"]) - 1.2),
         min(duration, float(item["end"]) + 1.2))
        for item in suspects
    ]
    # Stem separators often erase a brief spoken count-in completely. Inspect
    # short holes between neighboring lyrics as well; the acoustic gate below
    # refuses weak candidates from the full mix.
    gaps: list[tuple[float, float]] = []
    for left, right in zip(segments, segments[1:]):
        gap = float(right["start"]) - float(left["end"])
        if 1.0 <= gap <= 8.0:
            gaps.append((max(0.0, float(left["end"]) - 0.8),
                         min(duration, float(right["start"]) + 0.8)))
    # Reserve a third of the budget for fully erased count-ins and distribute
    # it across the mix instead of spending it all on the first songs.
    gap_budget = min(len(gaps), max(1, limit - len(intervals)))
    if gap_budget == 1 and gaps:
        intervals.append(gaps[len(gaps) // 2])
    elif gap_budget > 1:
        intervals.extend(gaps[round(i * (len(gaps) - 1) / (gap_budget - 1))]
                         for i in range(gap_budget))
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.5 and end - merged[-1][0] <= 29:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, min(end, start + 29.0)])
    return [(start, end) for start, end in merged[:limit] if end - start > 0.2]


def _consensus_protects(current: list[dict], start: float, end: float,
                        passes: int) -> bool:
    """True si la zona ya está escrita por mayoría y el rescate no debe tocarla.

    Con varias pasadas, dos o más votos sobre la misma palabra son consenso
    real: una sola hipótesis del mix con música no puede borrar ese texto. Sin
    votación (una pasada) no hay consenso que proteger y manda la acústica.
    Solo cuentan las palabras caídas dentro de la ventana: los bordes vecinos
    (el relleno de 0.8 s) no deben proteger un hueco vacío.
    """
    if passes <= 1:
        return False
    voted_inside = sum(
        1
        for segment in current
        for word in (segment.get("words") or ())
        if word.get("votes", 0) >= 2
        and start <= float(word.get("start", 0.0)) < end
    )
    return voted_inside >= RESCUE_MIN_VOTED_WORDS


def rescue_from_original(original: Path, result: PrecisionResult,
                         config: PrecisionConfig, emit,
                         max_windows: int = 24) -> PrecisionResult:
    """Recheck only dubious stem regions against the untouched mix.

    A replacement needs materially stronger acoustic evidence. This lets a
    count-in erased by the separator return without globally feeding music to
    Whisper or paying for another full pass.
    """
    if not result.segments or not original.is_file():
        return result
    duration = max(float(segment["end"]) for segment in result.segments)
    intervals = _rescue_intervals(result.segments, duration, max_windows)
    if not intervals:
        return result
    rescue_model = "large-v3" if config.model == "turbo" else config.model
    emit("status", stage="rescuing",
         message=(f"Revisando {len(intervals)} zonas dudosas en el audio original "
                  f"con {rescue_model}…"))
    pcm = aa.decode_pcm(original)
    engine_name = choose_engine(
        rescue_model, emit, config.engine_preference)
    _workers, threads = plan_resources(engine_name, workers=1)
    engine = te.build_engine(rescue_model, engine_name, threads)
    segments = list(result.segments)
    accepted = 0
    for index, (start, end) in enumerate(intervals):
        first = max(0, int(start * aa.SAMPLE_RATE))
        last = min(pcm.size, int(end * aa.SAMPLE_RATE))
        language = next(
            (segment.get("language") for segment in segments
             if float(segment["start"]) < end and float(segment["end"]) > start),
            result.language,
        )
        hypothesis = engine.transcribe(
            np.ascontiguousarray(pcm[first:last]), te.pass_specs(1)[0],
            language=language, initial_prompt=None)
        emit("progress", stage="rescuing",
             progress=(index + 1) / max(len(intervals), 1))
        if (not hypothesis.words or hypothesis.no_speech_prob >= 0.55
                or hypothesis.compression_ratio >= 2.4
                or hypothesis.avg_logprob < -1.0):
            continue
        current = [segment for segment in segments
                   if float(segment["start"]) < end
                   and float(segment["end"]) > start]
        if _consensus_protects(current, start, end, result.passes):
            continue
        current_score = (
            sum(float(segment.get("confidence", 0.0))
                * float(segment.get("agreement", 0.0)) for segment in current)
            / max(len(current), 1)
        )
        mean_probability = sum(word.probability for word in hypothesis.words) / len(
            hypothesis.words)
        log_score = max(0.0, min(1.0, hypothesis.avg_logprob + 1.0))
        rescue_score = 0.8 * mean_probability + 0.2 * log_score
        old_words = sum(len(segment.get("words") or ()) for segment in current)
        if (rescue_score < max(0.58, current_score + 0.10)
                or len(hypothesis.words) > max(12, old_words * 2 + 8)):
            continue
        raw = [
            {"text": word.text, "start": word.start + start,
             "end": word.end + start, "probability": word.probability}
            for word in hypothesis.words
        ]
        voted, _stats = cs.consensus_words([raw])
        replacements = cs.to_segments(voted, language=language)
        if not replacements:
            continue
        for segment in replacements:
            segment["rescuedFromOriginal"] = True
            segment["rescueModel"] = rescue_model
            segment["rescueScore"] = round(rescue_score, 4)
        segments = [segment for segment in segments
                    if not (float(segment["start"]) < end
                            and float(segment["end"]) > start)]
        segments.extend(replacements)
        segments.sort(key=lambda segment: (float(segment["start"]),
                                            float(segment["end"])))
        accepted += 1
    del pcm, engine
    for index, segment in enumerate(segments):
        segment["id"] = index
    result.segments = segments
    result.rescued_windows = accepted
    result.rescue_model = rescue_model
    emit("rescue", reviewed=len(intervals), accepted=accepted,
         model=rescue_model, engine=engine_name)
    return result
