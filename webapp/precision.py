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
DOMINANT_LANGUAGE_SHARE = 0.85


@dataclass
class PrecisionConfig:
    model: str = "large-v3"
    language: str | None = None
    min_passes: int = DEFAULT_MIN_PASSES
    max_passes: int = DEFAULT_MAX_PASSES
    stability_target: float = DEFAULT_STABILITY_TARGET
    workers: int | None = None
    initial_prompt: str | None = None
    prompt_spans: list[dict] = field(default_factory=list)
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
    if workers is None:
        # int8 weights are small enough for three copies of large-v3; the
        # float32 reference implementation only fits two comfortably.
        cap = 3 if engine == "faster-whisper" else 2
        workers = max(1, min(cap, cores // 3))
    workers = max(1, workers)
    return workers, max(1, cores // workers)


def choose_engine(model: str, emit=None) -> str:
    """Prefer the fast runtime, but never fail the job over it.

    The CTranslate2 weights are a different artifact from the ``.pt`` files in
    ``~/.cache/whisper``, so the first ultra run may need to fetch them.  If
    that is not possible (no network, no disk), the reference implementation is
    already on this machine and produces the same schema.
    """
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
            te._FASTER_MODEL_NAMES.get(model, "large-v3"), device="cpu",
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


def _detect_language(executor: ProcessPoolExecutor, name: str, total: int,
                     windows: list[aa.Window], regions: list[aa.SpeechRegion],
                     emit) -> LanguageMap:
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
        language = future.result().get("language")
        if language:
            detected.append((window.start, window.end, language))

    if not detected:
        return LanguageMap()
    detected.sort()
    # Region first, then smoothing: a whole region detected as something odd
    # between two regions that agree is still noise.
    spans = _smooth(_by_region(detected, regions))

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
               emit, done_base: int, total: int) -> tuple[dict[int, list[dict]], int]:
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
    done = done_base
    futures = [executor.submit(te.transcribe_task, task) for task in tasks]
    for future in as_completed(futures):
        result = future.result()
        collected[result["pass_index"]].extend(result["words"])
        done += 1
        emit("progress", stage="transcribing", progress=min(done / max(total, 1), 1.0))

    for words in collected.values():
        words.sort(key=lambda word: (word["start"], word["end"]))
    return collected, done


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

    engine = choose_engine(config.model, emit)
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
                executor, name, samples, base_windows, regions, emit)
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
                      f"{threads} hilos · hasta {max_passes} pasadas."))

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
            collected, done = _run_round(
                executor, name, samples, regions, specs, languages, config,
                emit, done, total_tasks)
            hypotheses.update(collected)
            used += batch

            words, stats = cs.consensus_words(
                [hypotheses[i] for i in sorted(hypotheses)],
                region_bounds=[(r.start, r.end) for r in regions],
            )
            report = cs.summarize(words, stats, used, previous)
            result.rounds.append(report.as_dict())
            emit("consensus", **report.as_dict())
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
