"""Uniform transcription surface over the two local Whisper runtimes.

`faster-whisper` (CTranslate2) is preferred when its weights are available: it
is several times faster than the reference implementation. On Apple Silicon,
MLX is preferred so large models do not remain limited to CTranslate2's CPU.
`openai-whisper` is the guaranteed fallback and uses the weights already in
``~/.cache/whisper``.

Both are wrapped so the consensus layer never has to care which one produced a
hypothesis: it always gets the same ``Hypothesis`` of timed, scored words.
"""

from __future__ import annotations

import math
import os
import platform
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
FASTER_CACHE = Path.home() / ".cache" / "faster-whisper"

# openai-whisper name -> the CTranslate2 repository that holds the same weights.
_FASTER_MODEL_NAMES = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "turbo": "large-v3-turbo",
}

_MLX_MODEL_NAMES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
}


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Hypothesis:
    """One pass' opinion about one window."""

    words: list[Word] = field(default_factory=list)
    language: str | None = None
    language_probability: float = 0.0
    # Per-segment health signals, kept so the consensus can drop a whole
    # segment that Whisper itself already flagged as dubious.
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()


@dataclass(frozen=True)
class PassSpec:
    """The knobs that make one pass disagree with another.

    Passes must be *decorrelated* to be worth running.  Identical parameters
    give identical output and a vote of 5 that means no more than a vote of 1.
    """

    index: int
    temperature: float = 0.0
    beam_size: int = 5
    phase: float = 0.0
    condition_on_previous_text: bool = False
    label: str = ""


# The default ladder.  Pass 0 is the strongest single configuration; every
# later pass trades a little accuracy for a different failure mode, which is
# what lets the vote catch the first pass' mistakes.
PASS_LADDER: tuple[PassSpec, ...] = (
    PassSpec(0, temperature=0.0, beam_size=5, phase=0.0, label="beam 5 · corte A"),
    PassSpec(1, temperature=0.0, beam_size=5, phase=0.5, label="beam 5 · corte B"),
    PassSpec(2, temperature=0.2, beam_size=5, phase=0.25, label="temp 0.2 · corte C"),
    PassSpec(3, temperature=0.0, beam_size=8, phase=0.75, label="beam 8 · corte D"),
    PassSpec(4, temperature=0.4, beam_size=3, phase=0.15, label="temp 0.4 · corte E"),
    PassSpec(5, temperature=0.0, beam_size=10, phase=0.6, label="beam 10 · corte F"),
    PassSpec(6, temperature=0.1, beam_size=5, phase=0.85, label="temp 0.1 · corte G"),
    PassSpec(7, temperature=0.6, beam_size=3, phase=0.4, label="temp 0.6 · corte H"),
)


def pass_specs(count: int) -> list[PassSpec]:
    """Return `count` decorrelated pass configurations."""
    specs = list(PASS_LADDER[:count])
    # Beyond the hand-tuned ladder, keep walking the phase circle so extra
    # passes still cut the timeline somewhere new.
    for i in range(len(specs), count):
        specs.append(PassSpec(
            i, temperature=0.2 * (i % 3), beam_size=5,
            phase=(i * 0.37) % 1.0, label=f"variante {i + 1}",
        ))
    return specs


# --------------------------------------------------------------------------- #
# Motores
# --------------------------------------------------------------------------- #
def faster_model_cached(model: str) -> bool:
    """Is a CTranslate2 conversion of this model already on disk?"""
    name = _FASTER_MODEL_NAMES.get(model)
    if not name:
        return False
    root = FASTER_CACHE / f"models--Systran--faster-whisper-{name}"
    return root.is_dir() and any(root.rglob("model.bin"))


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def mlx_whisper_available() -> bool:
    """MLX only accelerates Apple Silicon; importing it elsewhere is useless."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def mlx_model_cached(model: str) -> bool:
    """Whether the exact MLX/Hugging Face artifact is already complete."""
    repo = _MLX_MODEL_NAMES.get(model)
    if not repo:
        return False
    owner, name = repo.split("/", 1)
    root = (Path.home() / ".cache" / "huggingface" / "hub" /
            f"models--{owner}--{name}")
    snapshots = root / "snapshots"
    return snapshots.is_dir() and any(
        path.is_file() and path.stat().st_size > 0
        for pattern in ("*.safetensors", "weights.npz")
        for path in snapshots.rglob(pattern)
    )


class _FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(self, model: str, threads: int) -> None:
        from faster_whisper import WhisperModel

        name = _FASTER_MODEL_NAMES.get(model, "large-v3")
        # int8_float32 keeps the accuracy of float32 accumulation while holding
        # the weights at a quarter of the memory: that is what lets several
        # workers hold their own copy of large-v3 at the same time.
        self._model = WhisperModel(
            name, device="cpu", compute_type="int8_float32",
            cpu_threads=max(1, threads), num_workers=1,
            download_root=str(FASTER_CACHE),
        )

    def transcribe(self, pcm: np.ndarray, spec: PassSpec, *,
                   language: str | None, initial_prompt: str | None) -> Hypothesis:
        segments, info = self._model.transcribe(
            pcm,
            language=language or None,
            beam_size=spec.beam_size,
            temperature=spec.temperature,
            word_timestamps=True,
            condition_on_previous_text=spec.condition_on_previous_text,
            # A reference is vocabulary, not preceding speech. `hotwords`
            # biases spelling without inviting the decoder to echo whole lines.
            initial_prompt=None,
            hotwords=initial_prompt or None,
            # The window planner already removed the silence; a second VAD here
            # would re-cut what we deliberately padded.
            vad_filter=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=2.0,
            no_repeat_ngram_size=0,
        )
        words: list[Word] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        compression: list[float] = []
        for segment in segments:
            logprobs.append(float(segment.avg_logprob))
            no_speech.append(float(segment.no_speech_prob))
            compression.append(float(segment.compression_ratio))
            for word in segment.words or ():
                text = (word.word or "").strip()
                if not text:
                    continue
                words.append(Word(
                    text=text,
                    start=float(word.start),
                    end=max(float(word.end), float(word.start) + 1e-3),
                    probability=float(getattr(word, "probability", 1.0) or 0.0),
                ))
        return Hypothesis(
            words=words,
            language=getattr(info, "language", None),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            avg_logprob=_mean(logprobs, -1.0),
            no_speech_prob=_mean(no_speech, 0.0),
            compression_ratio=_mean(compression, 1.0),
        )

    def detect_language(self, pcm: np.ndarray) -> tuple[str | None, float]:
        _segments, info = self._model.transcribe(
            pcm, beam_size=1, without_timestamps=True, vad_filter=False,
            language_detection_segments=1,
        )
        return (getattr(info, "language", None),
                float(getattr(info, "language_probability", 0.0) or 0.0))


class _OpenAIWhisperEngine:
    name = "openai-whisper"

    def __init__(self, model: str, threads: int) -> None:
        import torch
        import whisper

        torch.set_num_threads(max(1, threads))
        self._whisper = whisper
        self._model = whisper.load_model(
            model, device="cpu",
            download_root=str(Path.home() / ".cache" / "whisper"),
        )

    def transcribe(self, pcm: np.ndarray, spec: PassSpec, *,
                   language: str | None, initial_prompt: str | None) -> Hypothesis:
        result = self._model.transcribe(
            pcm.astype(np.float32),
            language=language or None,
            temperature=spec.temperature,
            beam_size=spec.beam_size if spec.temperature == 0 else None,
            word_timestamps=True,
            condition_on_previous_text=spec.condition_on_previous_text,
            # The reference runtime has no hotword channel. The exact lyric is
            # applied later only after an acoustic match, never injected here.
            initial_prompt=None,
            hallucination_silence_threshold=2.0,
            fp16=False,
            verbose=None,
        )
        words: list[Word] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        compression: list[float] = []
        for segment in result.get("segments") or ():
            logprobs.append(float(segment.get("avg_logprob", -1.0)))
            no_speech.append(float(segment.get("no_speech_prob", 0.0)))
            compression.append(float(segment.get("compression_ratio", 1.0)))
            for word in segment.get("words") or ():
                text = str(word.get("word", "")).strip()
                if not text:
                    continue
                start = float(word.get("start", segment.get("start", 0.0)))
                end = float(word.get("end", start))
                words.append(Word(
                    text=text, start=start, end=max(end, start + 1e-3),
                    probability=float(word.get("probability", 1.0) or 0.0),
                ))
        return Hypothesis(
            words=words,
            language=result.get("language"),
            language_probability=1.0 if result.get("language") else 0.0,
            avg_logprob=_mean(logprobs, -1.0),
            no_speech_prob=_mean(no_speech, 0.0),
            compression_ratio=_mean(compression, 1.0),
        )

    def detect_language(self, pcm: np.ndarray) -> tuple[str | None, float]:
        audio = self._whisper.pad_or_trim(pcm.astype(np.float32))
        mel = self._whisper.log_mel_spectrogram(
            audio, n_mels=self._model.dims.n_mels).to(self._model.device)
        _tokens, probs = self._model.detect_language(mel)
        if not probs:
            return None, 0.0
        best = max(probs, key=probs.get)
        return best, float(probs[best])


class _MlxWhisperEngine:
    """Apple-Silicon-native Whisper with the same result contract."""

    name = "mlx-whisper"

    def __init__(self, model: str, threads: int) -> None:
        import mlx_whisper

        self._mlx = mlx_whisper
        self._model = _MLX_MODEL_NAMES.get(model, _MLX_MODEL_NAMES["large-v3"])

    def _run(self, pcm: np.ndarray, *, language: str | None,
             initial_prompt: str | None, spec: PassSpec,
             word_timestamps: bool = True) -> dict:
        return self._mlx.transcribe(
            pcm.astype(np.float32, copy=False),
            path_or_hf_repo=self._model,
            language=language or None,
            temperature=spec.temperature,
            # mlx-whisper aún no implementa BeamSearchDecoder. Enviarle
            # beam_size (incluso con temperature=0) lanza NotImplementedError;
            # la capa superior lo interpretaba como una hipótesis vacía y dos
            # pasadas fallidas terminaban votando por borrar toda la letra.
            # Greedy a temperatura 0 es la ruta compatible y determinista.
            beam_size=None,
            word_timestamps=word_timestamps,
            condition_on_previous_text=spec.condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=2.0,
            verbose=False,
        )

    def transcribe(self, pcm: np.ndarray, spec: PassSpec, *,
                   language: str | None, initial_prompt: str | None) -> Hypothesis:
        result = self._run(
            pcm, language=language, initial_prompt=None, spec=spec)
        words: list[Word] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        compression: list[float] = []
        for segment in result.get("segments") or ():
            logprobs.append(float(segment.get("avg_logprob", -1.0)))
            no_speech.append(float(segment.get("no_speech_prob", 0.0)))
            compression.append(float(segment.get("compression_ratio", 1.0)))
            for word in segment.get("words") or ():
                text = str(word.get("word", "")).strip()
                if not text:
                    continue
                start = float(word.get("start", segment.get("start", 0.0)))
                end = float(word.get("end", start))
                words.append(Word(
                    text=text, start=start, end=max(end, start + 1e-3),
                    probability=float(word.get("probability", 1.0) or 0.0),
                ))
        return Hypothesis(
            words=words, language=result.get("language"),
            language_probability=1.0 if result.get("language") else 0.0,
            avg_logprob=_mean(logprobs, -1.0),
            no_speech_prob=_mean(no_speech, 0.0),
            compression_ratio=_mean(compression, 1.0),
        )

    def detect_language(self, pcm: np.ndarray) -> tuple[str | None, float]:
        result = self._run(
            pcm, language=None, initial_prompt=None,
            spec=PassSpec(0, temperature=0.0, beam_size=1),
            word_timestamps=False,
        )
        language = result.get("language")
        return language, 1.0 if language else 0.0


def _mean(values: list[float], fallback: float) -> float:
    usable = [v for v in values if math.isfinite(v)]
    return sum(usable) / len(usable) if usable else fallback


# --------------------------------------------------------------------------- #
# Worker global: un modelo por proceso, cargado una sola vez
# --------------------------------------------------------------------------- #
_ENGINE: Any = None
# name -> (SharedMemory, ndarray view).  Attaching costs a syscall, so each
# worker attaches once and reuses the mapping for every window it is given.
_SHARED: dict[str, tuple[Any, np.ndarray]] = {}


def attach_audio(name: str) -> np.ndarray:
    """View the parent's decoded audio without copying it.

    A three-hour mix is ~700 MB of float32.  Sending each window's samples
    through the task queue would hold several gigabytes of duplicated audio in
    flight at once; sharing one mapping and passing offsets holds one copy for
    the whole job.  `track=False`: the parent owns this block and unlinks it,
    so a worker must not register it with the resource tracker as well.
    """
    cached = _SHARED.get(name)
    if cached is not None:
        return cached[1]
    from multiprocessing import shared_memory

    block = shared_memory.SharedMemory(name=name, track=False)
    view = np.ndarray((block.size // 4,), dtype="<f4", buffer=block.buf)
    _SHARED[name] = (block, view)
    return view


def init_worker(model: str, engine: str, threads: int) -> None:
    """ProcessPoolExecutor initializer.  Loading the model is the expensive
    part, so it happens once per worker and is reused by every window."""
    global _ENGINE
    # Every worker asking BLAS for all cores would oversubscribe the machine
    # and run slower than a single worker.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(max(1, threads))
    _ENGINE = build_engine(model, engine, threads)


def build_engine(model: str, engine: str, threads: int):
    if engine == "mlx-whisper":
        return _MlxWhisperEngine(model, threads)
    if engine == "faster-whisper":
        return _FasterWhisperEngine(model, threads)
    return _OpenAIWhisperEngine(model, threads)


def window_pcm(payload: dict) -> np.ndarray:
    """The samples this task covers, as a contiguous array the engine accepts."""
    view = attach_audio(payload["shm"])
    return np.ascontiguousarray(view[payload["first"]:payload["last"]])


def transcribe_task(payload: dict) -> dict:
    """Run one (pass, window) unit inside a worker process.

    Takes and returns plain dicts because everything crosses a process
    boundary; the caller rebuilds the dataclasses.
    """
    spec = PassSpec(**payload["spec"])
    pcm = window_pcm(payload)
    try:
        hypothesis = _ENGINE.transcribe(
            pcm, spec,
            language=payload.get("language"),
            initial_prompt=payload.get("initial_prompt"),
        )
    except Exception as error:  # noqa: BLE001
        # One bad window must not kill a three-hour job: an empty hypothesis
        # simply casts no vote, and the other passes still decide.
        return {"pass_index": spec.index, "window_index": payload["window_index"],
                "error": str(error), "words": []}

    offset = float(payload["offset"])
    return {
        "pass_index": spec.index,
        "window_index": payload["window_index"],
        "language": hypothesis.language,
        "language_probability": hypothesis.language_probability,
        "avg_logprob": hypothesis.avg_logprob,
        "no_speech_prob": hypothesis.no_speech_prob,
        "compression_ratio": hypothesis.compression_ratio,
        "words": [
            {"text": w.text, "start": w.start + offset,
             "end": w.end + offset, "probability": w.probability}
            for w in hypothesis.words
        ],
    }


def detect_language_task(payload: dict) -> dict:
    try:
        language, probability = _ENGINE.detect_language(window_pcm(payload))
    except Exception:  # noqa: BLE001
        return {"language": None, "probability": 0.0}
    return {"language": language, "probability": probability}
