"""Audio conditioning, voice-activity detection, and separation QC.

Pure signal work: decode with ffmpeg, measure, and decide *where* there is
singing.  Nothing here loads a transcription model, so it stays fast enough to
run before every pass and cheap enough to unit-test on synthetic PCM.

Why this exists: Whisper invents text on silence and on instrumental bleed.
Feeding it only the windows that actually contain voice removes the single
largest source of hallucination in a 1–3 h mix, and it also removes the compute
spent on the (usually large) instrumental-only part of the timeline.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# VAD tuning.  The frame/hop pair gives ~50 decisions per second, which is
# enough to place a cut inside a breath without chasing individual glottal
# pulses.
FRAME_SECONDS = 0.030
HOP_SECONDS = 0.020

# Hysteresis: open on a clearly-voiced frame, stay open until it is clearly
# quiet.  A single threshold flickers on vibrato and makes hundreds of
# 40 ms fragments.
OPEN_MARGIN_DB = 9.0
CLOSE_MARGIN_DB = 4.5

MIN_SPEECH_SECONDS = 0.20
MIN_SILENCE_SECONDS = 0.35
PAD_SECONDS = 0.30

# Windows handed to the transcription engine.  Whisper's receptive field is
# 30 s; leaving headroom means a window never gets truncated mid-word.
WINDOW_SECONDS = 26.0
WINDOW_HARD_MAX = 29.0


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Window:
    """One independent unit of work for a transcription pass."""

    start: float
    end: float
    index: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class StemReport:
    """What the separator actually handed us, in numbers the UI can show."""

    vocal_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    speech_coverage: float = 0.0
    speech_seconds: float = 0.0
    bleed_ratio: float = 0.0
    clipping: float = 0.0
    verdict: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "vocalDbfs": round(self.vocal_dbfs, 2),
            "peakDbfs": round(self.peak_dbfs, 2),
            "speechCoverage": round(self.speech_coverage, 4),
            "speechSeconds": round(self.speech_seconds, 2),
            "bleedRatio": round(self.bleed_ratio, 4),
            "clipping": round(self.clipping, 6),
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Decodificación
# --------------------------------------------------------------------------- #
def decode_pcm(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any container to mono float32 in [-1, 1] via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg no pudo decodificar {path.name}: "
            f"{proc.stderr.decode('utf-8', 'replace')[-300:]}"
        )
    pcm = np.frombuffer(proc.stdout, dtype="<f4")
    # frombuffer hands back a read-only view over the subprocess buffer.
    return np.nan_to_num(pcm, copy=True, posinf=0.0, neginf=0.0)


def condition_for_asr(source: Path, target: Path,
                      sample_rate: int = SAMPLE_RATE) -> Path:
    """Write the ASR-ready copy of a vocal stem.

    Deliberately gentle.  A high-pass removes the kick/bass bleed that survives
    separation and drags Whisper's mel energy around, and `loudnorm` puts every
    song at the same level so one decoding threshold behaves the same on a
    whispered verse and on a belted chorus.  No compression, no de-esser, no
    noise gate: those remove consonants, and a lost consonant is a lost word.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
         "-af", "highpass=f=70,loudnorm=I=-18:TP=-2.0:LRA=11",
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not target.is_file():
        # Conditioning is an optimization, never a hard requirement.
        return source
    return target


# --------------------------------------------------------------------------- #
# VAD
# --------------------------------------------------------------------------- #
def frame_energies(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE
                   ) -> tuple[np.ndarray, float]:
    """Per-frame RMS in dBFS plus the hop used, as a stride view (no copies)."""
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    if pcm.size < frame:
        return np.full(1, -120.0, dtype=np.float32), HOP_SECONDS

    count = 1 + (pcm.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        pcm, shape=(count, frame),
        strides=(pcm.strides[0] * hop, pcm.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return db.astype(np.float32), hop / sample_rate


def detect_speech_regions(
    pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
    min_speech: float = MIN_SPEECH_SECONDS,
    min_silence: float = MIN_SILENCE_SECONDS,
    pad: float = PAD_SECONDS,
) -> list[SpeechRegion]:
    """Find the voiced stretches of an isolated vocal stem.

    The threshold is derived from the track itself: the 20th percentile of
    frame energy is the stem's own noise floor (separation residue, room tone),
    so a quiet ballad and a loud rap track get the same treatment without a
    magic constant that only fits one of them.
    """
    if pcm.size == 0:
        return []
    db, hop = frame_energies(pcm, sample_rate)
    if db.size == 0:
        return []

    floor = float(np.percentile(db, 20))
    ceiling = float(np.percentile(db, 97))
    # A stem with no dynamic range at all is either silence or a solid tone;
    # either way there is nothing to gain from a relative threshold.
    if ceiling - floor < 6.0:
        if ceiling < -55.0:
            return []
        return [SpeechRegion(0.0, pcm.size / sample_rate)]

    open_db = min(floor + OPEN_MARGIN_DB, ceiling - 3.0)
    close_db = floor + CLOSE_MARGIN_DB

    regions: list[list[float]] = []
    voiced = False
    run_start = 0
    silence_run = 0
    min_silence_frames = max(1, int(round(min_silence / hop)))

    for i, value in enumerate(db):
        if not voiced:
            if value >= open_db:
                voiced = True
                run_start = i
                silence_run = 0
            continue
        if value >= close_db:
            silence_run = 0
            continue
        silence_run += 1
        if silence_run >= min_silence_frames:
            regions.append([run_start * hop, (i - silence_run + 1) * hop])
            voiced = False
    if voiced:
        regions.append([run_start * hop, db.size * hop])

    total = pcm.size / sample_rate
    out: list[SpeechRegion] = []
    for start, end in regions:
        if end - start < min_speech:
            continue
        start = max(0.0, start - pad)
        end = min(total, end + pad)
        if out and start <= out[-1].end:
            out[-1] = SpeechRegion(out[-1].start, max(out[-1].end, end))
        else:
            out.append(SpeechRegion(start, end))
    return out


def plan_windows(
    regions: list[SpeechRegion], *, window_seconds: float = WINDOW_SECONDS,
    hard_max: float = WINDOW_HARD_MAX, phase: float = 0.0,
) -> list[Window]:
    """Group speech regions into windows a transcription pass can own.

    `phase` shifts where long regions get cut.  Running one pass at phase 0 and
    another at phase 0.5 means the two passes disagree about where a boundary
    falls — which is exactly what makes the consensus able to detect a word
    that one pass swallowed at its own window edge.
    """
    windows: list[Window] = []
    for region in regions:
        remaining = region.duration
        if remaining <= hard_max:
            windows.append(Window(region.start, region.end))
            continue
        # First slice absorbs the phase offset; the rest are uniform.  Splitting
        # into equal parts avoids a final 2-second orphan window.
        first = window_seconds * (1.0 - phase) if phase else window_seconds
        first = min(max(first, window_seconds * 0.35), hard_max)
        cursor = region.start
        if remaining > first:
            windows.append(Window(cursor, cursor + first))
            cursor += first
            remaining -= first
        parts = max(1, math.ceil(remaining / window_seconds))
        step = remaining / parts
        for _ in range(parts):
            end = min(cursor + step, region.end)
            windows.append(Window(cursor, end))
            cursor = end

    return [Window(w.start, w.end, i) for i, w in enumerate(windows)
            if w.duration > 0.05]


# --------------------------------------------------------------------------- #
# Control de calidad de la separación
# --------------------------------------------------------------------------- #
def _band_energy(pcm: np.ndarray, sample_rate: int,
                 low: float, high: float) -> float:
    """Mean spectral energy inside a band, via a single real FFT of a subsample."""
    if pcm.size < 2048:
        return 0.0
    # 60 s of audio is plenty to characterize a mix and keeps the FFT small.
    limit = min(pcm.size, sample_rate * 60)
    step = max(1, pcm.size // limit)
    sample = pcm[::step][:limit]
    spectrum = np.abs(np.fft.rfft(sample * np.hanning(sample.size)))
    freqs = np.fft.rfftfreq(sample.size, 1.0 / sample_rate)
    band = spectrum[(freqs >= low) & (freqs < high)]
    return float(np.mean(np.square(band))) if band.size else 0.0


def _dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-10))


def assess_separation(
    vocals: np.ndarray, instrumental: np.ndarray | None,
    duration: float, sample_rate: int = SAMPLE_RATE,
    regions: list[SpeechRegion] | None = None,
) -> StemReport:
    """Judge a separation before spending an hour transcribing it.

    A perfect stem is not measurable without a reference, but the three ways
    separation actually fails *are*: the vocal track came out near-silent, it
    still carries the full instrumental, or it clips.  Each has a different fix,
    so the report names which one happened instead of a single quality score.
    """
    report = StemReport()
    if vocals.size == 0:
        report.verdict = "empty"
        report.notes.append("La pista de voz salió vacía.")
        return report

    if regions is None:
        regions = detect_speech_regions(vocals, sample_rate)
    speech = sum(r.duration for r in regions)
    total = duration if duration > 0 else vocals.size / sample_rate

    report.vocal_dbfs = _dbfs(vocals)
    report.peak_dbfs = 20.0 * math.log10(max(float(np.max(np.abs(vocals))), 1e-10))
    report.speech_seconds = speech
    report.speech_coverage = min(1.0, speech / total) if total > 0 else 0.0
    report.clipping = float(np.mean(np.abs(vocals) >= 0.999))

    # Bleed: sub-bass has no business in a vocal stem, so energy below 110 Hz
    # relative to everything above it is leaked kick and bass.  The reference
    # band starts at 150 Hz rather than at the 300 Hz speech band on purpose —
    # a low male voice has its fundamental near 90–200 Hz, and measuring
    # against 300 Hz would report that singer's own voice as bleed.
    low = _band_energy(vocals, sample_rate, 20.0, 110.0)
    rest = _band_energy(vocals, sample_rate, 150.0, 6000.0)
    report.bleed_ratio = float(low / rest) if rest > 0 else 0.0

    if report.vocal_dbfs < -55.0:
        report.verdict = "silent"
        report.notes.append(
            "La voz aislada está casi en silencio: probablemente la canción es "
            "instrumental, o el separador falló.")
    elif report.speech_coverage < 0.02:
        report.verdict = "instrumental"
        report.notes.append(
            "Casi no se detectó canto. Si esperabas letra, prueba el perfil "
            "Ultra o revisa que el audio no sea una pista instrumental.")
    elif report.bleed_ratio > 0.55:
        report.verdict = "bleed"
        report.notes.append(
            "Queda bastante instrumental filtrado en la voz. El perfil Ultra "
            "hace una segunda pasada que suele limpiarlo.")
    elif report.clipping > 0.001:
        report.verdict = "clipping"
        report.notes.append(
            "La voz aislada satura en algunos picos; puede comerse consonantes.")
    else:
        report.verdict = "ok"
        report.notes.append(
            f"Voz aislada limpia · {report.speech_coverage * 100:.0f}% del "
            f"tiempo tiene canto.")
    return report
