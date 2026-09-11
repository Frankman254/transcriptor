"""Utilidades del servidor que no son rutas: dispositivo,
modelos, caffeinate, timing de trabajo y nombres de archivo.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
import whisper
import paths


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def device_name() -> str:
    # Whisper (openai-whisper) por CLI usa cuda si existe, si no cpu.
    # No usa MPS por defecto, así que reportamos lo que realmente correrá.
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda (GPU)"
    except Exception:
        pass
    return "cpu"
def start_caffeinate() -> subprocess.Popen | None:
    """Impide que la Mac se duerma / bloquee / apague la pantalla mientras dura
    el trabajo. Devuelve el proceso (hay que terminarlo al acabar) o None si no
    es macOS. Flags: -d display, -i idle, -m disco, -s sistema (con corriente),
    -u actividad de usuario (mantiene la pantalla despierta).

    Nota: no evita el "clamshell sleep" al cerrar la tapa; para eso hace falta
    `sudo pmset -a disablesleep 1`. Pero el proceso sobrevive a ese sueño y
    continúa al despertar.
    """
    if not shutil.which("caffeinate"):
        return None
    try:
        return subprocess.Popen(["caffeinate", "-dimsu"])
    except Exception:
        return None
def model_cache_path(model: str) -> Path:
    """Ruta donde Whisper guarda el .pt del modelo."""
    url = whisper._MODELS[model]
    return paths.CACHE_DIR / os.path.basename(url)
def model_is_cached(model: str) -> bool:
    return model_cache_path(model).exists()
def ensure_model(model: str, emit) -> None:
    """Descarga el modelo con curl si no está en caché (evita el SSL de Python)."""
    if model_is_cached(model):
        return
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = whisper._MODELS[model]
    target = model_cache_path(model)
    emit("status", stage="downloading_model",
         message=f"Descargando modelo '{model}' (~una vez)…")
    proc = subprocess.run(
        ["curl", "-L", "--fail", "-o", str(target), url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        if target.exists():
            target.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo descargar el modelo '{model}': {proc.stderr[-300:]}"
        )
def _start_job_timing(job: dict) -> None:
    now = time.monotonic()
    job["_timing"] = {
        "started": now, "stageStarted": now,
        "stage": "starting", "durations": {}, "finished": False,
    }
def _mark_job_stage(job: dict, stage: str | None) -> None:
    timing = job.get("_timing")
    if not timing or timing.get("finished") or not stage or stage == timing["stage"]:
        return
    now = time.monotonic()
    elapsed = now - timing["stageStarted"]
    durations = timing["durations"]
    durations[timing["stage"]] = durations.get(timing["stage"], 0.0) + elapsed
    timing["stage"] = stage
    timing["stageStarted"] = now
def _finish_job_timing(job: dict) -> dict:
    timing = job.get("_timing") or {}
    if not timing:
        return {}
    if not timing.get("finished"):
        now = time.monotonic()
        stage = timing.get("stage", "unknown")
        durations = timing.setdefault("durations", {})
        durations[stage] = durations.get(stage, 0.0) + (
            now - timing.get("stageStarted", now))
        timing["elapsed"] = now - timing.get("started", now)
        timing["finished"] = True
    return {
        "elapsedSeconds": round(float(timing.get("elapsed", 0.0)), 2),
        "stages": {
            key: round(value, 2)
            for key, value in timing.get("durations", {}).items()
        },
    }
# Caracteres que no pueden ir en un nombre de archivo (o que rompen la
# cabecera Content-Disposition al descargar).
_BAD_FS_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
def safe_stem(name: str, fallback: str = "transcripcion") -> str:
    """Convierte un título en un nombre de archivo usable, sin perder legibilidad.

    Se usa para que el .txt/.srt/.json salgan como "Nombre del video.srt" en vez
    de "u2xI9o6siyI.srt" (id de YouTube) o "vocals.srt" (salida de Demucs).
    """
    stem = _BAD_FS_CHARS.sub("_", (name or "").strip())
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:120].strip() or fallback
_TS_RE = re.compile(
    r"\[(\d+:\d+(?::\d+)?\.\d+)\s*-->\s*(\d+:\d+(?::\d+)?\.\d+)\]\s?(.*)"
)
def parse_ts(ts: str) -> float:
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s
