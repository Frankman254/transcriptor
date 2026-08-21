"""
Transcriptor local — backend FastAPI.

Corre Whisper (openai-whisper) localmente sobre un link de YouTube o un archivo
de audio/video, y transmite el progreso en vivo por SSE:
  - modelo y dispositivo (CPU/GPU) que se está usando
  - etapa actual (descargando / transcribiendo)
  - cada segmento con su inicio --> fin apenas Whisper lo produce
  - porcentaje de avance (segundos transcritos / duración total)

No necesita internet para transcribir si el modelo ya está en la caché
(~/.cache/whisper). Si eliges un modelo que no está descargado, el backend lo
baja con `curl` (usa el llavero de macOS y evita el error SSL de proxys).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import whisper  # solo para conocer las URLs de los modelos
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = Path.home() / ".cache" / "whisper"
OUTPUT_DIR.mkdir(exist_ok=True)

# Modelos que ofrecemos en la interfaz (nombre -> etiqueta amigable).
MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]

# Idiomas comunes; "" = detección automática.
LANGUAGES = [
    ("", "Auto (detectar)"),
    ("Spanish", "Español"),
    ("English", "Inglés"),
    ("Portuguese", "Portugués"),
    ("French", "Francés"),
    ("Italian", "Italiano"),
    ("German", "Alemán"),
]

app = FastAPI(title="Transcriptor local")

# job_id -> dict(queue, done, ...)
JOBS: dict[str, dict] = {}


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
    return CACHE_DIR / os.path.basename(url)


def model_is_cached(model: str) -> bool:
    return model_cache_path(model).exists()


def ensure_model(model: str, emit) -> None:
    """Descarga el modelo con curl si no está en caché (evita el SSL de Python)."""
    if model_is_cached(model):
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


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


def download_youtube(url: str, outdir: Path, emit) -> tuple[Path, str, float]:
    """Descarga el audio como mp3. Devuelve (ruta, título, duración)."""
    emit("status", stage="downloading", message="Descargando audio de YouTube…")
    fmt = "after_move:%(id)s\t%(title)s\t%(duration)s\t%(filepath)s"
    proc = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3", "--no-simulate",
         "--print", fmt, "-o", str(outdir / "%(id)s.%(ext)s"), url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp falló: {proc.stderr[-400:]}")
    line = [l for l in proc.stdout.splitlines() if "\t" in l][-1]
    _id, title, dur, filepath = line.split("\t")
    duration = float(dur) if dur and dur != "NA" else 0.0
    return Path(filepath), title, duration


def download_best_audio(url: str, outdir: Path, emit) -> tuple[Path, str, float]:
    """Descarga el mejor stream de audio SIN recodificar (m4a/opus, sin pérdida).

    Devuelve (ruta, título, duración). Se usa en modo "solo audio".
    """
    emit("status", stage="downloading",
         message="Descargando audio en la mejor calidad…")
    fmt = "after_move:%(id)s\t%(title)s\t%(duration)s\t%(filepath)s"
    proc = subprocess.run(
        ["yt-dlp", "-f", "bestaudio/best", "--no-simulate",
         "--print", fmt, "-o", str(outdir / "%(title)s.%(ext)s"), url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp falló: {proc.stderr[-400:]}")
    line = [l for l in proc.stdout.splitlines() if "\t" in l][-1]
    _id, title, dur, filepath = line.split("\t")
    duration = float(dur) if dur and dur != "NA" else 0.0
    return Path(filepath), title, duration


def demucs_available() -> bool:
    """¿Está instalado Demucs (para el modo 'letra de canción')?"""
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate_vocals(audio: Path, outdir: Path, emit) -> Path:
    """Aísla la voz de la música con Demucs y devuelve la ruta de vocals.wav.

    La primera vez, Demucs descarga sus pesos (~una vez). Separar en CPU es
    lento (varios minutos por canción); es normal.
    """
    emit("status", stage="separating",
         message="Separando la voz de la música con Demucs (puede tardar)…")
    sep_dir = outdir / "demucs"
    proc = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals",
         "-o", str(sep_dir), str(audio)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs falló: {proc.stderr[-400:]}")
    matches = list(sep_dir.rglob("vocals.wav"))
    if not matches:
        raise RuntimeError("Demucs no produjo la pista de voz.")
    return matches[0]


def run_whisper(audio: Path, model: str, language: str, outdir: Path,
                duration: float, emit) -> dict:
    """Corre Whisper por CLI en modo verbose y transmite cada segmento."""
    cmd = ["whisper", str(audio), "--model", model,
           "--output_format", "all", "--output_dir", str(outdir),
           "--verbose", "True", "--model_dir", str(CACHE_DIR),
           # Tiempos por palabra: bordes de segmento más precisos (al ms) y
           # habilita words[] en el JSON para el puente con Lyrixa.
           "--word_timestamps", "True"]
    if language:
        cmd += ["--language", language]

    emit("status", stage="transcribing",
         message=f"Transcribiendo con '{model}' en {device_name()}…",
         model=model, device=device_name(), duration=duration)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        line = line.rstrip("\n")
        m = _TS_RE.search(line)
        if m:
            start = parse_ts(m.group(1))
            end = parse_ts(m.group(2))
            text = m.group(3).strip()
            progress = min(end / duration, 1.0) if duration else None
            emit("segment", start=start, end=end, text=text, progress=progress)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Whisper terminó con error.")

    stem = audio.stem
    files = {}
    for ext in ("txt", "srt", "vtt", "tsv", "json"):
        f = outdir / f"{stem}.{ext}"
        if f.exists():
            files[ext] = f.name
    return files


def build_lyrixa_bundle(outdir: Path, json_name: str | None,
                        title: str, duration: float) -> str | None:
    """Convierte el JSON de Whisper en un proyecto `.lyrixa.json` importable.

    Cada segmento se vuelve un clip cronometrado (startTime/endTime en segundos
    con decimales = precisión de ms) en la capa principal 'layer-main'. Lyrixa
    lo carga como proyecto y ya tiene la letra y los tiempos exactos. Si Whisper
    dio tiempos por palabra, se incluyen en `clip.words` (campo reservado que el
    renderer ignora hoy pero contempla a futuro).
    """
    if not json_name:
        return None
    try:
        data = json.loads((outdir / json_name).read_text(encoding="utf-8"))
    except Exception:
        return None

    clips: list[dict] = []
    lines: list[str] = []
    for i, seg in enumerate(data.get("segments", [])):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = round(float(seg.get("start", 0.0)), 3)
        end = round(float(seg.get("end", start)), 3)
        if end <= start:                 # respeta MIN_CLIP_DURATION de Lyrixa
            end = round(start + 0.25, 3)
        clip = {
            "id": f"clip-{i}",
            "text": text,
            "startTime": start,
            "endTime": end,
            "layerId": "layer-main",     # MAIN_LAYER_ID de Lyrixa
            "sourceIndex": len(lines),
            "createdBy": "import",
            "transitionIn": "fade",
            "transitionOut": "fade",
            "position": "center",
        }
        words = []
        for w in seg.get("words") or []:
            wt = (w.get("word") or w.get("text") or "").strip()
            if not wt or "start" not in w or "end" not in w:
                continue
            words.append({"text": wt,
                          "startTime": round(float(w["start"]), 3),
                          "endTime": round(float(w["end"]), 3)})
        if words:
            clip["words"] = words
        clips.append(clip)
        lines.append(text)

    envelope = {
        "schemaVersion": 1,
        "app": "Lyrixa",
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "project": {
            "id": f"transcriptor-{Path(json_name).stem}",
            "name": title or Path(json_name).stem,
            "lyricMode": "single",
            "rawLyricsText": "\n".join(lines),
            "normalizedLyrics": lines,
            "clips": clips,
            "currentTime": 0,
            "renderMode": "editor",
        },
    }
    out = outdir / f"{Path(json_name).stem}.lyrixa.json"
    out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out.name


def pipeline(job_id: str, source_type: str, url: str, upload_path: Path | None,
             model: str, language: str, mode: str = "transcribe") -> None:
    job = JOBS[job_id]

    def emit(event: str, **data):
        # Guardamos cada evento en un buffer (permite reconexión) y avisamos.
        with job["cond"]:
            job["events"].append({"event": event, "data": data})
            job["cond"].notify_all()

    # Mantiene la Mac despierta durante todo el trabajo.
    caffeinate = start_caffeinate()

    outdir = OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        # Modo "solo audio": descargar el mejor audio y terminar, sin Whisper.
        if mode == "audio_only":
            audio, title, duration = download_best_audio(url, outdir, emit)
            emit("meta", title=title, duration=duration, device=device_name())
            emit("done", title=title, files={"audio": audio.name},
                 job_id=job_id)
            return

        # Modo "letra de canción" necesita Demucs para aislar la voz.
        if mode == "lyrics" and not demucs_available():
            raise RuntimeError(
                "Demucs no está instalado. Instálalo con: pip3 install demucs"
            )

        ensure_model(model, emit)

        if source_type == "url":
            # Para letra usamos el mejor audio (mejor separación de voz).
            if mode == "lyrics":
                audio, title, duration = download_best_audio(url, outdir, emit)
            else:
                audio, title, duration = download_youtube(url, outdir, emit)
        else:
            audio = upload_path
            title = audio.name
            duration = ffprobe_duration(audio)

        emit("meta", title=title, duration=duration,
             model=model, device=device_name())

        if mode == "lyrics":
            audio = separate_vocals(audio, outdir, emit)

        files = run_whisper(audio, model, language, outdir, duration, emit)
        # Puente con Lyrixa: proyecto con clips ya cronometrados.
        lyrixa = build_lyrixa_bundle(outdir, files.get("json"), title, duration)
        if lyrixa:
            files["lyrixa"] = lyrixa
        emit("done", title=title, files=files, job_id=job_id)
    except Exception as e:  # noqa: BLE001
        emit("error", message=str(e))
    finally:
        if caffeinate is not None:
            caffeinate.terminate()  # deja que la Mac vuelva a poder dormir
        with job["cond"]:
            job["done"] = True  # marca fin del stream (para todos los lectores)
            job["cond"].notify_all()


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    return JSONResponse({
        "device": device_name(),
        "models": [{"name": m, "cached": model_is_cached(m)} for m in MODELS],
        "languages": [{"value": v, "label": l} for v, l in LANGUAGES],
    })


@app.post("/api/jobs")
async def create_job(
    source_type: str = Form(...),
    model: str = Form("medium"),
    language: str = Form(""),
    url: str = Form(""),
    mode: str = Form("transcribe"),
    file: UploadFile | None = File(None),
):
    job_id = uuid.uuid4().hex[:12]
    upload_path = None
    if source_type == "file":
        if file is None:
            return JSONResponse({"error": "Falta el archivo"}, status_code=400)
        outdir = OUTPUT_DIR / job_id
        outdir.mkdir(parents=True, exist_ok=True)
        upload_path = outdir / file.filename
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "events": [],                 # buffer de todos los eventos (para reconexión)
        "cond": threading.Condition(),
        "done": False,
    }
    threading.Thread(
        target=pipeline,
        args=(job_id, source_type, url, upload_path, model, language, mode),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job no existe"}, status_code=404)

    # Reconexión: el navegador reenvía el id del último evento recibido; retomamos
    # desde el siguiente y así no se pierde ni se duplica nada tras un corte.
    last = request.headers.get("last-event-id")
    start = 0
    if last is not None:
        try:
            start = int(last) + 1
        except ValueError:
            start = 0

    def gen():
        cond: threading.Condition = job["cond"]
        i = start
        while True:
            with cond:
                if i >= len(job["events"]) and not job["done"]:
                    cond.wait(timeout=15)  # espera evento nuevo o manda heartbeat
                pending = job["events"][i:]
                done = job["done"]
            if pending:
                for item in pending:
                    yield (f"id: {i}\n"
                           f"event: {item['event']}\n"
                           f"data: {json.dumps(item['data'])}\n\n")
                    i += 1
            else:
                yield ": keep-alive\n\n"  # mantiene viva la conexión TCP
            if done and i >= len(job["events"]):
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/download/{job_id}/{name}")
def download(job_id: str, name: str):
    path = OUTPUT_DIR / job_id / name
    if not path.exists():
        return JSONResponse({"error": "no existe"}, status_code=404)
    return FileResponse(path, filename=name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
