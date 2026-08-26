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
from collections import deque
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

import audio_analysis
import precision
import track_guides
import translation
from lyrixa_bridge import build_lyrixa_envelope, clip_languages, clip_lines

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = Path.home() / ".cache" / "whisper"
ROFORMER_VENV_DIR = BASE_DIR / ".separator-venv"
ROFORMER_CLI = ROFORMER_VENV_DIR / "bin" / "audio-separator"
ROFORMER_MODEL_DIR = BASE_DIR / ".separator-models"
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUTPUT_DIR.mkdir(exist_ok=True)
LYRICS_CACHE_DIR = OUTPUT_DIR / ".lyrics-cache"
MANIFEST_NAME = "job.json"

# Modelos que ofrecemos en la interfaz (nombre -> etiqueta amigable).
MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]

# Formatos de audio ofrecidos al descargar.
#   value      -> lo que se le pasa a `yt-dlp --audio-format` ("best" = no recodificar)
#   lossless   -> no tiene bitrate que elegir
#   cover      -> el contenedor admite carátula incrustada
AUDIO_FORMATS = [
    {"value": "best", "label": "Original (sin recodificar)",
     "hint": "Tal cual viene de YouTube (opus/m4a). Máxima fidelidad y lo más rápido.",
     "lossless": True, "cover": True},
    {"value": "mp3", "label": "MP3",
     "hint": "El más compatible: suena en cualquier reproductor.",
     "lossless": False, "cover": True},
    {"value": "m4a", "label": "M4A (AAC)",
     "hint": "Mejor calidad que MP3 al mismo peso. Ideal para Apple Music/iPhone.",
     "lossless": False, "cover": True},
    {"value": "opus", "label": "Opus",
     "hint": "El más eficiente: máxima calidad en el menor tamaño.",
     "lossless": False, "cover": True},
    {"value": "vorbis", "label": "OGG Vorbis",
     "hint": "Libre y compatible con Android y reproductores abiertos.",
     "lossless": False, "cover": True},
    {"value": "flac", "label": "FLAC",
     "hint": "Sin pérdida y comprimido. Para archivar o escuchar en alta fidelidad.",
     "lossless": True, "cover": True},
    {"value": "alac", "label": "ALAC",
     "hint": "Sin pérdida de Apple. Para la biblioteca de Música de macOS/iOS.",
     "lossless": True, "cover": True},
    {"value": "wav", "label": "WAV",
     "hint": "Sin comprimir, para editar. Pesa mucho y no admite carátula.",
     "lossless": True, "cover": False},
]

# Bitrates para los formatos con pérdida (mp3/m4a/opus/vorbis).
AUDIO_QUALITIES = ["320K", "256K", "192K", "128K", "96K"]

# Perfiles de aislamiento. BS-RoFormer es la opción de mayor precisión; Demucs
# queda disponible como alternativa equilibrada o rápida.
SEPARATION_PROFILES = {
    "ultra": {
        "label": "Ultra · BS-RoFormer ×2 (máxima limpieza)",
        "hint": ("Dos pasadas de BS-RoFormer: la segunda quita el instrumental "
                 "que sobrevive a la primera. Se queda con la que mide mejor, "
                 "así que nunca sale peor que Preciso. Tarda el doble."),
        "engine": "roformer",
        "passes": 2,
    },
    "precise": {
        "label": "Preciso · BS-RoFormer (recomendado)",
        "hint": ("Modelo moderno que recupera mejor voces débiles y coros. "
                 "La primera vez descarga ~610 MB; después queda en caché."),
        "engine": "roformer",
        "passes": 1,
    },
    "quality": {
        "label": "Equilibrado · Demucs",
        "hint": "Demucs con 2 pasadas y mayor solapamiento. Buena calidad, pero puede omitir voces mezcladas como instrumentos.",
        "engine": "demucs",
        "shifts": 2,
        "overlap": 0.5,
    },
    "fast": {
        "label": "Rápido · Demucs",
        "hint": "Una pasada de Demucs. Más rápido, pero puede dejar voces fantasma o perder palabras.",
        "engine": "demucs",
        "shifts": 1,
        "overlap": 0.25,
    },
}

_FORMATS_BY_VALUE = {f["value"]: f for f in AUDIO_FORMATS}

# Idiomas comunes; "" = detección automática. En el modo ultra la detección
# automática vota entre varios puntos del audio, así que sirve también para
# ruso, japonés o coreano aunque no estén en esta lista.
LANGUAGES = [
    ("", "Auto (detectar)"),
    ("Spanish", "Español"),
    ("English", "Inglés"),
    ("Portuguese", "Portugués"),
    ("French", "Francés"),
    ("Italian", "Italiano"),
    ("German", "Alemán"),
    ("Japanese", "Japonés"),
    ("Russian", "Ruso"),
    ("Korean", "Coreano"),
    ("Chinese", "Chino"),
]

# Idiomas de destino para la capa 2 de Lyrixa (la traducción).
TARGET_LANGUAGES = ["es", "en", "pt", "fr", "it", "de", "ja", "ko", "zh", "ru"]

# Cuántas pasadas de consenso corre el modo ultra. El bucle para solo cuando
# una pasada más ya no cambia el texto, así que el máximo es un techo, no una
# promesa: casi siempre converge antes.
PRECISION_LEVELS = {
    "balanced": {
        "label": "Equilibrado · hasta 5 pasadas",
        "hint": ("Tres pasadas para votar y hasta dos más si el texto todavía "
                 "cambia. Es el punto dulce para la mayoría de canciones."),
        "min_passes": 3, "max_passes": 5, "stability": 0.99,
    },
    "fast": {
        "label": "Rápido · 3 pasadas",
        "hint": ("El mínimo para que haya mayoría. Ya elimina casi todo el "
                 "texto inventado y es lo más rápido del modo ultra."),
        "min_passes": 3, "max_passes": 3, "stability": 0.0,
    },
    "maximum": {
        "label": "Máximo · hasta 8 pasadas",
        "hint": ("Para letras difíciles: voces tapadas, coros o mezclas de "
                 "tres horas. Exige más acuerdo antes de darse por satisfecho."),
        "min_passes": 4, "max_passes": 8, "stability": 0.995,
    },
}

app = FastAPI(title="Transcriptor local")

# job_id -> dict(queue, done, ...)
JOBS: dict[str, dict] = {}
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SOURCE_SUFFIXES = {
    ".mp3", ".m4a", ".opus", ".ogg", ".flac", ".wav",
    ".mp4", ".webm", ".mov", ".mkv",
}


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


def recovery_assets(job_id: str) -> dict | None:
    """Descubre checkpoints útiles de un job cuyo servidor se reinició.

    Los procesos no sobreviven al cierre de la sesión de macOS, pero los WAV sí.
    Si existe ``asr-input.wav`` podemos retomar justo antes del consenso, que es
    la etapa cara posterior a la separación.
    """
    if not _JOB_ID_RE.fullmatch(job_id):
        return None
    outdir = (OUTPUT_DIR / job_id).resolve()
    if not outdir.is_dir() or OUTPUT_DIR.resolve() not in outdir.parents:
        return None

    asr = outdir / "asr-input.wav"
    vocals = next((p for p in outdir.rglob("vocals.wav") if p.is_file()), None)
    instrumental = next(
        (p for name in ("instrumental.wav", "no_vocals.wav")
         for p in outdir.rglob(name) if p.is_file()),
        None,
    )
    sources = [
        p for p in outdir.iterdir()
        if p.is_file() and p.suffix.lower() in _SOURCE_SUFFIXES
        and p.name != "asr-input.wav"
    ]
    original = max(sources, key=lambda p: p.stat().st_mtime) if sources else None
    checkpoint = asr if asr.is_file() else vocals
    title = original.stem if original else "transcripcion recuperada"
    duration_source = original or checkpoint
    duration = ffprobe_duration(duration_source) if duration_source else 0.0
    return {
        "outdir": outdir,
        "asr": asr if asr.is_file() else None,
        "vocals": vocals,
        "instrumental": instrumental,
        "original": original,
        "checkpoint": checkpoint,
        "title": title,
        "duration": duration,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest(outdir: Path) -> dict:
    try:
        data = json.loads((outdir / MANIFEST_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def update_manifest(job_id: str, **changes) -> dict:
    """Persist the small part of a job that must survive a server restart."""
    outdir = OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(outdir)
    manifest.update(changes)
    manifest.setdefault("id", job_id)
    manifest.setdefault("createdAt", _now())
    manifest["updatedAt"] = _now()
    target = outdir / MANIFEST_NAME
    temporary = outdir / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(target)
    return manifest


def load_track_guide(outdir: Path) -> dict | None:
    try:
        data = json.loads((outdir / "track-guide.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def prepare_track_guide(outdir: Path, text: str, duration: float,
                        fetch_lyrics: bool, emit) -> dict | None:
    """Build once, cache on disk, and reuse on every later iteration."""
    tracklist_path = outdir / "tracklist.txt"
    incoming = (text or "").strip()
    if incoming:
        tracklist_path.write_text(incoming + "\n", encoding="utf-8")
    elif tracklist_path.is_file():
        incoming = tracklist_path.read_text(encoding="utf-8")
        existing = load_track_guide(outdir)
        if existing:
            emit("reference", stage="loaded", **existing.get("references", {}),
                 section=existing.get("section"))
            return existing
    if not incoming:
        return None

    emit("status", stage="references",
         message="Leyendo el tracklist y preparando una guía por canción…")

    def progress(done: int, total: int, track: track_guides.Track) -> None:
        emit("progress", stage="references", progress=done / max(total, 1))
        if fetch_lyrics:
            emit("status", stage="references",
                 message=(f"Buscando letras de referencia {done}/{total} · "
                          f"{track.artist} — {track.title}"))

    guide = track_guides.build_guide(
        incoming, duration, fetch_lyrics=fetch_lyrics,
        cache_dir=LYRICS_CACHE_DIR, progress=progress,
    )
    if not guide:
        raise RuntimeError(
            "No pude leer el tracklist. Usa una línea por canción: "
            "00:00 Artista - Título"
        )
    (outdir / "track-guide.json").write_text(
        json.dumps(guide, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("reference", stage="prepared", **guide["references"],
         section=guide["section"], tracks=guide["tracks"])
    return guide


def _archive_outputs(outdir: Path) -> int:
    """Keep small prior transcripts when the user iterates the same stems."""
    manifest = read_manifest(outdir)
    iteration = int(manifest.get("iteration") or 0) + 1
    archive = outdir / "iterations" / f"iteration-{iteration:02d}"
    candidates: list[Path] = []
    for path in outdir.iterdir():
        if not path.is_file() or path.name in {
            MANIFEST_NAME, "track-guide.json", "tracklist.txt", "asr-input.wav",
        }:
            continue
        name = path.name.lower()
        if (path.suffix.lower() in {".srt", ".vtt", ".tsv"}
                or name.endswith(".lyrixa.json")
                or name.endswith(".letra.txt")
                or (path.suffix.lower() == ".json" and name != MANIFEST_NAME)
                or path.suffix.lower() == ".txt"):
            candidates.append(path)
    if candidates:
        archive.mkdir(parents=True, exist_ok=True)
        for path in candidates:
            shutil.move(str(path), str(archive / path.name))
    update_manifest(outdir.name, iteration=iteration)
    return iteration


def _discover_result_files(job_id: str, outdir: Path, assets: dict | None) -> dict:
    manifest_files = read_manifest(outdir).get("files") or {}
    files: dict = {}
    for kind, descriptor in manifest_files.items():
        name = descriptor if isinstance(descriptor, str) else descriptor.get("name", "")
        if kind in {"vocals", "instrumental"}:
            continue
        if name and (outdir / Path(name).name).is_file():
            files[kind] = Path(name).name

    root_files = [path for path in outdir.iterdir() if path.is_file()]
    def newest(predicate):
        matches = [path for path in root_files if predicate(path)]
        return max(matches, key=lambda path: path.stat().st_mtime).name if matches else None

    guesses = {
        "lyrixa": newest(lambda p: p.name.endswith(".lyrixa.json")),
        "lyrics": newest(lambda p: p.name.endswith(".letra.txt")),
        "srt": newest(lambda p: p.suffix.lower() == ".srt"),
        "vtt": newest(lambda p: p.suffix.lower() == ".vtt"),
        "tsv": newest(lambda p: p.suffix.lower() == ".tsv"),
        "json": newest(lambda p: p.suffix.lower() == ".json" and
                       p.name not in {MANIFEST_NAME, "track-guide.json"} and
                       not p.name.endswith(".lyrixa.json")),
        "txt": newest(lambda p: p.suffix.lower() == ".txt" and
                      p.name != "tracklist.txt" and
                      not p.name.endswith(".letra.txt")),
    }
    for kind, name in guesses.items():
        if name:
            files.setdefault(kind, name)
    if (outdir / "track-guide.json").is_file():
        files["guide"] = "track-guide.json"
    if assets and assets.get("original"):
        files.setdefault("audio", assets["original"].name)
    if assets and assets.get("vocals"):
        files["vocals"] = {
            "name": "voz aislada.wav",
            "url": f"/api/jobs/{job_id}/stems/vocals",
        }
    if assets and assets.get("instrumental"):
        files["instrumental"] = {
            "name": "instrumental.wav",
            "url": f"/api/jobs/{job_id}/stems/instrumental",
        }
    return files


def saved_job(job_id: str) -> dict | None:
    assets = recovery_assets(job_id)
    if assets is None:
        return None
    outdir = assets["outdir"]
    manifest = read_manifest(outdir)
    live = JOBS.get(job_id)
    files = _discover_result_files(job_id, outdir, assets)
    completed = any(kind in files for kind in ("json", "txt", "lyrixa"))
    if live and not live["done"]:
        status = "running"
    elif completed:
        status = "completed"
    elif assets.get("checkpoint"):
        status = "interrupted"
    else:
        status = manifest.get("status") or "saved"
    guide = load_track_guide(outdir)
    modified = datetime.fromtimestamp(outdir.stat().st_mtime, timezone.utc).isoformat()
    return {
        "id": job_id,
        "title": manifest.get("title") or assets["title"],
        "duration": float(manifest.get("duration") or assets["duration"] or 0),
        "status": status,
        "recoverable": assets.get("checkpoint") is not None,
        "checkpoint": "asr" if assets.get("asr") else
                      "vocals" if assets.get("vocals") else None,
        "createdAt": manifest.get("createdAt") or modified,
        "updatedAt": manifest.get("updatedAt") or modified,
        "iteration": int(manifest.get("iteration") or 0),
        "files": files,
        "reference": (guide or {}).get("references"),
        "section": (guide or {}).get("section"),
        "error": manifest.get("error"),
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


# Progreso de yt-dlp: lo pedimos en un formato fijo y parseable en vez de
# tener que leer la barra bonita que imprime por defecto.
_YT_PROGRESS_TMPL = (
    "download:@PROG@%(progress.downloaded_bytes)s|"
    "%(progress.total_bytes,progress.total_bytes_estimate)s|"
    "%(progress.fragment_index)s|%(progress.fragment_count)s"
)
_PROG_RE = re.compile(r"^@PROG@(\S*)\|(\S*)\|(\S*)\|(\S*)$")

# Nombres internos de los postprocesadores de yt-dlp -> qué contarle al usuario.
_PP_LABELS = {
    "ExtractAudio": "Convirtiendo el audio…",
    "ThumbnailsConvertor": "Preparando la carátula…",
    "EmbedThumbnail": "Incrustando la carátula…",
    "FFmpegMetadata": "Escribiendo los metadatos…",
    "Metadata": "Escribiendo los metadatos…",
}


def _pos_float(text: str) -> float | None:
    """Convierte a float los campos de yt-dlp, que valen 'NA' cuando no se saben."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _yt_dlp_error(tail: str) -> str:
    """Traduce el error crudo de yt-dlp a algo accionable."""
    if "403" in tail or "Forbidden" in tail:
        return ("YouTube rechazó la descarga (HTTP 403). Casi siempre significa "
                "que yt-dlp se quedó viejo: actualízalo con "
                "`brew upgrade yt-dlp` (o `pip3 install -U yt-dlp`) y reintenta.")
    if "Sign in to confirm" in tail or "bot" in tail.lower():
        return ("YouTube pide verificar que no eres un bot. Prueba a exportar las "
                "cookies del navegador: `yt-dlp --cookies-from-browser chrome …`")
    if "Video unavailable" in tail or "Private video" in tail:
        return "El video no está disponible (privado, borrado o con bloqueo regional)."
    return f"yt-dlp falló: {tail[-400:]}"


def audio_download_args(audio_format: str, quality: str) -> tuple[list[str], dict]:
    """Argumentos de yt-dlp para el formato pedido + la ficha de ese formato.

    "best" conserva el códec original (no recodifica); cualquier otro valor
    convierte con ffmpeg. En ambos casos incrustamos metadatos y, si el
    contenedor lo admite, la carátula del video como portada del álbum.
    """
    spec = _FORMATS_BY_VALUE.get(audio_format) or _FORMATS_BY_VALUE["best"]

    # Siempre extraemos el audio. Con "best", ffmpeg solo lo re-empaqueta (copia
    # el códec tal cual, sin recodificar): el resultado es .opus o .m4a en vez
    # del .webm crudo, que ni admite carátula ni lo abren muchos reproductores.
    args: list[str] = ["-x", "--audio-format", spec["value"]]
    if not spec["lossless"]:
        args += ["--audio-quality", quality or "192K"]

    args += [
        "--embed-metadata",
        # La descripción entera de YouTube dentro del tag ensucia cualquier
        # reproductor (son miles de caracteres); la vaciamos.
        "--parse-metadata", ":(?P<meta_description>)",
        "--parse-metadata", ":(?P<meta_synopsis>)",
        # Si el video no declara artista/álbum (o sea: no viene de YouTube
        # Music), caemos al canal para no dejar los campos vacíos.
        "--parse-metadata", "%(artist,creator,uploader)s:%(meta_artist)s",
        "--parse-metadata", "%(album,playlist_title,uploader)s:%(meta_album)s",
    ]
    if spec["cover"]:
        args += ["--embed-thumbnail", "--convert-thumbnails", "jpg"]
    return args, spec


def download_audio(url: str, outdir: Path, emit, audio_format: str = "best",
                   quality: str = "192K",
                   name_template: str = "%(title)s.%(ext)s",
                   ) -> tuple[Path, str, float]:
    """Descarga el audio de un link y emite el avance en vivo.

    Devuelve (ruta, título, duración). El archivo sale con sus metadatos y su
    carátula ya incrustados.
    """
    extra, spec = audio_download_args(audio_format, quality)
    emit("status", stage="downloading",
         message=f"Descargando audio ({spec['label']})…")
    emit("progress", stage="downloading", progress=0.0)

    cmd = [
        "yt-dlp", "-f", "bestaudio/best", "--no-simulate", "--no-playlist",
        # --print activa --quiet, que se lleva por delante el progreso;
        # --progress lo vuelve a encender.
        "--newline", "--progress",
        # Descargas largas: reintenta en vez de morir a mitad de camino.
        "--retries", "10", "--fragment-retries", "10",
        "--concurrent-fragments", "4",
        "--progress-template", _YT_PROGRESS_TMPL,
        "--progress-template", "postprocess:@PP@%(progress.postprocessor)s",
        "--print", "after_move:@DONE@%(id)s\t%(title)s\t%(duration)s\t%(filepath)s",
        "-o", str(outdir / name_template),
    ] + extra + [url]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    tail: deque[str] = deque(maxlen=30)   # por si hay que explicar un fallo
    done_line: str | None = None
    last_pct = -1
    last_pp = None

    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("@DONE@"):
            done_line = line[len("@DONE@"):]
            continue
        if line.startswith("@PP@"):
            label = _PP_LABELS.get(line[len("@PP@"):].strip())
            if label and label != last_pp:
                last_pp = label
                emit("status", stage="postprocessing", message=label)
                emit("progress", stage="postprocessing", progress=None)
            continue
        m = _PROG_RE.match(line)
        if m:
            got, total, frag_i, frag_n = m.groups()
            fraction = None
            if _pos_float(total):
                fraction = (_pos_float(got) or 0.0) / _pos_float(total)
            elif _pos_float(frag_n):          # HLS: no hay bytes, pero sí fragmentos
                fraction = (_pos_float(frag_i) or 0.0) / _pos_float(frag_n)
            if fraction is not None:
                pct = int(min(fraction, 1.0) * 100)
                if pct != last_pct:           # 1 evento por punto porcentual
                    last_pct = pct
                    emit("progress", stage="downloading",
                         progress=min(fraction, 1.0))
            continue
        tail.append(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(_yt_dlp_error("\n".join(tail)))

    if not done_line:
        raise RuntimeError("yt-dlp terminó pero no reportó el archivo descargado.")
    _id, title, dur, filepath = done_line.split("\t")
    duration = float(dur) if dur and dur != "NA" else 0.0
    emit("progress", stage="downloading", progress=1.0)
    return Path(filepath), title, duration


def demucs_available() -> bool:
    """¿Está instalado Demucs (para el modo 'letra de canción')?"""
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def roformer_available() -> bool:
    """¿Está instalado el separador moderno en su entorno aislado?"""
    return ROFORMER_CLI.is_file() and os.access(ROFORMER_CLI, os.X_OK)


def separation_profile_available(profile_name: str) -> bool:
    profile = SEPARATION_PROFILES.get(profile_name)
    if not profile:
        return False
    if profile["engine"] == "roformer":
        return roformer_available()
    return demucs_available()


# Demucs escribe una barra tqdm en stderr:
#   " 25%|█████▌      | 5.85/23.4 [00:01<00:03, 5.66seconds/s]"
# Nos quedamos con la fracción exacta (procesado/total), no con el entero.
_DEMUCS_RE = re.compile(r"(\d+)%\|.*?\|\s*([\d.]+)/([\d.]+)\s")
_DEMUCS_BAG_RE = re.compile(r"bag of (\d+) models")


def separate_vocals_demucs(
    audio: Path, outdir: Path, emit, profile_name: str = "quality",
) -> tuple[Path, Path]:
    """Aísla voz/música y devuelve ``(vocals.wav, no_vocals.wav)``.

    La primera vez, Demucs descarga sus pesos (~una vez). Separar en CPU es
    lento (varios minutos por canción); por eso vamos emitiendo el avance.

    Los modelos "bag of N" (como htdemucs_ft) pasan la pista por N modelos y
    dibujan N barras seguidas, así que las encadenamos en un solo porcentaje
    global: cada barra completa vale 1/N del total.
    """
    profile = SEPARATION_PROFILES.get(profile_name, SEPARATION_PROFILES["quality"])
    emit("status", stage="separating",
         message=f"Separando la voz con Demucs — {profile['label']} (puede tardar)…")
    emit("progress", stage="separating", progress=0.0)

    sep_dir = outdir / "demucs"
    proc = subprocess.Popen(
        [sys.executable, "-m", "demucs", "-n", "htdemucs",
         "--two-stems=vocals", "--shifts", str(profile["shifts"]),
         "--overlap", str(profile["overlap"]),
         "-o", str(sep_dir), str(audio)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    tail: deque[str] = deque(maxlen=30)
    n_models = 1        # cuántos modelos contiene el bag
    n_passes = int(profile["shifts"])  # modelos × shifts
    finished = 0        # pasadas ya completadas
    prev_fraction = 0.0
    last_pct = -1

    # text=True traduce los \r de la barra a \n, así que cada refresco de tqdm
    # llega aquí como una línea suelta.
    for line in proc.stdout:
        line = line.rstrip("\n").strip()
        if not line:
            continue
        bag = _DEMUCS_BAG_RE.search(line)
        if bag:
            n_models = max(1, int(bag.group(1)))
            n_passes = n_models * int(profile["shifts"])
            tail.append(line)
            continue
        m = _DEMUCS_RE.search(line)
        if not m:
            tail.append(line)
            continue
        done, total = float(m.group(2)), float(m.group(3))
        fraction = min(done / total, 1.0) if total else 0.0
        if fraction < prev_fraction - 0.01:   # la barra reinició = nuevo modelo
            finished = min(finished + 1, n_passes - 1)
        prev_fraction = fraction
        overall = (finished + fraction) / n_passes
        pct = int(overall * 100)
        if pct != last_pct:
            last_pct = pct
            emit("progress", stage="separating", progress=min(overall, 1.0))

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs falló: {chr(10).join(tail)[-400:]}")

    matches = list(sep_dir.rglob("vocals.wav"))
    if not matches:
        raise RuntimeError("Demucs no produjo la pista de voz.")
    vocals = matches[0]
    instrumental = vocals.with_name("no_vocals.wav")
    if not instrumental.is_file():
        raise RuntimeError("Demucs no produjo la pista instrumental.")
    emit("progress", stage="separating", progress=1.0)
    return vocals, instrumental


_ROFORMER_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")
_ROFORMER_CHUNK_RE = re.compile(r"Processing chunk (\d+)/(\d+)")


def separate_vocals_roformer(
    audio: Path, outdir: Path, emit, tag: str = "roformer",
) -> tuple[Path, Path]:
    """Aísla voz con BS-RoFormer usando un entorno Python independiente.

    ``audio-separator`` trae un stack grande de PyTorch/ONNX que puede entrar en
    conflicto con Whisper y Demucs. Por eso ``setup_separator.sh`` lo instala en
    ``.separator-venv`` y aquí sólo invocamos su CLI.
    """
    if not roformer_available():
        raise RuntimeError(
            "BS-RoFormer no está instalado. Ejecuta: cd webapp && "
            "./setup_separator.sh"
        )

    model_cached = (ROFORMER_MODEL_DIR / ROFORMER_MODEL).is_file()
    if model_cached:
        emit("status", stage="separating",
             message="Separando la voz con BS-RoFormer (puede tardar)…")
    else:
        emit("status", stage="downloading_separator_model",
             message="Descargando BS-RoFormer (~610 MB, sólo esta vez)…")
    emit("progress", stage="separating", progress=0.0)

    sep_dir = outdir / tag
    sep_dir.mkdir(parents=True, exist_ok=True)
    ROFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROFORMER_CLI), str(audio),
        "--model_filename", ROFORMER_MODEL,
        "--model_file_dir", str(ROFORMER_MODEL_DIR),
        "--output_dir", str(sep_dir),
        "--output_format", "WAV",
        # Los mixes de una hora pueden ocupar varios GB al convertirse a
        # float32. Los bloques de 10 min mantienen estable el uso de memoria.
        "--chunk_duration", "600",
        "--custom_output_names",
        json.dumps({"Vocals": "vocals", "Instrumental": "instrumental"}),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    tail: deque[str] = deque(maxlen=40)
    current_chunk = 0
    total_chunks = 1
    separating_started = model_cached
    last_pct = -1
    for line in proc.stdout:
        line = line.rstrip("\n").strip()
        if not line:
            continue
        tail.append(line)

        if not separating_started and "MDXC Separator initialisation complete" in line:
            separating_started = True
            emit("status", stage="separating",
                 message="Separando la voz con BS-RoFormer (puede tardar)…")
            emit("progress", stage="separating", progress=0.0)

        chunk = _ROFORMER_CHUNK_RE.search(line)
        if chunk:
            current_chunk = max(0, int(chunk.group(1)) - 1)
            total_chunks = max(1, int(chunk.group(2)))

        progress = _ROFORMER_PROGRESS_RE.search(line)
        if not progress:
            continue
        fraction = min(int(progress.group(1)) / 100.0, 1.0)
        if not separating_started:
            emit("progress", stage="downloading_separator_model",
                 progress=fraction)
            continue
        overall = min((current_chunk + fraction) / total_chunks, 0.99)
        pct = int(overall * 100)
        if pct != last_pct:
            last_pct = pct
            emit("progress", stage="separating", progress=overall)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"BS-RoFormer falló: {chr(10).join(tail)[-600:]}"
        )

    vocals = sep_dir / "vocals.wav"
    instrumental = sep_dir / "instrumental.wav"
    if not vocals.is_file() or not instrumental.is_file():
        raise RuntimeError("BS-RoFormer no produjo ambas pistas de audio.")
    emit("progress", stage="separating", progress=1.0)
    return vocals, instrumental


def measure_stems(vocals: Path, instrumental: Path | None,
                  duration: float) -> audio_analysis.StemReport:
    """Mide la voz aislada. Nunca tumba el trabajo: si no se puede medir,
    devolvemos un informe vacío y seguimos."""
    try:
        pcm = audio_analysis.decode_pcm(vocals)
        other = (audio_analysis.decode_pcm(instrumental)
                 if instrumental and instrumental.is_file() else None)
        return audio_analysis.assess_separation(pcm, other, duration)
    except Exception:  # noqa: BLE001
        return audio_analysis.StemReport()


def _second_pass_is_better(first: audio_analysis.StemReport,
                           second: audio_analysis.StemReport) -> bool:
    """¿La segunda pasada limpió sin comerse la voz?

    Volver a pasar el separador sobre su propia salida quita el instrumental
    que sobrevivió, pero también puede borrar susurros y coros. Solo la
    aceptamos si además de bajar la fuga conserva casi todo el canto: medir es
    la única forma de que "más limpio" no signifique "más vacío".
    """
    if second.verdict in {"empty", "silent"}:
        return False
    if second.speech_coverage < first.speech_coverage * 0.85:
        return False
    return second.bleed_ratio < first.bleed_ratio * 0.9


def separate_vocals(
    audio: Path, outdir: Path, emit, profile_name: str = "precise",
    duration: float = 0.0,
) -> tuple[Path, Path, audio_analysis.StemReport]:
    """Aísla la voz y devuelve (voz, instrumental, informe de calidad)."""
    profile = SEPARATION_PROFILES.get(profile_name, SEPARATION_PROFILES["precise"])
    if profile["engine"] != "roformer":
        vocals, instrumental = separate_vocals_demucs(
            audio, outdir, emit, profile_name)
        return vocals, instrumental, measure_stems(vocals, instrumental, duration)

    vocals, instrumental = separate_vocals_roformer(audio, outdir, emit)
    report = measure_stems(vocals, instrumental, duration)
    emit("stem", stage="separated", pass_number=1, **report.as_dict())

    if profile.get("passes", 1) < 2 or report.verdict in {"empty", "silent"}:
        return vocals, instrumental, report

    emit("status", stage="separating",
         message="Segunda pasada de BS-RoFormer para quitar el instrumental "
                 "que quedó en la voz…")
    try:
        refined, residue = separate_vocals_roformer(
            vocals, outdir, emit, tag="roformer-2")
    except RuntimeError as error:
        emit("status", stage="separating",
             message=f"La segunda pasada falló ({error}); me quedo con la primera.")
        return vocals, instrumental, report

    refined_report = measure_stems(refined, residue, duration)
    emit("stem", stage="separated", pass_number=2, **refined_report.as_dict())
    if _second_pass_is_better(report, refined_report):
        emit("status", stage="separating",
             message=(f"Me quedo con la segunda pasada: la fuga instrumental "
                      f"bajó de {report.bleed_ratio:.3f} a "
                      f"{refined_report.bleed_ratio:.3f}."))
        # El instrumental sigue siendo el de la primera pasada: es el
        # complemento real de la mezcla original.
        return refined, instrumental, refined_report

    emit("status", stage="separating",
         message="La segunda pasada no mejoró la medición; me quedo con la primera.")
    return vocals, instrumental, report


def run_whisper(audio: Path, model: str, language: str, outdir: Path,
                duration: float, emit, out_stem: str | None = None) -> dict:
    """Corre Whisper por CLI en modo verbose y transmite cada segmento.

    `out_stem` es el nombre con el que queremos que salgan los archivos. Whisper
    los nombra según el audio que le damos, que puede ser el id de YouTube o el
    `vocals.wav` de Demucs; aquí los renombramos al título de verdad.
    """
    cmd = ["whisper", str(audio), "--model", model,
           "--output_format", "all", "--output_dir", str(outdir),
           "--verbose", "True", "--model_dir", str(CACHE_DIR),
           # Tiempos por palabra: bordes de segmento más precisos (al ms) y
           # habilita words[] en el JSON para el puente con Lyrixa.
           "--word_timestamps", "True",
           # Reduce bucles de texto sobre silencios/residuos de Demucs. No
           # elimina coros repetidos por postproceso: la decisión sigue siendo
           # de Whisper y los resultados raw permanecen auditables.
           "--condition_on_previous_text", "False",
           "--hallucination_silence_threshold", "2.0"]
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
    target = safe_stem(out_stem or stem, fallback=stem)
    files = {}
    for ext in ("txt", "srt", "vtt", "tsv", "json"):
        f = outdir / f"{stem}.{ext}"
        if not f.exists():
            continue
        if target != stem:
            renamed = outdir / f"{target}.{ext}"
            f.replace(renamed)      # replace: sobrescribe si ya existía
            f = renamed
        files[ext] = f.name
    return files


def write_whisper_outputs(result: dict, outdir: Path, stem: str) -> dict:
    """Escribe txt/srt/vtt/tsv/json usando los writers del propio Whisper.

    El modo ultra no pasa por la CLI de Whisper, pero sus segmentos tienen
    exactamente la misma forma; reutilizar los writers garantiza que un SRT del
    modo ultra y uno del modo normal sean indistinguibles para cualquier
    reproductor.
    """
    from whisper.utils import get_writer

    files: dict[str, str] = {}
    options = {"max_line_width": None, "max_line_count": None,
               "highlight_words": False, "max_words_per_line": None}
    for extension in ("txt", "srt", "vtt", "tsv", "json"):
        try:
            writer = get_writer(extension, str(outdir))
            writer(result, f"{stem}.wav", options)
        except Exception:  # noqa: BLE001
            continue
        target = outdir / f"{stem}.{extension}"
        if target.is_file():
            files[extension] = target.name
    return files


def run_precision_mode(audio: Path, model: str, language: str, outdir: Path,
                       duration: float, emit, out_stem: str,
                       level: str, stem_report, guide: dict | None = None
                       ) -> tuple[dict, precision.PrecisionResult]:
    """Transcribe por consenso de varias pasadas y escribe las mismas salidas."""
    spec = PRECISION_LEVELS.get(level, PRECISION_LEVELS["balanced"])
    config = precision.PrecisionConfig(
        model=model, language=language or None,
        min_passes=spec["min_passes"], max_passes=spec["max_passes"],
        stability_target=spec["stability"],
        prompt_spans=track_guides.prompt_spans(guide),
        segment_refiner=(
            (lambda segments: track_guides.refine_segments(segments, guide))
            if guide else None
        ),
    )
    result = precision.run(audio, config, emit, stem_report)
    if not result.segments:
        raise RuntimeError(
            "No se encontró voz cantada en el audio. Revisa la pista de voz "
            "aislada: puede que la canción sea instrumental."
        )

    if guide:
        references = dict(guide.get("references") or {})
        references["matchedLines"] = sum(
            1 for segment in result.segments if segment.get("referenceScore")
        )
        result.reference = references
        emit("reference", stage="aligned", section=guide.get("section"),
             **references)

    payload = {
        "text": "\n".join(segment["text"] for segment in result.segments),
        "segments": result.segments,
        "language": result.language or language or "",
    }
    target = safe_stem(out_stem, fallback=audio.stem)
    files = write_whisper_outputs(payload, outdir, target)
    return files, result


def group_by_language(languages: list[str | None],
                      fallback: str | None) -> dict[str | None, list[int]]:
    """Line indices grouped by the language they were sung in."""
    groups: dict[str | None, list[int]] = {}
    for index, code in enumerate(languages):
        key = translation.normalize_code(code) or translation.normalize_code(fallback)
        groups.setdefault(key, []).append(index)
    return groups


def translate_clip_lines(lines: list[str], languages: list[str | None], *,
                         source: str | None, target: str,
                         romanize: bool, emit) -> dict:
    """Romaniza y traduce las líneas ya confirmadas por el consenso.

    Se hace al final a propósito: traducir un texto que todavía puede cambiar
    es trabajo tirado, y una traducción hecha sobre texto inventado se ve
    perfectamente correcta, que es la peor forma de estar mal.

    Las líneas se agrupan por el idioma en el que se cantaron, no por el del
    archivo: en un mix de una hora conviven canciones en japonés y en inglés, y
    cada grupo necesita su propia ruta (ja→en→es, en→es) y su propio romaji.
    """
    out: dict = {"lines": None, "romaji": None, "backend": None,
                 "routeLabel": None, "targetLanguage": None, "error": None,
                 "routes": []}
    if not lines:
        return out

    groups = group_by_language(languages, source)
    errors: list[str] = []
    routes: list[str] = []

    # --- Romaji: sólo los tramos que realmente se cantaron en japonés ---
    romaji_lines: list[str] | None = None
    if romanize:
        for code, indices in groups.items():
            if code not in translation.ROMANIZABLE:
                continue
            emit("status", stage="romanizing",
                 message=(f"Pasando a romaji {len(indices)} líneas en "
                          f"{translation.language_label(code)}…"))
            try:
                converted, backend = translation.romanize(
                    [lines[i] for i in indices],
                    progress=lambda done, total: emit(
                        "progress", stage="romanizing",
                        progress=done / max(total, 1)),
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"No se pudo romanizar: {error}")
                continue
            if romaji_lines is None:
                romaji_lines = list(lines)
            for position, index in enumerate(indices):
                romaji_lines[index] = converted[position]
            out["romajiBackend"] = backend
        out["romaji"] = romaji_lines

    if not target:
        out["error"] = "; ".join(errors) or None
        return out

    # --- Traducción: una ruta por idioma de origen ---
    translated = list(lines)
    did_translate = False
    for code, indices in groups.items():
        route = translation.plan_route(code, target)
        if not route:
            continue                    # ya está en el idioma de destino
        label = translation.route_label(route)
        emit("status", stage="translating",
             message=(f"Traduciendo {len(indices)} líneas ya confirmadas · "
                      f"{label}…"))
        try:
            result = translation.translate(
                [lines[i] for i in indices], source=code, target=target,
                progress=lambda p: emit("progress", stage="translating",
                                        progress=p),
            )
        except translation.TranslationUnavailable as error:
            out["error"] = str(error)
            return out
        except Exception as error:  # noqa: BLE001
            errors.append(f"{label}: {error}")
            continue
        for position, index in enumerate(indices):
            translated[index] = result["lines"][position]
        routes.append(f"{label} ({len(indices)})")
        out["backend"] = result["backend"]
        did_translate = True

    out["error"] = "; ".join(errors) or None
    if not did_translate:
        return out
    out["lines"] = translated
    out["routes"] = routes
    out["routeLabel"] = " · ".join(routes)
    out["targetLanguage"] = translation.normalize_code(target)
    return out


def read_clip_lines(json_path: Path) -> tuple[list[str], list[str | None]]:
    """Las líneas exactas que serán clips, con el idioma en que se cantó cada una."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [], []
    return clip_lines(data), clip_languages(data)


def write_lyrics_sheet(outdir: Path, title: str, original: list[str],
                       romaji: list[str] | None,
                       translated: list[str] | None) -> str | None:
    """Hoja de letra legible: original, romaji y traducción una debajo de otra.

    El `.lyrixa.json` es para Lyrixa; esto es para leerlo. Sin ella, comprobar
    una traducción obliga a abrir un JSON de varios megas.
    """
    if not original or not (romaji or translated):
        return None
    blocks: list[str] = [f"# {title}", ""]
    for index, line in enumerate(original):
        blocks.append(line)
        if romaji and index < len(romaji) and romaji[index] != line:
            blocks.append(f"    {romaji[index]}")
        if translated and index < len(translated):
            blocks.append(f"    → {translated[index]}")
        blocks.append("")
    target = outdir / f"{safe_stem(title)}.letra.txt"
    target.write_text("\n".join(blocks), encoding="utf-8")
    return target.name


def build_lyrixa_bundle(
    outdir: Path, json_name: str | None, title: str, duration: float,
    audio_file_name: str | None = None, model: str | None = None,
    language: str | None = None, translations: list[str] | None = None,
    translation_language: str | None = None, romaji: list[str] | None = None,
    precision_report: dict | None = None,
) -> str | None:
    """Convierte el JSON de Whisper en un proyecto `.lyrixa.json` importable.

    El normalizador puro vive en ``lyrixa_bridge``. Este adaptador se limita a
    leer el JSON raw de Whisper y escribir el envelope de proyecto de Lyrixa;
    nunca modifica ni reemplaza la salida original de Whisper.
    """
    if not json_name:
        return None
    try:
        data = json.loads((outdir / json_name).read_text(encoding="utf-8"))
    except Exception:
        return None

    envelope = build_lyrixa_envelope(
        data,
        title=title,
        duration=duration,
        audio_file_name=audio_file_name,
        model=model,
        language=language,
        project_stem=Path(json_name).stem,
        translations=translations,
        translation_language=translation_language,
        romaji=romaji,
        precision=precision_report,
    )
    out = outdir / f"{Path(json_name).stem}.lyrixa.json"
    out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out.name


def pipeline(job_id: str, source_type: str, url: str, upload_path: Path | None,
             model: str, language: str, mode: str = "transcribe",
             audio_format: str = "best", audio_quality: str = "192K",
             separation_quality: str = "precise",
             precision_level: str = "balanced", translate_to: str = "",
             romaji: bool = True, tracklist_text: str = "",
             fetch_lyrics: bool = True) -> None:
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
    update_manifest(job_id, status="running", error=None)
    try:
        # Modo "solo audio": descargar en el formato pedido y terminar, sin Whisper.
        if mode == "audio_only":
            audio, title, duration = download_audio(
                url, outdir, emit, audio_format, audio_quality)
            emit("meta", title=title, duration=duration, device=device_name())
            files = {"audio": audio.name}
            update_manifest(job_id, status="completed", title=title,
                            duration=duration, files=files)
            emit("done", title=title, files=files,
                 job_id=job_id)
            return

        separates = mode in {"lyrics", "ultra"}
        if separates and not separation_profile_available(separation_quality):
            profile = SEPARATION_PROFILES[separation_quality]
            if profile["engine"] == "roformer":
                raise RuntimeError(
                    "BS-RoFormer no está instalado. Ejecuta: cd webapp && "
                    "./setup_separator.sh"
                )
            raise RuntimeError(
                "Demucs no está instalado. Instálalo con: pip3 install demucs"
            )
        if mode == "ultra" and translate_to and not translation.translation_available():
            raise RuntimeError(
                "Pediste traducción pero no hay traductor local. Arranca Ollama "
                "y baja un modelo multilingüe: ollama pull qwen3.5"
            )

        # El modo ultra usa faster-whisper con sus propios pesos; los .pt de
        # ~/.cache/whisper solo hacen falta cuando corre el motor de referencia.
        if mode != "ultra":
            ensure_model(model, emit)

        source_audio = None
        if source_type == "url":
            # Para letra conviene el audio sin recodificar: mejor separación de voz.
            fmt = "best" if separates else audio_format
            source_audio, title, duration = download_audio(
                url, outdir, emit, fmt, audio_quality)
            audio = source_audio
        else:
            audio = upload_path
            title = audio.stem          # sin extensión: es el nombre de las salidas
            duration = ffprobe_duration(audio)
        original_audio = audio

        emit("meta", title=title, duration=duration,
             model=model, device=device_name())
        update_manifest(job_id, title=title, duration=duration,
                        sourceFile=original_audio.name)

        guide = None
        if mode == "ultra" and tracklist_text.strip():
            guide = prepare_track_guide(
                outdir, tracklist_text, duration, fetch_lyrics, emit)

        stem_files = {}
        stem_report = None
        if separates:
            vocals, instrumental, stem_report = separate_vocals(
                audio, outdir, emit, separation_quality, duration)
            audio = vocals
            job["stem_paths"] = {
                "vocals": str(vocals.resolve()),
                "instrumental": str(instrumental.resolve()),
            }
            emit("stem", stage="final", pass_number=0, **stem_report.as_dict())
            # Los WAV se sirven desde la carpeta del separador elegido, sin
            # copiarlos ni recomprimirlos. En mixes largos pueden pesar >600 MB.
            stem_files = {
                "vocals": {
                    "name": f"{safe_stem(title)} - voz aislada.wav",
                    "url": f"/api/jobs/{job_id}/stems/vocals",
                },
                "instrumental": {
                    "name": f"{safe_stem(title)} - instrumental.wav",
                    "url": f"/api/jobs/{job_id}/stems/instrumental",
                },
            }

        precision_report = None
        detected_language = language
        if mode == "ultra":
            # Nivelar y filtrar el grave antes de transcribir: no cambia lo que
            # se canta, pero sí lo estable que es el umbral de Whisper.
            audio = audio_analysis.condition_for_asr(
                audio, outdir / "asr-input.wav")
            files, result = run_precision_mode(
                audio, model, language, outdir, duration, emit,
                out_stem=title, level=precision_level, stem_report=stem_report,
                guide=guide)
            precision_report = result.as_report()
            detected_language = result.language or language
            emit("precision", **precision_report)
        else:
            files = run_whisper(audio, model, language, outdir, duration, emit,
                                out_stem=title)

        # Traducción y romaji: sólo después de que la letra ya está decidida.
        translated_lines = None
        romaji_lines = None
        translation_language = None
        if mode == "ultra" and files.get("json"):
            lines, line_languages = read_clip_lines(outdir / files["json"])
            language_report = translate_clip_lines(
                lines, line_languages, source=detected_language,
                target=translate_to, romanize=romaji, emit=emit)
            translated_lines = language_report["lines"]
            romaji_lines = language_report["romaji"]
            translation_language = language_report["targetLanguage"]
            emit("translation",
                 routeLabel=language_report["routeLabel"],
                 backend=language_report["backend"],
                 romajiBackend=language_report.get("romajiBackend"),
                 targetLanguage=translation_language,
                 error=language_report["error"])
            lyrics_file = write_lyrics_sheet(
                outdir, title, lines, romaji_lines, translated_lines)
            if lyrics_file:
                files["lyrics"] = lyrics_file

        # Puente con Lyrixa: proyecto con clips ya cronometrados.
        lyrixa = build_lyrixa_bundle(
            outdir, files.get("json"), title, duration,
            audio_file_name=original_audio.name, model=model,
            language=detected_language, translations=translated_lines,
            translation_language=translation_language, romaji=romaji_lines,
            precision_report=precision_report,
        )
        if lyrixa:
            files["lyrixa"] = lyrixa
        # El audio descargado (con sus metadatos y carátula) también se ofrece.
        if source_audio is not None and source_audio.exists():
            files["audio"] = source_audio.name
        if guide:
            files["guide"] = "track-guide.json"
        files.update(stem_files)
        update_manifest(job_id, status="completed", title=title,
                        duration=duration, files=files, error=None)
        emit("done", title=title, files=files, job_id=job_id)
    except Exception as e:  # noqa: BLE001
        update_manifest(job_id, status="error", error=str(e))
        emit("error", message=str(e))
    finally:
        if caffeinate is not None:
            caffeinate.terminate()  # deja que la Mac vuelva a poder dormir
        with job["cond"]:
            job["done"] = True  # marca fin del stream (para todos los lectores)
            job["cond"].notify_all()


def resume_precision_pipeline(
    job_id: str, model: str, language: str, precision_level: str,
    translate_to: str, romaji: bool, tracklist_text: str = "",
    fetch_lyrics: bool = True,
) -> None:
    """Continúa un modo ultra desde los WAV que sobrevivieron al reinicio."""
    job = JOBS[job_id]

    def emit(event: str, **data):
        with job["cond"]:
            job["events"].append({"event": event, "data": data})
            job["cond"].notify_all()

    caffeinate = start_caffeinate()
    update_manifest(job_id, status="running", error=None)
    try:
        assets = recovery_assets(job_id)
        if not assets or not assets["checkpoint"]:
            raise RuntimeError("No quedó un audio intermedio que se pueda retomar.")

        outdir: Path = assets["outdir"]
        title = assets["title"]
        duration = assets["duration"]
        original: Path | None = assets["original"]
        vocals: Path | None = assets["vocals"]
        instrumental: Path | None = assets["instrumental"]
        audio: Path = assets["checkpoint"]

        emit("status", stage="recovering",
             message="Retomando desde el audio ya separado; no repetiré BS-RoFormer…")
        emit("meta", title=title, duration=duration,
             model=model, device=device_name())

        guide = prepare_track_guide(
            outdir, tracklist_text, duration, fetch_lyrics, emit)

        stem_files: dict = {}
        stem_report = None
        if vocals:
            if assets["asr"] is None:
                audio = audio_analysis.condition_for_asr(
                    vocals, outdir / "asr-input.wav")
            stem_report = measure_stems(vocals, instrumental, duration)
            emit("stem", stage="recovered", pass_number=0,
                 **stem_report.as_dict())
            job["stem_paths"] = {"vocals": str(vocals.resolve())}
            stem_files["vocals"] = {
                "name": f"{safe_stem(title)} - voz aislada.wav",
                "url": f"/api/jobs/{job_id}/stems/vocals",
            }
            if instrumental:
                job["stem_paths"]["instrumental"] = str(instrumental.resolve())
                stem_files["instrumental"] = {
                    "name": f"{safe_stem(title)} - instrumental.wav",
                    "url": f"/api/jobs/{job_id}/stems/instrumental",
                }

        files, result = run_precision_mode(
            audio, model, language, outdir, duration, emit,
            out_stem=title, level=precision_level, stem_report=stem_report,
            guide=guide)
        precision_report = result.as_report()
        detected_language = result.language or language
        emit("precision", **precision_report)

        translated_lines = None
        romaji_lines = None
        translation_language = None
        if files.get("json"):
            lines, line_languages = read_clip_lines(outdir / files["json"])
            language_report = translate_clip_lines(
                lines, line_languages, source=detected_language,
                target=translate_to, romanize=romaji, emit=emit)
            translated_lines = language_report["lines"]
            romaji_lines = language_report["romaji"]
            translation_language = language_report["targetLanguage"]
            emit("translation",
                 routeLabel=language_report["routeLabel"],
                 backend=language_report["backend"],
                 romajiBackend=language_report.get("romajiBackend"),
                 targetLanguage=translation_language,
                 error=language_report["error"])
            lyrics_file = write_lyrics_sheet(
                outdir, title, lines, romaji_lines, translated_lines)
            if lyrics_file:
                files["lyrics"] = lyrics_file

        lyrixa = build_lyrixa_bundle(
            outdir, files.get("json"), title, duration,
            audio_file_name=(original.name if original else audio.name),
            model=model, language=detected_language,
            translations=translated_lines,
            translation_language=translation_language, romaji=romaji_lines,
            precision_report=precision_report,
        )
        if lyrixa:
            files["lyrixa"] = lyrixa
        if original and original.exists():
            files["audio"] = original.name
        if guide:
            files["guide"] = "track-guide.json"
        files.update(stem_files)
        update_manifest(job_id, status="completed", title=title,
                        duration=duration, files=files, error=None)
        emit("done", title=title, files=files, job_id=job_id)
    except Exception as error:  # noqa: BLE001
        update_manifest(job_id, status="error", error=str(error))
        emit("error", message=str(error))
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
        with job["cond"]:
            job["done"] = True
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
        # Los dos motores guardan pesos distintos: los .pt de openai-whisper y
        # la conversión CTranslate2 que usa el modo ultra. Un modelo puede
        # estar en caché para uno y no para el otro.
        "models": [{"name": m, "cached": model_is_cached(m),
                    "ultra_cached": precision.te.faster_model_cached(m)}
                   for m in MODELS],
        "languages": [{"value": v, "label": l} for v, l in LANGUAGES],
        "audio_formats": AUDIO_FORMATS,
        "audio_qualities": AUDIO_QUALITIES,
        "separation_profiles": [
            {"value": value, "label": spec["label"], "hint": spec["hint"],
             "available": separation_profile_available(value)}
            for value, spec in SEPARATION_PROFILES.items()
        ],
        "precision_levels": [
            {"value": value, "label": spec["label"], "hint": spec["hint"],
             "min_passes": spec["min_passes"], "max_passes": spec["max_passes"]}
            for value, spec in PRECISION_LEVELS.items()
        ],
        "target_languages": [
            {"value": code, "label": translation.language_label(code)}
            for code in TARGET_LANGUAGES
        ],
        "demucs": demucs_available(),
        "roformer": roformer_available(),
        "faster_whisper": precision.te.faster_whisper_available(),
        "translator": {
            "available": translation.translation_available(),
            "model": translation.pick_model(),
        },
    })


@app.get("/api/jobs")
def list_saved_jobs():
    jobs = []
    for outdir in OUTPUT_DIR.iterdir():
        if not outdir.is_dir() or not _JOB_ID_RE.fullmatch(outdir.name):
            continue
        record = saved_job(outdir.name)
        if record:
            jobs.append(record)
    jobs.sort(key=lambda item: item["updatedAt"], reverse=True)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def saved_job_detail(job_id: str):
    record = saved_job(job_id)
    if record is None:
        return JSONResponse({"error": "job no existe"}, status_code=404)
    tracklist = OUTPUT_DIR / job_id / "tracklist.txt"
    record["tracklistText"] = (
        tracklist.read_text(encoding="utf-8") if tracklist.is_file() else ""
    )
    return record


@app.post("/api/tracklists/preview")
async def preview_tracklist(
    tracklist_text: str = Form(...),
    duration: float = Form(0.0),
):
    sections = track_guides.parse_tracklist(tracklist_text)
    selected = track_guides.choose_section(sections, duration)
    if not selected:
        return JSONResponse(
            {"error": "No encontré líneas con formato 00:00 Artista - Título."},
            status_code=400,
        )
    return {
        "selected": selected.name,
        "sections": [
            {
                "name": section.name,
                "tracks": len(section.tracks),
                "first": section.tracks[0].start if section.tracks else 0,
                "last": section.tracks[-1].start if section.tracks else 0,
            }
            for section in sections
        ],
        "tracks": [
            {"start": track.start, "artist": track.artist,
             "title": track.title, "label": track.label}
            for track in selected.tracks
        ],
    }


@app.post("/api/jobs")
async def create_job(
    source_type: str = Form(...),
    model: str = Form("medium"),
    language: str = Form(""),
    url: str = Form(""),
    mode: str = Form("transcribe"),
    audio_format: str = Form("best"),
    audio_quality: str = Form("192K"),
    separation_quality: str = Form("precise"),
    precision_level: str = Form("balanced"),
    translate_to: str = Form(""),
    romaji: str = Form("1"),
    tracklist_text: str = Form(""),
    fetch_lyrics: str = Form("1"),
    file: UploadFile | None = File(None),
):
    if audio_format not in _FORMATS_BY_VALUE:
        return JSONResponse({"error": f"Formato desconocido: {audio_format}"},
                            status_code=400)
    if audio_quality not in AUDIO_QUALITIES:
        audio_quality = "192K"
    if separation_quality not in SEPARATION_PROFILES:
        separation_quality = "precise"
    if precision_level not in PRECISION_LEVELS:
        precision_level = "balanced"
    translate_to = translation.normalize_code(translate_to) or ""
    if translate_to and translate_to not in TARGET_LANGUAGES:
        return JSONResponse(
            {"error": f"Idioma de traducción desconocido: {translate_to}"},
            status_code=400)

    if source_type not in {"url", "file"}:
        return JSONResponse({"error": "Fuente desconocida"}, status_code=400)
    if source_type == "file" and file is None:
        return JSONResponse({"error": "Falta el archivo"}, status_code=400)
    if len(tracklist_text.encode("utf-8")) > 2_000_000:
        return JSONResponse({"error": "El tracklist supera 2 MB."}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    upload_path = None
    outdir = OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    if source_type == "file":
        # Nunca confiamos en el nombre que manda el navegador para armar rutas.
        upload_path = outdir / Path(file.filename or "audio").name
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    if tracklist_text.strip():
        (outdir / "tracklist.txt").write_text(
            tracklist_text.strip() + "\n", encoding="utf-8")
    update_manifest(
        job_id, status="queued", mode=mode, iteration=0,
        settings={
            "model": model, "language": language,
            "separationQuality": separation_quality,
            "precisionLevel": precision_level, "translateTo": translate_to,
            "romaji": romaji not in {"", "0", "false"},
            "hasTracklist": bool(tracklist_text.strip()),
            "fetchLyrics": fetch_lyrics not in {"", "0", "false"},
        },
    )

    JOBS[job_id] = {
        "events": [],                 # buffer de todos los eventos (para reconexión)
        "cond": threading.Condition(),
        "done": False,
    }
    threading.Thread(
        target=pipeline,
        args=(job_id, source_type, url, upload_path, model, language, mode,
              audio_format, audio_quality, separation_quality, precision_level,
              translate_to, romaji not in {"", "0", "false"},
              tracklist_text, fetch_lyrics not in {"", "0", "false"}),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/state")
def job_state(job_id: str):
    """Indica si el job sigue vivo o si quedó un checkpoint recuperable."""
    job = JOBS.get(job_id)
    if job is not None:
        return {"status": "known", "active": not job["done"],
                "recoverable": False}

    assets = recovery_assets(job_id)
    if assets is None:
        return JSONResponse({"error": "job no existe"}, status_code=404)
    saved = [
        path.name for key in ("original", "vocals", "instrumental", "asr")
        if (path := assets.get(key)) is not None
    ]
    return {
        "status": "interrupted",
        "active": False,
        "recoverable": assets["checkpoint"] is not None,
        "checkpoint": "asr" if assets["asr"] else "vocals" if assets["vocals"] else None,
        "title": assets["title"],
        "duration": assets["duration"],
        "saved": saved,
    }


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(
    job_id: str,
    model: str = Form("small"),
    language: str = Form(""),
    precision_level: str = Form("balanced"),
    translate_to: str = Form(""),
    romaji: str = Form("1"),
    tracklist_text: str = Form(""),
    fetch_lyrics: str = Form("1"),
):
    current = JOBS.get(job_id)
    if current is not None and not current["done"]:
        return JSONResponse({"error": "El trabajo ya está corriendo."},
                            status_code=409)
    assets = recovery_assets(job_id)
    if assets is None or assets["checkpoint"] is None:
        return JSONResponse(
            {"error": "No quedó un audio intermedio recuperable."},
            status_code=404,
        )
    if model not in MODELS:
        model = "small"
    if precision_level not in PRECISION_LEVELS:
        precision_level = "balanced"
    translate_to = translation.normalize_code(translate_to) or ""
    if translate_to and translate_to not in TARGET_LANGUAGES:
        return JSONResponse(
            {"error": f"Idioma de traducción desconocido: {translate_to}"},
            status_code=400,
        )
    if translate_to and not translation.translation_available():
        return JSONResponse(
            {"error": "No hay traductor local disponible."}, status_code=409)
    if len(tracklist_text.encode("utf-8")) > 2_000_000:
        return JSONResponse({"error": "El tracklist supera 2 MB."}, status_code=400)

    iteration = _archive_outputs(assets["outdir"])
    if tracklist_text.strip():
        (assets["outdir"] / "tracklist.txt").write_text(
            tracklist_text.strip() + "\n", encoding="utf-8")
        # A new guide must be rebuilt; the previous one belongs to another
        # version/section of the pasted list.
        (assets["outdir"] / "track-guide.json").unlink(missing_ok=True)
    update_manifest(
        job_id, status="queued", iteration=iteration,
        settings={
            "model": model, "language": language,
            "precisionLevel": precision_level, "translateTo": translate_to,
            "romaji": romaji not in {"", "0", "false"},
            "hasTracklist": (assets["outdir"] / "tracklist.txt").is_file(),
            "fetchLyrics": fetch_lyrics not in {"", "0", "false"},
        },
    )

    JOBS[job_id] = {
        "events": [], "cond": threading.Condition(), "done": False,
    }
    threading.Thread(
        target=resume_precision_pipeline,
        args=(job_id, model, language, precision_level, translate_to,
              romaji not in {"", "0", "false"}, tracklist_text,
              fetch_lyrics not in {"", "0", "false"}),
        daemon=True,
    ).start()
    return {"job_id": job_id, "resumed": True}


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
    # El nombre viene de la URL: nos quedamos solo con el archivo y verificamos
    # que el resultado siga dentro de la carpeta del job (nada de "../").
    job_dir = (OUTPUT_DIR / job_id).resolve()
    path = (job_dir / Path(name).name).resolve()
    if not path.is_file() or job_dir not in path.parents:
        return JSONResponse({"error": "no existe"}, status_code=404)
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs/{job_id}/stems/{stem}")
def download_stem(job_id: str, stem: str):
    """Sirve los WAV del separador elegido sin duplicar archivos grandes."""
    names = {
        "vocals": ("vocals.wav",),
        "instrumental": ("instrumental.wav", "no_vocals.wav"),
    }
    target_names = names.get(stem)
    if target_names is None:
        return JSONResponse({"error": "pista desconocida"}, status_code=404)
    job_dir = (OUTPUT_DIR / job_id).resolve()
    matches = []
    job = JOBS.get(job_id)
    stored = (job or {}).get("stem_paths", {}).get(stem)
    if stored:
        matches.append(Path(stored).resolve())
    if not matches:
        for target_name in target_names:
            matches.extend(path.resolve() for path in job_dir.rglob(target_name))
    matches = [path for path in matches if path.is_file() and job_dir in path.parents]
    if len(matches) != 1:
        return JSONResponse({"error": "pista no disponible"}, status_code=404)
    title = next(
        (path.stem for path in job_dir.iterdir()
         if path.is_file() and path.suffix.lower() in {".mp3", ".m4a", ".opus", ".ogg"}),
        "audio",
    )
    label = "voz aislada" if stem == "vocals" else "instrumental"
    return FileResponse(matches[0], filename=f"{safe_stem(title)} - {label}.wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
