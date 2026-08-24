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
from pathlib import Path

import whisper  # solo para conocer las URLs de los modelos
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from lyrixa_bridge import build_lyrixa_envelope

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = Path.home() / ".cache" / "whisper"
ROFORMER_VENV_DIR = BASE_DIR / ".separator-venv"
ROFORMER_CLI = ROFORMER_VENV_DIR / "bin" / "audio-separator"
ROFORMER_MODEL_DIR = BASE_DIR / ".separator-models"
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "precise": {
        "label": "Preciso · BS-RoFormer (recomendado)",
        "hint": ("Modelo moderno que recupera mejor voces débiles y coros. "
                 "La primera vez descarga ~610 MB; después queda en caché."),
        "engine": "roformer",
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
    audio: Path, outdir: Path, emit,
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

    sep_dir = outdir / "roformer"
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


def separate_vocals(
    audio: Path, outdir: Path, emit, profile_name: str = "precise",
) -> tuple[Path, Path]:
    profile = SEPARATION_PROFILES.get(profile_name, SEPARATION_PROFILES["precise"])
    if profile["engine"] == "roformer":
        return separate_vocals_roformer(audio, outdir, emit)
    return separate_vocals_demucs(audio, outdir, emit, profile_name)


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


def build_lyrixa_bundle(
    outdir: Path, json_name: str | None, title: str, duration: float,
    audio_file_name: str | None = None, model: str | None = None,
    language: str | None = None,
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
    )
    out = outdir / f"{Path(json_name).stem}.lyrixa.json"
    out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out.name


def pipeline(job_id: str, source_type: str, url: str, upload_path: Path | None,
             model: str, language: str, mode: str = "transcribe",
             audio_format: str = "best", audio_quality: str = "192K",
             separation_quality: str = "precise") -> None:
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
        # Modo "solo audio": descargar en el formato pedido y terminar, sin Whisper.
        if mode == "audio_only":
            audio, title, duration = download_audio(
                url, outdir, emit, audio_format, audio_quality)
            emit("meta", title=title, duration=duration, device=device_name())
            emit("done", title=title, files={"audio": audio.name},
                 job_id=job_id)
            return

        if mode == "lyrics" and not separation_profile_available(separation_quality):
            profile = SEPARATION_PROFILES[separation_quality]
            if profile["engine"] == "roformer":
                raise RuntimeError(
                    "BS-RoFormer no está instalado. Ejecuta: cd webapp && "
                    "./setup_separator.sh"
                )
            raise RuntimeError(
                "Demucs no está instalado. Instálalo con: pip3 install demucs"
            )

        ensure_model(model, emit)

        source_audio = None
        if source_type == "url":
            # Para letra conviene el audio sin recodificar: mejor separación de voz.
            fmt = "best" if mode == "lyrics" else audio_format
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

        stem_files = {}
        if mode == "lyrics":
            vocals, instrumental = separate_vocals(
                audio, outdir, emit, separation_quality)
            audio = vocals
            job["stem_paths"] = {
                "vocals": str(vocals.resolve()),
                "instrumental": str(instrumental.resolve()),
            }
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

        files = run_whisper(audio, model, language, outdir, duration, emit,
                            out_stem=title)
        # Puente con Lyrixa: proyecto con clips ya cronometrados.
        lyrixa = build_lyrixa_bundle(
            outdir, files.get("json"), title, duration,
            audio_file_name=original_audio.name, model=model, language=language,
        )
        if lyrixa:
            files["lyrixa"] = lyrixa
        # El audio descargado (con sus metadatos y carátula) también se ofrece.
        if source_audio is not None and source_audio.exists():
            files["audio"] = source_audio.name
        files.update(stem_files)
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
        "audio_formats": AUDIO_FORMATS,
        "audio_qualities": AUDIO_QUALITIES,
        "separation_profiles": [
            {"value": value, "label": spec["label"], "hint": spec["hint"],
             "available": separation_profile_available(value)}
            for value, spec in SEPARATION_PROFILES.items()
        ],
        "demucs": demucs_available(),
        "roformer": roformer_available(),
    })


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
    file: UploadFile | None = File(None),
):
    if audio_format not in _FORMATS_BY_VALUE:
        return JSONResponse({"error": f"Formato desconocido: {audio_format}"},
                            status_code=400)
    if audio_quality not in AUDIO_QUALITIES:
        audio_quality = "192K"
    if separation_quality not in SEPARATION_PROFILES:
        separation_quality = "precise"

    job_id = uuid.uuid4().hex[:12]
    upload_path = None
    if source_type == "file":
        if file is None:
            return JSONResponse({"error": "Falta el archivo"}, status_code=400)
        outdir = OUTPUT_DIR / job_id
        outdir.mkdir(parents=True, exist_ok=True)
        # Nunca confiamos en el nombre que manda el navegador para armar rutas.
        upload_path = outdir / Path(file.filename or "audio").name
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "events": [],                 # buffer de todos los eventos (para reconexión)
        "cond": threading.Condition(),
        "done": False,
    }
    threading.Thread(
        target=pipeline,
        args=(job_id, source_type, url, upload_path, model, language, mode,
              audio_format, audio_quality, separation_quality),
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
