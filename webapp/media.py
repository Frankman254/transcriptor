"""Traer el audio y limpiarlo: descarga, medición y separación de voces.

Todo lo que habla con un binario externo — yt-dlp, ffprobe, Demucs,
BS-RoFormer — vive aquí.  Es la capa más frágil del backend, porque depende
de programas que el usuario puede no tener instalados, y por eso está
separada: cada `*_available()` responde sin reventar, y el resto del motor
puede preguntar antes de prometerle nada al usuario.

Las funciones largas reciben un `emit` en vez de escribir en el bus de
eventos, así que no saben que existe un servidor HTTP.  Eso es lo que permite
llamarlas desde una CLI, desde una prueba o desde una shell de escritorio.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "demucs_available", "download_audio", "ffprobe_duration", "measure_stems",
    "roformer_available", "separate_vocals", "separation_profile_available",
    "separator_acceleration",
]

import json
import os
import re
import subprocess
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import audio_analysis
import paths
from catalogs import SEPARATION_PROFILES, FORMATS_BY_VALUE


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
    spec = FORMATS_BY_VALUE.get(audio_format) or FORMATS_BY_VALUE["best"]

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
    return paths.ROFORMER_CLI.is_file() and os.access(paths.ROFORMER_CLI, os.X_OK)


@lru_cache(maxsize=1)
def separator_acceleration() -> dict:
    """Expose what audio-separator actually configured, not what we expect."""
    if not roformer_available():
        return {"mode": "unavailable", "accelerated": False}
    try:
        proc = subprocess.run(
            [str(paths.ROFORMER_CLI), "--env_info"], capture_output=True, text=True,
            timeout=12,
        )
        report = f"{proc.stdout}\n{proc.stderr}"
    except Exception as error:  # noqa: BLE001
        return {"mode": "unknown", "accelerated": False,
                "detail": str(error)}
    if "CoreMLExecutionProvider" in report:
        return {"mode": "CoreML", "accelerated": True}
    if "CUDAExecutionProvider" in report:
        return {"mode": "CUDA", "accelerated": True}
    if "No hardware acceleration" in report:
        return {"mode": "CPU", "accelerated": False}
    return {"mode": "unknown", "accelerated": False}


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

    model_cached = (paths.ROFORMER_MODEL_DIR / paths.ROFORMER_MODEL).is_file()
    if model_cached:
        emit("status", stage="separating",
             message="Separando la voz con BS-RoFormer (puede tardar)…")
    else:
        emit("status", stage="downloading_separator_model",
             message="Descargando BS-RoFormer (~610 MB, sólo esta vez)…")
    emit("progress", stage="separating", progress=0.0)

    sep_dir = outdir / tag
    sep_dir.mkdir(parents=True, exist_ok=True)
    paths.ROFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(paths.ROFORMER_CLI), str(audio),
        "--model_filename", paths.ROFORMER_MODEL,
        "--model_file_dir", str(paths.ROFORMER_MODEL_DIR),
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
