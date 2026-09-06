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
import threading
import time
import uuid
from pathlib import Path

import whisper  # solo para conocer las URLs de los modelos
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

import audio_analysis
import config
import jobstore
import lyrixa_bridge
import media
import paths
import precision
import track_guides
import translation
from lyrixa_bridge import build_lyrixa_envelope, clip_languages, clip_lines
from catalogs import (
    ASR_ENGINES,
    AUDIO_FORMATS,
    AUDIO_QUALITIES,
    LANGUAGES,
    MODEL_ARTIFACTS,
    MODEL_DETAILS,
    MODELS,
    PRECISION_LEVELS,
    SEPARATION_PROFILES,
    TARGET_LANGUAGES,
    _FORMATS_BY_VALUE,
)


app = FastAPI(title="Transcriptor local")

# Transcriptor es el motor; Lyrixa y el renderer son clientes que corren en
# sus propios puertos, así que sin CORS no pueden llamarlo desde el navegador.
# El permiso es solo para localhost (ver `config`): el servidor escucha en
# loopback, de modo que la única forma de que una página remota llegase aquí
# sería a través del navegador del usuario, y ese es justo el caso que se
# niega. Sin credenciales: esta API no usa cookies ni sesiones.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_origin_regex=config.ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# job_id -> dict(queue, done, ...)
JOBS: dict[str, dict] = {}


def job_is_running(job_id: str) -> bool:
    """Si hay un pipeline vivo para este trabajo en *este* proceso."""
    live = JOBS.get(job_id)
    return bool(live and not live["done"])
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



def run_whisper(audio: Path, model: str, language: str, outdir: Path,
                duration: float, emit, out_stem: str | None = None) -> dict:
    """Corre Whisper por CLI en modo verbose y transmite cada segmento.

    `out_stem` es el nombre con el que queremos que salgan los archivos. Whisper
    los nombra según el audio que le damos, que puede ser el id de YouTube o el
    `vocals.wav` de Demucs; aquí los renombramos al título de verdad.
    """
    cmd = ["whisper", str(audio), "--model", model,
           "--output_format", "all", "--output_dir", str(outdir),
           "--verbose", "True", "--model_dir", str(paths.CACHE_DIR),
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
                       level: str, stem_report, guide: dict | None = None,
                       asr_engine: str = "auto",
                       rescue_audio: Path | None = None,
                       ) -> tuple[dict, precision.PrecisionResult]:
    """Transcribe por consenso de varias pasadas y escribe las mismas salidas."""
    spec = PRECISION_LEVELS.get(level, PRECISION_LEVELS["balanced"])
    config = precision.PrecisionConfig(
        model=model, language=language or None,
        min_passes=spec["min_passes"], max_passes=spec["max_passes"],
        stability_target=spec["stability"],
        engine_preference=asr_engine,
        prompt_spans=track_guides.prompt_spans(guide),
        segment_refiner=(
            (lambda segments: track_guides.refine_segments(segments, guide))
            if guide else None
        ),
    )
    result = precision.run(audio, config, emit, stem_report)
    if rescue_audio is not None:
        try:
            result = precision.rescue_from_original(
                rescue_audio, result, config, emit)
            if config.segment_refiner:
                result.segments = config.segment_refiner(result.segments)
        except Exception as error:  # noqa: BLE001
            emit("rescue", reviewed=0, accepted=0, error=str(error))
    if not result.segments:
        if result.windows:
            raise RuntimeError(
                "Whisper no devolvió palabras utilizables aunque el detector "
                f"encontró {result.windows} ventanas con voz. Esto indica un "
                "fallo del motor o de la votación, no una pista instrumental."
            )
        raise RuntimeError(
            "No se detectaron regiones de voz en el audio guardado. Revisa la "
            "pista aislada antes de volver a procesar."
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
             romaji: bool = False, tracklist_text: str = "",
             fetch_lyrics: bool = True, asr_engine: str = "auto",
             rescue_original: bool = True) -> None:
    job = JOBS[job_id]
    _start_job_timing(job)

    def emit(event: str, **data):
        if event == "status":
            _mark_job_stage(job, data.get("stage"))
        # Guardamos cada evento en un buffer (permite reconexión) y avisamos.
        with job["cond"]:
            job["events"].append({"event": event, "data": data})
            job["cond"].notify_all()

    # Mantiene la Mac despierta durante todo el trabajo.
    caffeinate = start_caffeinate()

    outdir = paths.OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    jobstore.update_manifest(job_id, status="running", error=None,
                    startedAt=jobstore.utc_now(), finishedAt=None)
    try:
        # Modo "solo audio": descargar en el formato pedido y terminar, sin Whisper.
        if mode == "audio_only":
            audio, title, duration = media.download_audio(
                url, outdir, emit, audio_format, audio_quality)
            emit("meta", title=title, duration=duration, device=device_name())
            files = {"audio": audio.name}
            jobstore.update_manifest(job_id, status="completed", title=title,
                            duration=duration, files=files,
                            finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job))
            emit("done", title=title, files=files, job_id=job_id,
                 links=jobstore.job_links(job_id, files))
            return

        separates = mode in {"lyrics", "ultra"}
        if separates and not media.separation_profile_available(separation_quality):
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
            source_audio, title, duration = media.download_audio(
                url, outdir, emit, fmt, audio_quality)
            audio = source_audio
        else:
            audio = upload_path
            title = audio.stem          # sin extensión: es el nombre de las salidas
            duration = media.ffprobe_duration(audio)
        original_audio = audio

        emit("meta", title=title, duration=duration,
             model=model, device=device_name())
        jobstore.update_manifest(job_id, title=title, duration=duration,
                        sourceFile=original_audio.name)

        guide = None
        if mode == "ultra" and tracklist_text.strip():
            guide = jobstore.prepare_track_guide(
                outdir, tracklist_text, duration, fetch_lyrics, emit)

        stem_files = {}
        stem_report = None
        if separates:
            vocals, instrumental, stem_report = media.separate_vocals(
                audio, outdir, emit, separation_quality, duration)
            audio = vocals
            job["stem_paths"] = {
                "vocals": str(vocals.resolve()),
                "instrumental": str(instrumental.resolve()),
            }
            emit("stem", stage="final", pass_number=0, **stem_report.as_dict())
            # Persistir el control de calidad permite validar/reutilizar estas
            # pistas en una iteración sin volver a decodificar archivos enormes.
            jobstore.update_manifest(job_id, stem=stem_report.as_dict())
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
                guide=guide, asr_engine=asr_engine,
                rescue_audio=original_audio if rescue_original else None)
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
            language_report = translation.translate_clip_lines(
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
        jobstore.update_manifest(job_id, status="completed", title=title,
                        duration=duration, files=files, error=None,
                        finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job),
                        precision=precision_report)
        emit("done", title=title, files=files, job_id=job_id,
             links=jobstore.job_links(job_id, files))
    except Exception as e:  # noqa: BLE001
        jobstore.update_manifest(job_id, status="error", error=str(e),
                        finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job))
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
    fetch_lyrics: bool = True, asr_engine: str = "auto",
    rescue_original: bool = True,
) -> None:
    """Continúa un modo ultra desde los WAV que sobrevivieron al reinicio."""
    job = JOBS[job_id]
    _start_job_timing(job)

    def emit(event: str, **data):
        if event == "status":
            _mark_job_stage(job, data.get("stage"))
        with job["cond"]:
            job["events"].append({"event": event, "data": data})
            job["cond"].notify_all()

    caffeinate = start_caffeinate()
    jobstore.update_manifest(job_id, status="running", error=None,
                    startedAt=jobstore.utc_now(), finishedAt=None)
    try:
        assets = jobstore.recovery_assets(job_id)
        if not assets or not assets["checkpoint"]:
            raise RuntimeError("No quedó un audio intermedio que se pueda retomar.")

        outdir: Path = assets["outdir"]
        title = assets["title"]
        duration = assets["duration"]
        original: Path | None = assets["original"]
        vocals: Path | None = assets["vocals"]
        instrumental: Path | None = assets["instrumental"]
        audio, checkpoint_kind = jobstore.reusable_checkpoint(assets)
        if audio is None:
            raise RuntimeError(
                "Los audios guardados están incompletos o dañados; no es seguro "
                "retomar la transcripción desde ellos."
            )

        emit("status", stage="recovering",
             message=("Checkpoint validado: "
                      + ("usaré asr-input.wav" if checkpoint_kind == "asr"
                         else "usaré la voz aislada")
                      + "; no repetiré BS-RoFormer…"))
        emit("meta", title=title, duration=duration,
             model=model, device=device_name())

        guide = jobstore.prepare_track_guide(
            outdir, tracklist_text, duration, fetch_lyrics, emit)

        stem_files: dict = {}
        stem_report = None
        if vocals:
            if checkpoint_kind != "asr":
                audio = audio_analysis.condition_for_asr(
                    vocals, outdir / "asr-input.wav")
            stem_report = jobstore.stored_stem_report(outdir)
            if stem_report is None:
                stem_report = media.measure_stems(vocals, instrumental, duration)
                jobstore.update_manifest(job_id, stem=stem_report.as_dict())
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
            guide=guide, asr_engine=asr_engine,
            rescue_audio=original if rescue_original else None)
        precision_report = result.as_report()
        detected_language = result.language or language
        emit("precision", **precision_report)

        translated_lines = None
        romaji_lines = None
        translation_language = None
        if files.get("json"):
            lines, line_languages = read_clip_lines(outdir / files["json"])
            language_report = translation.translate_clip_lines(
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
        jobstore.update_manifest(job_id, status="completed", title=title,
                        duration=duration, files=files, error=None,
                        finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job),
                        precision=precision_report)
        emit("done", title=title, files=files, job_id=job_id,
             links=jobstore.job_links(job_id, files))
    except Exception as error:  # noqa: BLE001
        jobstore.update_manifest(job_id, status="error", error=str(error),
                        finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job))
        emit("error", message=str(error))
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
        with job["cond"]:
            job["done"] = True
            job["cond"].notify_all()


def translate_saved_pipeline(job_id: str, target: str, romanize: bool) -> None:
    """Add or replace language layers without rerunning separation or ASR."""
    job = JOBS[job_id]
    _start_job_timing(job)

    def emit(event: str, **data):
        if event == "status":
            _mark_job_stage(job, data.get("stage"))
        with job["cond"]:
            job["events"].append({"event": event, "data": data})
            job["cond"].notify_all()

    jobstore.update_manifest(job_id, status="running", error=None,
                    startedAt=jobstore.utc_now(), finishedAt=None)
    try:
        assets = jobstore.recovery_assets(job_id)
        if not assets:
            raise RuntimeError("El trabajo guardado ya no existe.")
        outdir: Path = assets["outdir"]
        manifest = jobstore.read_manifest(outdir)
        files = jobstore._discover_result_files(job_id, outdir, assets)
        json_name = files.get("json")
        if not isinstance(json_name, str):
            raise RuntimeError("Este trabajo no conserva el JSON cronometrado.")
        lines, languages = read_clip_lines(outdir / json_name)
        precision_report = manifest.get("precision") or {}
        settings = dict(manifest.get("settings") or {})
        source_language = (
            precision_report.get("language") or settings.get("language") or ""
        )
        emit("status", stage="translating",
             message="Añadiendo capas al resultado guardado; no repetiré Whisper…")
        report = translation.translate_clip_lines(
            lines, languages, source=source_language, target=target,
            romanize=romanize, emit=emit)
        emit("translation", routeLabel=report["routeLabel"],
             backend=report["backend"],
             romajiBackend=report.get("romajiBackend"),
             targetLanguage=report["targetLanguage"], error=report["error"])
        lyrics = write_lyrics_sheet(
            outdir, manifest.get("title") or assets["title"], lines,
            report["romaji"], report["lines"])
        if lyrics:
            files["lyrics"] = lyrics
        lyrixa = build_lyrixa_bundle(
            outdir, json_name, manifest.get("title") or assets["title"],
            float(manifest.get("duration") or assets["duration"] or 0),
            audio_file_name=(assets["original"].name if assets.get("original")
                             else None),
            model=settings.get("model"), language=source_language,
            translations=report["lines"],
            translation_language=report["targetLanguage"],
            romaji=report["romaji"], precision_report=precision_report or None,
        )
        if lyrixa:
            files["lyrixa"] = lyrixa
        settings.update({"translateTo": target, "romaji": romanize})
        jobstore.update_manifest(
            job_id, status="completed", files=files, settings=settings,
            error=None, finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job))
        emit("done", title=manifest.get("title") or assets["title"],
             files=files, job_id=job_id,
             links=jobstore.job_links(job_id, files))
    except Exception as error:  # noqa: BLE001
        jobstore.update_manifest(job_id, status="error", error=str(error),
                        finishedAt=jobstore.utc_now(), timing=_finish_job_timing(job))
        emit("error", message=str(error))
    finally:
        with job["cond"]:
            job["done"] = True
            job["cond"].notify_all()


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
_AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".opus": "audio/opus", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".flac": "audio/flac", ".wav": "audio/wav", ".aac": "audio/aac",
    ".webm": "video/webm", ".mkv": "video/x-matroska",
}

# Un tope para que una llamada equivocada no ocupe Ollama media hora. Una capa
# entera de un mix de tres horas cabe de sobra.
MAX_TRANSLATE_LINES = 4000


def audio_media_type(path: Path) -> str:
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


class TranslateRequest(BaseModel):
    """Una línea suelta (`text`) o una capa entera (`lines`)."""

    text: str | None = None
    lines: list[str] | None = None
    sourceLanguage: str | None = None
    targetLanguage: str | None = None
    romaji: bool = False

    def payload_lines(self) -> list[str]:
        if self.lines is not None:
            return self.lines
        return [self.text] if self.text is not None else []


@app.get("/", response_class=HTMLResponse)
def index():
    return (paths.STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    auto_engine = (
        "mlx-whisper" if precision.te.mlx_whisper_available()
        else "faster-whisper" if precision.te.faster_whisper_available()
        else "openai-whisper"
    )
    return JSONResponse({
        "device": device_name(),
        # Cada motor guarda un artefacto distinto: .pt original, conversión
        # CTranslate2 o pesos MLX. La UI muestra exactamente cuál utilizará.
        "models": [
            {
                "name": name, **MODEL_DETAILS[name],
                "artifacts": MODEL_ARTIFACTS[name],
                "cached": model_is_cached(name),
                "faster_cached": precision.te.faster_model_cached(name),
                "mlx_cached": precision.te.mlx_model_cached(name),
                "ultra_cached": (
                    precision.te.mlx_model_cached(name)
                    if auto_engine == "mlx-whisper"
                    else precision.te.faster_model_cached(name)
                ),
            }
            for name in MODELS
        ],
        "auto_engine": auto_engine,
        "languages": [{"value": v, "label": l} for v, l in LANGUAGES],
        "audio_formats": AUDIO_FORMATS,
        "audio_qualities": AUDIO_QUALITIES,
        "separation_profiles": [
            {"value": value, "label": spec["label"], "hint": spec["hint"],
             "available": media.separation_profile_available(value)}
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
        "demucs": media.demucs_available(),
        "roformer": media.roformer_available(),
        "media.separator_acceleration": media.separator_acceleration(),
        "faster_whisper": precision.te.faster_whisper_available(),
        "mlx_whisper": precision.te.mlx_whisper_available(),
        "asr_engines": [
            {"value": "auto", "label": "Auto · mejor para esta Mac",
             "available": True,
             "hint": f"Elegirá {auto_engine} automáticamente."},
            {"value": "mlx-whisper", "label": "MLX · Apple Silicon",
             "available": precision.te.mlx_whisper_available(),
             "hint": "GPU Metal y memoria unificada. Es la ruta más rápida en esta Mac."},
            {"value": "faster-whisper", "label": "faster-whisper · CPU",
             "available": precision.te.faster_whisper_available(),
             "hint": "CTranslate2 cuantizado: eficiente en CPU y con sus propios pesos."},
            {"value": "openai-whisper", "label": "Whisper original · CPU",
             "available": True,
             "hint": "Implementación original de OpenAI; compatible, pero normalmente más lenta."},
        ],
        "translator": {
            "available": translation.translation_available(),
            "model": translation.pick_model(),
        },
        # Lo que un cliente necesita para integrarse sin adivinar puertos.
        "service": {
            **config.as_dict(),
            "bridgeSchemaVersion": lyrixa_bridge.BRIDGE_SCHEMA_VERSION,
            "layerRoles": list(lyrixa_bridge.LAYER_ROLES),
            "translationLanguages": sorted(translation.LANGUAGE_NAMES),
            "romanizable": sorted(translation.ROMANIZABLE),
        },
    })


@app.get("/api/jobs")
def list_saved_jobs():
    jobs = []
    for outdir in paths.OUTPUT_DIR.iterdir():
        if not outdir.is_dir() or not _JOB_ID_RE.fullmatch(outdir.name):
            continue
        record = jobstore.saved_job(outdir.name, running=job_is_running(outdir.name))
        if record:
            jobs.append(record)
    jobs.sort(key=lambda item: item["updatedAt"], reverse=True)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def saved_job_detail(job_id: str):
    record = jobstore.saved_job(job_id, running=job_is_running(job_id))
    if record is None:
        return JSONResponse({"error": "job no existe"}, status_code=404)
    tracklist = paths.OUTPUT_DIR / job_id / "tracklist.txt"
    record["tracklistText"] = (
        tracklist.read_text(encoding="utf-8") if tracklist.is_file() else ""
    )
    return record


@app.delete("/api/jobs/{job_id}")
def delete_saved_job(job_id: str):
    """Permanently delete one exact, inactive output directory."""
    if not _JOB_ID_RE.fullmatch(job_id):
        return JSONResponse({"error": "ID de trabajo inválido."}, status_code=400)
    current = JOBS.get(job_id)
    if current is not None and not current["done"]:
        return JSONResponse(
            {"error": "El trabajo sigue en curso. Espera a que termine o detén el servidor."},
            status_code=409,
        )
    outdir = (paths.OUTPUT_DIR / job_id).resolve()
    if not outdir.is_dir() or paths.OUTPUT_DIR.resolve() not in outdir.parents:
        return JSONResponse({"error": "El trabajo ya no existe."}, status_code=404)
    shutil.rmtree(outdir)
    JOBS.pop(job_id, None)
    return {"deleted": True, "job_id": job_id}


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
    asr_engine: str = Form("auto"),
    rescue_original: str = Form("1"),
    translate_to: str = Form(""),
    romaji: str = Form("0"),
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
    if asr_engine not in ASR_ENGINES:
        asr_engine = "auto"
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
    outdir = paths.OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    if source_type == "file":
        # Nunca confiamos en el nombre que manda el navegador para armar rutas.
        upload_path = outdir / Path(file.filename or "audio").name
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    if tracklist_text.strip():
        (outdir / "tracklist.txt").write_text(
            tracklist_text.strip() + "\n", encoding="utf-8")
    jobstore.update_manifest(
        job_id, status="queued", mode=mode, iteration=0,
        settings={
            "model": model, "language": language,
            "separationQuality": separation_quality,
            "precisionLevel": precision_level, "asrEngine": asr_engine,
            "rescueOriginal": rescue_original not in {"", "0", "false"},
            "translateTo": translate_to,
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
              tracklist_text, fetch_lyrics not in {"", "0", "false"},
              asr_engine, rescue_original not in {"", "0", "false"}),
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

    assets = jobstore.recovery_assets(job_id)
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
    asr_engine: str = Form("auto"),
    rescue_original: str = Form("1"),
    translate_to: str = Form(""),
    romaji: str = Form("0"),
    tracklist_text: str = Form(""),
    fetch_lyrics: str = Form("1"),
):
    current = JOBS.get(job_id)
    if current is not None and not current["done"]:
        return JSONResponse({"error": "El trabajo ya está corriendo."},
                            status_code=409)
    assets = jobstore.recovery_assets(job_id)
    if assets is None or assets["checkpoint"] is None:
        return JSONResponse(
            {"error": "No quedó un audio intermedio recuperable."},
            status_code=404,
        )
    checkpoint, _checkpoint_kind = jobstore.reusable_checkpoint(assets)
    if checkpoint is None:
        return JSONResponse(
            {"error": "La voz aislada o asr-input.wav quedó incompleto. "
                      "No archivé la salida anterior ni reinicié BS-RoFormer."},
            status_code=409,
        )
    if model not in MODELS:
        model = "small"
    if precision_level not in PRECISION_LEVELS:
        precision_level = "balanced"
    if asr_engine not in ASR_ENGINES:
        asr_engine = "auto"
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

    iteration = jobstore._archive_outputs(assets["outdir"])
    if tracklist_text.strip():
        (assets["outdir"] / "tracklist.txt").write_text(
            tracklist_text.strip() + "\n", encoding="utf-8")
        # A new guide must be rebuilt; the previous one belongs to another
        # version/section of the pasted list.
        (assets["outdir"] / "track-guide.json").unlink(missing_ok=True)
    jobstore.update_manifest(
        job_id, status="queued", iteration=iteration,
        settings={
            "model": model, "language": language,
            "precisionLevel": precision_level, "asrEngine": asr_engine,
            "rescueOriginal": rescue_original not in {"", "0", "false"},
            "translateTo": translate_to,
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
              fetch_lyrics not in {"", "0", "false"}, asr_engine,
              rescue_original not in {"", "0", "false"}),
        daemon=True,
    ).start()
    return {"job_id": job_id, "resumed": True}


@app.post("/api/jobs/{job_id}/translate")
async def translate_saved_job(
    job_id: str,
    translate_to: str = Form(""),
    romaji: str = Form("0"),
):
    current = JOBS.get(job_id)
    if current is not None and not current["done"]:
        return JSONResponse({"error": "El trabajo ya está corriendo."},
                            status_code=409)
    record = jobstore.saved_job(job_id, running=job_is_running(job_id))
    if not record or not record.get("files", {}).get("json"):
        return JSONResponse(
            {"error": "No hay un JSON cronometrado que reutilizar."},
            status_code=404,
        )
    target = translation.normalize_code(translate_to) or ""
    wants_romaji = romaji not in {"", "0", "false"}
    if target and target not in TARGET_LANGUAGES:
        return JSONResponse(
            {"error": f"Idioma de traducción desconocido: {target}"},
            status_code=400,
        )
    if not target and not wants_romaji:
        return JSONResponse(
            {"error": "Elige un idioma o activa romaji."}, status_code=400)
    if target and not translation.translation_available():
        return JSONResponse(
            {"error": "No hay traductor local disponible."}, status_code=409)
    JOBS[job_id] = {
        "events": [], "cond": threading.Condition(), "done": False,
    }
    threading.Thread(
        target=translate_saved_pipeline,
        args=(job_id, target, wants_romaji), daemon=True,
    ).start()
    return {"job_id": job_id, "translating": True}


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


@app.get("/api/health")
def health():
    """¿Está vivo el motor? Lo primero que preguntará una shell de escritorio.

    Deliberadamente barato: no toca disco ni consulta a Ollama, porque un
    supervisor lo va a llamar en bucle mientras arranca.
    """
    return {
        "status": "ok",
        "service": "transcriptor",
        "bridgeSchemaVersion": lyrixa_bridge.BRIDGE_SCHEMA_VERSION,
        "publicUrl": config.PUBLIC_URL,
    }


@app.get("/api/jobs/{job_id}/lyrixa")
def lyrixa_project(job_id: str, download: bool = False):
    """El proyecto Lyrixa de este trabajo, por id y no por nombre de archivo.

    Se sirve como archivo en vez de releerlo y volver a serializarlo: en un mix
    de tres horas el envelope pesa varios megas y volver a construirlo no
    añadiría nada.
    """
    path = jobstore.lyrixa_project_path(job_id)
    if path is None:
        return JSONResponse(
            {"error": "Este trabajo todavía no tiene proyecto de Lyrixa."},
            status_code=404,
        )
    return FileResponse(
        path, media_type="application/json",
        filename=path.name if download else None,
    )


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str, download: bool = False):
    """El audio original del trabajo, para que Lyrixa no lo pida a mano.

    `FileResponse` responde a `Range`, así que un `<audio>` puede buscar dentro
    de una mezcla de tres horas sin descargarla entera.
    """
    assets = jobstore.recovery_assets(job_id)
    original = (assets or {}).get("original")
    if original is None:
        return JSONResponse({"error": "No hay audio original guardado."},
                            status_code=404)
    return FileResponse(
        original, media_type=audio_media_type(original),
        filename=original.name if download else None,
    )


@app.post("/api/translate")
def translate_text(request: TranslateRequest):
    """Traducción y romaji como servicio, con el mismo motor que el pipeline.

    Lyrixa corrige una línea a mano y necesita su traducción sin volver a
    transcribir nada. Que llame aquí es lo que evita que acabe con su propia
    copia de la configuración de Ollama, del pivote por inglés y del romaji.
    """
    lines = request.payload_lines()
    if not lines:
        return JSONResponse({"error": "No hay texto que traducir."},
                            status_code=400)
    if len(lines) > MAX_TRANSLATE_LINES:
        return JSONResponse(
            {"error": f"Máximo {MAX_TRANSLATE_LINES} líneas por llamada."},
            status_code=413,
        )

    source = translation.normalize_code(request.sourceLanguage)
    target = translation.normalize_code(request.targetLanguage)
    if target and target not in translation.LANGUAGE_NAMES:
        return JSONResponse(
            {"error": f"Idioma de destino desconocido: {request.targetLanguage}"},
            status_code=400,
        )
    if not target and not request.romaji:
        return JSONResponse({"error": "Elige un idioma o activa romaji."},
                            status_code=400)

    out: dict = {"sourceLanguage": source, "targetLanguage": target,
                 "lines": None, "romaji": None, "route": [], "routeLabel": None,
                 "backend": None, "romajiBackend": None}

    if request.romaji:
        try:
            converted, backend = translation.romanize(lines)
        except translation.TranslationUnavailable as error:
            return JSONResponse({"error": str(error)}, status_code=503)
        except Exception as error:  # noqa: BLE001
            return JSONResponse({"error": str(error)}, status_code=502)
        out["romaji"] = converted
        out["romajiBackend"] = backend

    if target:
        route = translation.plan_route(source, target)
        if not route:
            # Ya está en el idioma pedido: devolverlo tal cual es la respuesta
            # correcta, no un error.
            out["lines"] = list(lines)
        else:
            try:
                result = translation.translate(
                    lines, source=source, target=target,
                    cache_dir=translation.DEFAULT_CACHE_DIR)
            except translation.TranslationUnavailable as error:
                return JSONResponse({"error": str(error)}, status_code=503)
            except Exception as error:  # noqa: BLE001
                return JSONResponse({"error": str(error)}, status_code=502)
            out["lines"] = result["lines"]
            out["backend"] = result["backend"]
            out["route"] = [list(step) for step in route]
            out["routeLabel"] = translation.route_label(route)

    if request.text is not None:
        # Petición de una sola línea: se responde en la misma forma en que
        # llegó, para que el cliente no tenga que desenvolver una lista de uno.
        out["text"] = (out["lines"] or lines)[0]
        if out["romaji"]:
            out["romajiText"] = out["romaji"][0]
    return out


@app.get("/download/{job_id}/{name}")
def download(job_id: str, name: str):
    # El nombre viene de la URL: nos quedamos solo con el archivo y verificamos
    # que el resultado siga dentro de la carpeta del job (nada de "../").
    job_dir = (paths.OUTPUT_DIR / job_id).resolve()
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
    job_dir = (paths.OUTPUT_DIR / job_id).resolve()
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

    uvicorn.run(app, host=config.HOST, port=config.PORT)
