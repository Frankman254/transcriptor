"""Rutas de trabajos: crear, observar (SSE), reanudar,
traducir y borrar.
"""

__all__ = ["router"]

from fastapi import APIRouter

router = APIRouter()

import re
import shutil
import threading
import uuid
from pathlib import Path
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import jobstore
import paths
import track_guides
import translation
from jobhub import hub
from pipelines import pipeline, resume_precision_pipeline, translate_saved_pipeline
from catalogs import (
    ASR_ENGINES,
    AUDIO_QUALITIES,
    MODELS,
    PRECISION_LEVELS,
    SEPARATION_PROFILES,
    TARGET_LANGUAGES,
    FORMATS_BY_VALUE,
)

def job_is_running(job_id: str) -> bool:
    """Si hay un pipeline vivo para este trabajo en *este* proceso."""
    return hub.live(job_id)
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
@router.get("/api/jobs")
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
@router.get("/api/jobs/{job_id}")
def saved_job_detail(job_id: str):
    record = jobstore.saved_job(job_id, running=job_is_running(job_id))
    if record is None:
        return JSONResponse({"error": "job no existe"}, status_code=404)
    tracklist = paths.OUTPUT_DIR / job_id / "tracklist.txt"
    record["tracklistText"] = (
        tracklist.read_text(encoding="utf-8") if tracklist.is_file() else ""
    )
    return record
@router.delete("/api/jobs/{job_id}")
def delete_saved_job(job_id: str):
    """Permanently delete one exact, inactive output directory."""
    if not _JOB_ID_RE.fullmatch(job_id):
        return JSONResponse({"error": "ID de trabajo inválido."}, status_code=400)
    if hub.live(job_id):
        return JSONResponse(
            {"error": "El trabajo sigue en curso. Espera a que termine o detén el servidor."},
            status_code=409,
        )
    outdir = (paths.OUTPUT_DIR / job_id).resolve()
    if not outdir.is_dir() or paths.OUTPUT_DIR.resolve() not in outdir.parents:
        return JSONResponse({"error": "El trabajo ya no existe."}, status_code=404)
    shutil.rmtree(outdir)
    hub.pop(job_id)
    return {"deleted": True, "job_id": job_id}
@router.post("/api/tracklists/preview")
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
@router.post("/api/jobs")
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
    if audio_format not in FORMATS_BY_VALUE:
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

    hub.create(job_id)
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
@router.get("/api/jobs/{job_id}/state")
def job_state(job_id: str):
    """Indica si el job sigue vivo o si quedó un checkpoint recuperable."""
    job = hub.get(job_id)
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
@router.post("/api/jobs/{job_id}/resume")
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
    if hub.live(job_id):
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

    iteration = jobstore.archive_outputs(assets["outdir"])
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

    hub.create(job_id)
    threading.Thread(
        target=resume_precision_pipeline,
        args=(job_id, model, language, precision_level, translate_to,
              romaji not in {"", "0", "false"}, tracklist_text,
              fetch_lyrics not in {"", "0", "false"}, asr_engine,
              rescue_original not in {"", "0", "false"}),
        daemon=True,
    ).start()
    return {"job_id": job_id, "resumed": True}
@router.post("/api/jobs/{job_id}/translate")
async def translate_saved_job(
    job_id: str,
    translate_to: str = Form(""),
    romaji: str = Form("0"),
):
    if hub.live(job_id):
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
    hub.create(job_id)
    threading.Thread(
        target=translate_saved_pipeline,
        args=(job_id, target, wants_romaji), daemon=True,
    ).start()
    return {"job_id": job_id, "translating": True}
@router.get("/api/jobs/{job_id}/events")
def events(job_id: str, request: Request):
    job = hub.get(job_id)
    if not job:
        return JSONResponse({"error": "job no existe"}, status_code=404)

    # Reconexión: `?since=<id>` (recarga de página: el cliente dice el último
    # evento que pintó) o la cabecera `Last-Event-ID` (reconexión automática de
    # EventSource). Retomamos desde el siguiente; si el punto ya fue podado del
    # anillo, `hub.stream` manda antes un `snapshot` con la historia retirada.
    last = request.query_params.get("since") or request.headers.get("last-event-id")
    start = 0
    if last is not None:
        try:
            start = int(last) + 1
        except ValueError:
            start = 0

    return StreamingResponse(hub.stream(job, start),
                             media_type="text/event-stream")
