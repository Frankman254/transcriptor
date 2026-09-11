"""Rutas de archivo: audio del trabajo, stems, descarga y
el proyecto Lyrixa.
"""

from fastapi import APIRouter

router = APIRouter()

from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
import jobstore
import paths
from jobhub import hub
from serveutil import safe_stem

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
_AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".opus": "audio/opus", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".flac": "audio/flac", ".wav": "audio/wav", ".aac": "audio/aac",
    ".webm": "video/webm", ".mkv": "video/x-matroska",
}
def audio_media_type(path: Path) -> str:
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
@router.get("/api/jobs/{job_id}/lyrixa")
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
@router.get("/api/jobs/{job_id}/audio")
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
@router.get("/download/{job_id}/{name}")
def download(job_id: str, name: str):
    # El nombre viene de la URL: nos quedamos solo con el archivo y verificamos
    # que el resultado siga dentro de la carpeta del job (nada de "../").
    job_dir = (paths.OUTPUT_DIR / job_id).resolve()
    path = (job_dir / Path(name).name).resolve()
    if not path.is_file() or job_dir not in path.parents:
        return JSONResponse({"error": "no existe"}, status_code=404)
    return FileResponse(path, filename=path.name)
@router.get("/api/jobs/{job_id}/stems/{stem}")
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
    job = hub.get(job_id)
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
