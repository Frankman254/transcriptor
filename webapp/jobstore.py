"""Un trabajo, tal y como queda en el disco.

Cada transcripción vive en `output/<job_id>/`: el audio de origen, los stems,
las salidas de Whisper, el proyecto de Lyrixa y un `job.json` que es lo único
que sobrevive a un reinicio del servidor.  Este módulo es el que sabe leer esa
carpeta y contar qué hay dentro.

La frontera con `server` es deliberada: aquí no se sabe qué trabajos están
corriendo ahora mismo.  `saved_job` recibe ese dato (`running=`) en vez de
consultarlo, porque el registro en memoria es del servidor y una carpeta en
disco tiene que poder describirse sin él — desde una prueba, desde una CLI o
después de un reinicio en el que no queda ningún trabajo vivo.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "archive_outputs", "discover_result_files", "job_links",
    "lyrixa_project_path", "prepare_track_guide", "read_manifest",
    "recovery_assets", "reusable_checkpoint", "saved_job",
    "stored_stem_report", "update_manifest", "utc_now",
]

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import audio_analysis
import config
import media
import paths
import track_guides

# Un id de trabajo son doce dígitos hexadecimales y nada más.  Es la única
# validación que impide que un `job_id` llegado por la URL se convierta en un
# `../` sobre el disco del usuario.
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

_SOURCE_SUFFIXES = {
    ".mp3", ".m4a", ".opus", ".ogg", ".flac", ".wav",
    ".mp4", ".webm", ".mov", ".mkv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reusable_checkpoint(assets: dict) -> tuple[Path | None, str | None]:
    """Pick a complete saved ASR/stem file without decoding the whole mix.

    A WAV may exist while it is still being written or after an interrupted
    separator left only its header.  ffprobe is effectively instant and lets
    resume prefer the conditioned ASR file, fall back to the vocal stem, and
    refuse a truncated checkpoint instead of silently producing worse text.
    """
    expected = float(assets.get("duration") or 0.0)
    for kind in ("asr", "vocals"):
        path = assets.get(kind)
        if not path or not path.is_file() or path.stat().st_size <= 44:
            continue
        measured = media.ffprobe_duration(path)
        if measured <= 0:
            continue
        tolerance = max(5.0, expected * 0.01)
        if expected > 0 and abs(measured - expected) > tolerance:
            continue
        return path, kind
    return None, None


def stored_stem_report(outdir: Path) -> audio_analysis.StemReport | None:
    """Restore separation QC so an iteration needn't decode two huge WAVs."""
    manifest = read_manifest(outdir)
    data = manifest.get("stem")
    if not isinstance(data, dict):
        data = (manifest.get("precision") or {}).get("stem")
    if not isinstance(data, dict):
        return None
    return audio_analysis.StemReport(
        vocal_dbfs=float(data.get("vocalDbfs", -120.0)),
        peak_dbfs=float(data.get("peakDbfs", -120.0)),
        speech_coverage=float(data.get("speechCoverage", 0.0)),
        speech_seconds=float(data.get("speechSeconds", 0.0)),
        bleed_ratio=float(data.get("bleedRatio", 0.0)),
        clipping=float(data.get("clipping", 0.0)),
        verdict=str(data.get("verdict") or "unknown"),
        notes=list(data.get("notes") or []),
    )


def recovery_assets(job_id: str) -> dict | None:
    """Descubre checkpoints útiles de un job cuyo servidor se reinició.

    Los procesos no sobreviven al cierre de la sesión de macOS, pero los WAV sí.
    Si existe ``asr-input.wav`` podemos retomar justo antes del consenso, que es
    la etapa cara posterior a la separación.
    """
    if not _JOB_ID_RE.fullmatch(job_id):
        return None
    outdir = (paths.OUTPUT_DIR / job_id).resolve()
    if not outdir.is_dir() or paths.OUTPUT_DIR.resolve() not in outdir.parents:
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
    duration = media.ffprobe_duration(duration_source) if duration_source else 0.0
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


def read_manifest(outdir: Path) -> dict:
    try:
        data = json.loads((outdir / paths.MANIFEST_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def update_manifest(job_id: str, **changes) -> dict:
    """Persist the small part of a job that must survive a server restart."""
    outdir = paths.OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(outdir)
    manifest.update(changes)
    manifest.setdefault("id", job_id)
    manifest.setdefault("createdAt", utc_now())
    manifest["updatedAt"] = utc_now()
    target = outdir / paths.MANIFEST_NAME
    temporary = outdir / f".{paths.MANIFEST_NAME}.tmp"
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
                        fetch_lyrics: bool, emit, *,
                        retry_errors: bool = False) -> dict | None:
    """Build once, cache on disk, and reuse on every later iteration."""
    tracklist_path = outdir / "tracklist.txt"
    incoming = (text or "").strip()
    if incoming:
        tracklist_path.write_text(incoming + "\n", encoding="utf-8")
    elif tracklist_path.is_file():
        incoming = tracklist_path.read_text(encoding="utf-8")
        existing = load_track_guide(outdir)
        if existing:
            references = existing.get("references", {})
            # Las versiones anteriores guardaban cualquier error HTTPS como
            # una referencia vacía. Rehacer esas guías una vez permite que la
            # corrección del almacén de certificados también repare trabajos
            # ya existentes, sin pedir al usuario que borre la caché.
            old_empty = (
                fetch_lyrics
                and int(references.get("found") or 0) == 0
                and "failed" not in references
            )
            retry_failures = fetch_lyrics and int(references.get("failed") or 0) > 0
            if not old_empty and not retry_failures:
                emit("reference", stage="loaded", **references,
                     section=existing.get("section"),
                     tracks=existing.get("tracks", []))
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
        cache_dir=paths.LYRICS_CACHE_DIR, progress=progress,
        retry_errors=retry_errors,
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


def archive_outputs(outdir: Path) -> int:
    """Archive this run's outputs and return the new iteration number.

    Keeps the small prior transcripts when the user iterates over the same
    stems.  Mutates the folder: moves the result files into
    ``iterations/iteration-NN/`` next to a frozen copy of the manifest, so a
    later reader can tell which model and options produced them.  Call it
    before queueing the next run, never after.
    """
    manifest = read_manifest(outdir)
    iteration = int(manifest.get("iteration") or 0) + 1
    archive = outdir / "iterations" / f"iteration-{iteration:02d}"
    # El manifiesto raíz cambia en cada iteración. Congelar una copia junto a
    # sus resultados permite saber después qué modelo, motor y opciones los
    # produjeron.
    archive.mkdir(parents=True, exist_ok=True)
    if (outdir / paths.MANIFEST_NAME).is_file():
        shutil.copy2(outdir / paths.MANIFEST_NAME, archive / "run.json")
    candidates: list[Path] = []
    for path in outdir.iterdir():
        if not path.is_file() or path.name in {
            paths.MANIFEST_NAME, "track-guide.json", "tracklist.txt", "asr-input.wav",
        }:
            continue
        name = path.name.lower()
        if (path.suffix.lower() in {".srt", ".vtt", ".tsv"}
                or name.endswith(".lyrixa.json")
                or name.endswith(".letra.txt")
                or (path.suffix.lower() == ".json" and name != paths.MANIFEST_NAME)
                or path.suffix.lower() == ".txt"):
            candidates.append(path)
    if candidates:
        for path in candidates:
            shutil.move(str(path), str(archive / path.name))
    update_manifest(outdir.name, iteration=iteration)
    return iteration


def discover_result_files(job_id: str, outdir: Path, assets: dict | None) -> dict:
    """What a finished job has on disk, as ``kind -> file name``.

    Reads the folder, not just the manifest: a run interrupted before writing
    its manifest still has results, and the newest file of each kind wins.

    Two shapes, and the caller has to handle both.  Most kinds map to a bare
    file name, downloadable under ``/download/<job>/<name>``.  The two stems
    (``vocals``, ``instrumental``) map to ``{"name", "url"}`` instead: they
    live in their own subfolder and are served by their own route, so the URL
    is not derivable from the name.
    """
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
                       p.name not in {paths.MANIFEST_NAME, "track-guide.json"} and
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


def lyrixa_project_path(job_id: str) -> Path | None:
    """El `.lyrixa.json` más reciente de un trabajo, o None."""
    if not _JOB_ID_RE.fullmatch(job_id):
        return None
    outdir = (paths.OUTPUT_DIR / job_id).resolve()
    if not outdir.is_dir() or paths.OUTPUT_DIR.resolve() not in outdir.parents:
        return None
    matches = [path for path in outdir.iterdir()
               if path.is_file() and path.name.endswith(".lyrixa.json")]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def job_links(job_id: str, files: dict) -> dict:
    """Rutas estables de un trabajo, para que nadie tenga que componerlas.

    Un cliente que arme `/download/{id}/{nombre}` depende del nombre del
    archivo, que sale del título del vídeo y cambia en cada iteración. Estas
    rutas no cambian. Son relativas a propósito: se resuelven contra la base
    por la que el cliente llegó, así que `localhost` y `127.0.0.1` siguen
    siendo el mismo origen para el navegador.
    """
    links = {
        "self": f"/api/jobs/{job_id}",
        "events": f"/api/jobs/{job_id}/events",
        # Absoluta: apunta a otra aplicación, no a esta API.
        "openInLyrixa": config.lyrixa_open_url(job_id),
    }
    if files.get("lyrixa"):
        links["lyrixa"] = f"/api/jobs/{job_id}/lyrixa"
    if files.get("audio"):
        links["audio"] = f"/api/jobs/{job_id}/audio"
    for stem in ("vocals", "instrumental"):
        if files.get(stem):
            links[stem] = f"/api/jobs/{job_id}/stems/{stem}"
    return links


def saved_job(job_id: str, *, running: bool = False) -> dict | None:
    assets = recovery_assets(job_id)
    if assets is None:
        return None
    outdir = assets["outdir"]
    manifest = read_manifest(outdir)
    files = discover_result_files(job_id, outdir, assets)
    completed = any(kind in files for kind in ("json", "txt", "lyrixa"))
    manifest_status = manifest.get("status")
    if running:
        status = "running"
    elif manifest_status == "cancelled":
        status = "cancelled"
    elif completed:
        status = "completed"
    elif manifest_status == "running":
        # `running` solo describe memoria del proceso actual. Si quedó escrito
        # en disco y el hub ya no conoce el trabajo, macOS o el servidor se
        # cerraron: enseñarlo como activo crea un botón que no puede cancelar
        # nada y además bloquea Eliminar en la interfaz.
        status = "interrupted"
    elif assets.get("checkpoint"):
        status = "interrupted"
    else:
        status = manifest_status or "saved"
    guide = load_track_guide(outdir)
    modified = datetime.fromtimestamp(outdir.stat().st_mtime, timezone.utc).isoformat()
    disk_bytes = 0
    for path in outdir.rglob("*"):
        try:
            if path.is_file():
                disk_bytes += path.stat().st_size
        except OSError:
            # Un trabajo activo puede mover un archivo mientras se refresca la
            # biblioteca. El tamaño es informativo, no debe romper el listado.
            continue
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
        "settings": manifest.get("settings") or {},
        "timing": manifest.get("timing") or {},
        "precision": manifest.get("precision") or {},
        "diskBytes": disk_bytes,
        "links": job_links(job_id, files),
    }
