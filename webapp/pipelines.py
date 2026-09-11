"""Los pipelines de trabajo: transcribir, modo ultra, reanudar y
traducir un trabajo guardado. Corren en su propio hilo; `jobhub.hub`
es su única conexión con las rutas.
"""

from lyrixa_bridge import build_lyrixa_envelope, clip_languages, clip_lines

import json
from pathlib import Path
import audio_analysis
import jobstore
import media
import paths
import precision
import track_guides
import translation
from jobhub import hub
from serveutil import _finish_job_timing, _mark_job_stage, _start_job_timing, device_name, ensure_model, safe_stem, start_caffeinate
from whispercli import run_whisper, write_whisper_outputs
from catalogs import PRECISION_LEVELS, SEPARATION_PROFILES


def make_emit(job: dict):
    """Fabrica el emisor de eventos de un trabajo.

    Único punto donde un pipeline publica: un evento `status` anota primero la
    etapa en el timing del manifiesto en vivo, y todo evento entra al anillo
    de `jobhub` con su id monotónico — el contrato SSE (ids `id: <n>`,
    reanudación por `since`/`Last-Event-ID` y `snapshot` al caer del anillo)
    vive allí, no aquí.
    """
    def emit(event: str, **data):
        if event == "status":
            _mark_job_stage(job, data.get("stage"))
        hub.publish(job, event, data)
    return emit


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
    job = hub.require(job_id)
    _start_job_timing(job)

    emit = make_emit(job)

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
        hub.finish(job)  # marca fin del stream (para todos los lectores)
def resume_precision_pipeline(
    job_id: str, model: str, language: str, precision_level: str,
    translate_to: str, romaji: bool, tracklist_text: str = "",
    fetch_lyrics: bool = True, asr_engine: str = "auto",
    rescue_original: bool = True,
) -> None:
    """Continúa un modo ultra desde los WAV que sobrevivieron al reinicio."""
    job = hub.require(job_id)
    _start_job_timing(job)

    emit = make_emit(job)

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
        hub.finish(job)
def translate_saved_pipeline(job_id: str, target: str, romanize: bool) -> None:
    """Add or replace language layers without rerunning separation or ASR."""
    job = hub.require(job_id)
    _start_job_timing(job)

    emit = make_emit(job)

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
        hub.finish(job)
