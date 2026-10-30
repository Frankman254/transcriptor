"""Rutas de sistema: index, /api/info, /api/health y la
traducción suelta.
"""

__all__ = ["router"]

from fastapi import APIRouter

router = APIRouter()

from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import config
import lyrixa_bridge
import media
import paths
import precision
import translation
from serveutil import device_name, model_is_cached
from catalogs import (
    AUDIO_FORMATS,
    AUDIO_QUALITIES,
    LANGUAGES,
    MODELS,
    MODEL_ARTIFACTS,
    MODEL_DETAILS,
    PRECISION_LEVELS,
    SEPARATION_PROFILES,
    TARGET_LANGUAGES,
    VIDEO_FORMATS,
    VIDEO_QUALITIES,
)

# Un tope para que una llamada equivocada no ocupe Ollama media hora. Una capa
# entera de un mix de tres horas cabe de sobra.
MAX_TRANSLATE_LINES = 4000
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
@router.get("/", response_class=HTMLResponse)
def index():
    html = (paths.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # El servidor puede reiniciarse mientras la pestaña sigue abierta. El HTML
    # no se cachea y las URLs de CSS/JS cambian con sus mtimes, así una recarga
    # normal nunca conserva una interfaz anterior contra un backend nuevo.
    assets = (paths.STATIC_DIR / "app.css", paths.STATIC_DIR / "app.js")
    version = f"{max(asset.stat().st_mtime_ns for asset in assets):x}"
    return HTMLResponse(
        html.replace("__SCRIBIX_ASSET_VERSION__", version),
        headers={"Cache-Control": "no-store"},
    )
@router.get("/api/info")
def info():
    engines = precision.engine_availability(MODELS)
    auto_engine = engines["auto"]
    return JSONResponse({
        "device": device_name(),
        # Cada motor guarda un artefacto distinto: .pt original, conversión
        # CTranslate2 o pesos MLX. La UI muestra exactamente cuál utilizará.
        "models": [
            {
                "name": name, **MODEL_DETAILS[name],
                "artifacts": MODEL_ARTIFACTS[name],
                "cached": model_is_cached(name),
                "faster_cached": engines["cached"][name]["faster-whisper"],
                "mlx_cached": engines["cached"][name]["mlx-whisper"],
                "ultra_cached": engines["cached"][name]["auto"],
            }
            for name in MODELS
        ],
        "auto_engine": auto_engine,
        "languages": [{"value": v, "label": l} for v, l in LANGUAGES],
        "audio_formats": AUDIO_FORMATS,
        "audio_qualities": AUDIO_QUALITIES,
        "video_formats": VIDEO_FORMATS,
        "video_qualities": VIDEO_QUALITIES,
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
        "faster_whisper": engines["available"]["faster-whisper"],
        "mlx_whisper": engines["available"]["mlx-whisper"],
        "asr_engines": [
            {"value": "auto", "label": "Auto · mejor para esta Mac",
             "available": True,
             "hint": f"Elegirá {auto_engine} automáticamente."},
            {"value": "mlx-whisper", "label": "MLX · Apple Silicon",
             "available": engines["available"]["mlx-whisper"],
             "hint": "GPU Metal y memoria unificada. Es la ruta más rápida en esta Mac."},
            {"value": "faster-whisper", "label": "faster-whisper · CPU",
             "available": engines["available"]["faster-whisper"],
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
@router.get("/api/health")
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
@router.post("/api/translate")
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
