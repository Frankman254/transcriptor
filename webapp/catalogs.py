"""El catálogo de opciones que ofrece el motor.

Datos, no comportamiento: qué modelos hay, qué formatos de audio se pueden
pedir, qué perfiles de separación existen y qué significa cada uno.  Vive
aparte de `server` porque son doscientas líneas de tablas que no tienen nada
que ver con enrutado ni con el pipeline, y porque `GET /api/info` no es su
único cliente: una shell de escritorio que quiera dibujar su propio
formulario necesita exactamente esto y nada más.

Cada entrada lleva su `hint` en el idioma de la interfaz.  No son comentarios
para quien lee el código, son el texto que ve el usuario.
"""

from __future__ import annotations

# Modelos que ofrecemos en la interfaz (nombre -> etiqueta amigable).
MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
MODEL_DETAILS = {
    "tiny": {
        "label": "Tiny", "parameters": "39 M", "download": "≈75 MB",
        "quality": "Básica", "speed": "Muy rápido",
        "hint": "Para pruebas cortas. Pierde palabras con música o voces débiles.",
    },
    "base": {
        "label": "Base", "parameters": "74 M", "download": "≈145 MB",
        "quality": "Básica+", "speed": "Muy rápido",
        "hint": "Algo mejor que Tiny; todavía limitado para canciones y mixes.",
    },
    "small": {
        "label": "Small", "parameters": "244 M", "download": "≈460 MB",
        "quality": "Media", "speed": "Rápido",
        "hint": "Buen borrador rápido, pero puede confundir canto y efectos.",
    },
    "medium": {
        "label": "Medium", "parameters": "769 M", "download": "≈1.5 GB",
        "quality": "Alta", "speed": "Intermedio",
        "hint": "Buen equilibrio general entre precisión, memoria y tiempo.",
    },
    "large-v3": {
        "label": "Large v3", "parameters": "1,550 M", "download": "≈3 GB",
        "quality": "Máxima", "speed": "Lento",
        "hint": "La mayor precisión, especialmente en idiomas y audio difícil.",
    },
    "turbo": {
        "label": "Large v3 Turbo", "parameters": "809 M", "download": "≈1.6 GB",
        "quality": "Muy alta", "speed": "Rápido",
        "hint": "Recomendado para mixes largos: deriva de Large v3 y transcribe mucho más rápido.",
    },
}
MODEL_ARTIFACTS = {
    name: {
        "openai-whisper": f"OpenAI Whisper/{name}.pt",
        "faster-whisper": "Systran/faster-whisper-" + (
            "large-v3-turbo" if name == "turbo" else name),
        "mlx-whisper": "mlx-community/whisper-" + (
            "turbo" if name == "turbo" else f"{name}-mlx"),
    }
    for name in MODELS
}
ASR_ENGINES = {"auto", "mlx-whisper", "faster-whisper", "openai-whisper"}

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

# Presupuesto de reconocimiento del modo ultra. Los perfiles públicos están
# acotados a una o tres pasadas; el rescate posterior trabaja solo en ventanas
# dudosas, no vuelve a recorrer el mix completo.
PRECISION_LEVELS = {
    "balanced": {
        "label": "Equilibrado · 3 pasadas",
        "hint": ("Tres lecturas decorrelacionadas y una votación. Evita repetir "
                 "todo el mix cinco veces cuando el stem no puede converger."),
        "min_passes": 3, "max_passes": 3, "stability": 0.0,
    },
    "fast": {
        "label": "Rápido · 1 pasada",
        "hint": ("Una lectura fuerte, sin votación. Ideal para revisar un mix "
                 "antes de invertir tiempo en las zonas difíciles."),
        "min_passes": 1, "max_passes": 1, "stability": 0.0,
    },
    "maximum": {
        "label": "Máximo útil · 3 pasadas + rescate",
        "hint": ("Tres votos y revisión selectiva desde el audio original. No "
                 "añade pasadas globales que ya demostraron no converger."),
        "min_passes": 3, "max_passes": 3, "stability": 0.0,
    },
}
