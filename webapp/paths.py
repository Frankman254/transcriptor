"""Dónde escribe este backend.

Un solo módulo con las rutas del disco, por dos razones.  La primera es que
`server`, el almacén de trabajos y las utilidades de medios necesitan las
mismas carpetas y ninguno debería ser el dueño de las de los otros.  La
segunda es que así hay *un* sitio que redirigir: las pruebas apuntan
`paths.OUTPUT_DIR` a un directorio temporal y todo el backend las sigue, en
lugar de tener que parchear cada módulo que resulte usarlas.

Por eso los consumidores leen `paths.OUTPUT_DIR` y no importan el nombre
suelto: un `from paths import OUTPUT_DIR` congela el valor en el momento del
import y deja de seguir al que redirige.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"

# Caché de modelos de Whisper.  Es la de la librería, no nuestra: la
# compartimos con cualquier otra herramienta del sistema que use whisper.
CACHE_DIR = Path.home() / ".cache" / "whisper"

# BS-RoFormer vive en su propio entorno porque sus dependencias chocan con las
# de Whisper.  Se instala bajo demanda; que no exista es normal.
ROFORMER_VENV_DIR = BASE_DIR / ".separator-venv"
ROFORMER_CLI = ROFORMER_VENV_DIR / "bin" / "audio-separator"
ROFORMER_MODEL_DIR = BASE_DIR / ".separator-models"
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

OUTPUT_DIR.mkdir(exist_ok=True)

LYRICS_CACHE_DIR = OUTPUT_DIR / ".lyrics-cache"
TRANSLATION_CACHE_DIR = OUTPUT_DIR / ".translation-cache"

# El manifiesto que convierte una carpeta de salida en un trabajo recuperable.
MANIFEST_NAME = "job.json"
