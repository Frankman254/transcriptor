# Arquitectura del backend

## El problema que resuelve

`server.py` llegó a 2543 líneas. Dentro había enrutado HTTP, el pipeline de
transcripción, llamadas a yt-dlp y ffprobe, los separadores de voz, el catálogo
de opciones de la interfaz y la lectura de la carpeta de cada trabajo. Nada
impedía añadirle una línea más, así que se le añadían.

Eso importa más allá de la estética. Con todo en un módulo, cualquier cosa que
quisiera reutilizar el motor —una CLI, una prueba, la futura shell de
escritorio— tenía que importar FastAPI y levantar la aplicación para llegar a
una función que solo llamaba a ffprobe.

## Las capas

Las flechas van hacia abajo. El servidor conoce el motor; el motor no conoce el
servidor.

```
                    ┌──────────────┐
                    │   server     │  HTTP, SSE, pipeline
                    └──────┬───────┘
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        ┌─────────┐  ┌───────────┐  ┌───────────┐
        │jobstore │  │ precision │  │translation│
        └────┬────┘  └─────┬─────┘  └─────┬─────┘
             ▼             ▼              │
        ┌─────────┐  ┌──────────────────┐ │
        │  media  │  │transcribe_engines│ │
        └────┬────┘  └────────┬─────────┘ │
             └───────┬────────┴───────────┘
                     ▼
      paths · catalogs · config · textjoin · consensus ·
      audio_analysis · track_guides · lyrixa_bridge
```

| Módulo | Líneas | De qué es dueño |
|---|---:|---|
| `server` | ~1480 | La superficie HTTP y el pipeline que la orquesta. |
| `precision` | 685 | El modo de varias pasadas y su presupuesto. |
| `lyrixa_bridge` | ~680 | El sobre `.lyrixa.json`: identidad de línea, capas, clips. |
| `consensus` | 633 | La votación entre pasadas y el corte en segmentos. |
| `translation` | ~595 | Ollama, rutas de idioma, romaji, caché, mezclas multilingües. |
| `media` | 494 | yt-dlp, ffprobe, Demucs, BS-RoFormer. Todo binario externo. |
| `transcribe_engines` | 485 | MLX / faster-whisper / openai-whisper tras una interfaz. |
| `track_guides` | 449 | Tracklists y guías de sección. |
| `jobstore` | ~385 | Un trabajo tal y como queda en `output/<id>/`. |
| `audio_analysis` | 352 | Medición de stems. |
| `catalogs` | 174 | Los catálogos de opciones. Datos, sin comportamiento. |
| `config` | ~105 | Dónde vive el servicio y quién puede llamarlo. |
| `textjoin` | 94 | Unir palabras respetando el sistema de escritura. |
| `paths` | 40 | Las carpetas del disco. |

## Las tres reglas

**1. Nada por debajo de `server` importa `server`.** Es la que hace que el
motor se pueda usar sin HTTP. Las funciones largas del pipeline reciben un
`emit` en lugar de escribir en el bus de eventos, así que no saben que existe
un servidor.

**2. `paths` es el único dueño de las rutas.** Los demás leen
`paths.OUTPUT_DIR`, nunca `from paths import OUTPUT_DIR`: importar el nombre
suelto congela el valor y deja de seguir a quien lo redirige. Las pruebas
apuntan `paths.OUTPUT_DIR` a un temporal y todo el backend las sigue.

**3. `jobstore` no sabe qué está corriendo.** Describe carpetas en disco.
`saved_job` recibe `running=` en vez de consultarlo, porque el registro en
memoria es del servidor y una carpeta tiene que poder describirse después de un
reinicio en el que no queda ningún trabajo vivo.

## Cómo se hacen cumplir

`tests/test_module_boundaries.py` lee los imports de cada módulo con `ast` y los
compara con una tabla de lo permitido. Un módulo nuevo falla hasta que alguien
decida a qué capa pertenece. También comprueba que el grafo no tenga ciclos y
—con pyflakes, si está instalado— que ningún módulo use un nombre que nunca
importó.

Esa última comprobación no es teórica: al mover el descargador a `media.py` se
quedó atrás su `from collections import deque`. El módulo importaba
perfectamente y el fallo solo habría aparecido al descargar un vídeo. Un import
limpio no demuestra que un módulo funcione.

## Lo que queda

- `server.py` sigue mezclando el pipeline (~800 líneas) con las rutas (~630).
  El siguiente corte natural es un `pipeline.py`, pero necesita resolver antes
  el acoplamiento con `JOBS` y con `emit`.
- Los catálogos son datos que solo consume `/api/info`. Si la shell de
  escritorio acaba dibujando su propio formulario, querrán ser un JSON servido
  y no un módulo Python.
