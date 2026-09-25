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

## La interfaz

`webapp/static/` son tres archivos servidos tal cual: `index.html` (marcado),
`app.css` (diseño) y `app.js` (comportamiento). Hasta el rediseño eran uno
solo de 1543 líneas; se partieron porque el CSS y el JS ya no cabían en la
cabeza de nadie y porque el navegador puede cachearlos por separado.
`server.py` monta `paths.STATIC_DIR` en `/static` —es su único motivo para
importar `paths`— y `routes_system` sigue sirviendo el `index.html` en `/`.

La pantalla es un shell de dos paneles:

```
┌─ topbar ──────────────────────────────────────────────┐
│ 🎙️ Transcriptor  [device]              [+ Nuevo]      │
├──────────────┬────────────────────────────────────────┤
│ BIBLIOTECA   │ Nuevo trabajo  |  Trabajo              │
│ buscar · 🔍  │                                        │
│ filtros      │ (formulario  o  progreso + resultados) │
│ orden        │                                        │
│ N filas ↕    │                                        │
└──────────────┴────────────────────────────────────────┘
```

Dos decisiones explican el resto:

- **La biblioteca es una columna fija con su propio scroll, no una tarjeta
  apilada.** Con 57 trabajos guardados, la versión anterior obligaba a
  recorrer la lista entera para llegar al formulario. Ahora la lista no
  compite por el espacio vertical con nada, y tiene búsqueda (`/`), filtros
  por estado con su contador y orden por fecha, título, duración o tamaño.
- **El panel derecho enseña una sola cosa a la vez.** El formulario y el
  trabajo en curso son dos vistas del mismo espacio (`setView`), no dos
  tarjetas que conviven. Abrir un trabajo guardado, lanzar uno nuevo o
  recuperar uno interrumpido llevan siempre a la vista correcta.

Por debajo no cambió nada: los mismos `id` de elemento, el mismo stream SSE
con `?since=`, el mismo `esc()` antes de cualquier `innerHTML` —los títulos de
YouTube y los mensajes del servidor son texto ajeno, y esta API local incluye
`DELETE /api/jobs/{id}`.
