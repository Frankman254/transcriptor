# 🎙️ Transcriptor local

App web para transcribir audio/video **en tu propia máquina** con Whisper.
Muestra en vivo: el **modelo** que se usa, el **dispositivo** (CPU/GPU), la **barra
de progreso**, y **cada segmento con su tiempo de inicio → fin** mientras avanza.

Tiene tres modos:
- **🎧 Transcribir** — texto con timestamps de un YouTube o archivo.
- **🎤 Letra de canción** — aísla la voz de la música con Demucs y la transcribe
  (para sacar la letra). Mucho mejor que transcribir la canción directa.
- **⬇️ Solo audio** — descarga un YouTube en MP3, M4A, Opus, OGG,
  FLAC, ALAC, WAV o en su formato original, sin transcribir.

## Requisitos (ya instalados en esta máquina)
- `python3`
- `yt-dlp` y `ffmpeg` (Homebrew) — para YouTube y para leer audio/video
- `openai-whisper` (pip) — el motor de transcripción
- `demucs` (pip, opcional) — solo para el modo "letra de canción"
- `mutagen` (pip) — permite que `yt-dlp` incruste portadas en todos los
  contenedores compatibles

## Arrancar

```bash
cd webapp
./run.sh
```

Luego abre **http://127.0.0.1:8000** en el navegador.

La primera vez instala `fastapi`, `uvicorn`, `python-multipart` y `mutagen`
automáticamente.

## Cómo funciona
1. Pegas un **link de YouTube** o subes un **archivo** (mp3, m4a, wav, mp4…).
2. Eliges **modelo** (tiny → large) e **idioma** (o auto).
3. El backend:
   - Descarga el audio con `yt-dlp` (si es YouTube), en el formato y bitrate
     elegidos. Incrusta título, artista, álbum, fecha y portada cuando el
     contenedor lo permite (WAV no admite portada).
   - Se asegura de tener el modelo: si no está en `~/.cache/whisper`, lo baja con
     `curl` (evita el error SSL de proxys corporativos).
   - Corre `whisper --verbose True` y **parsea cada segmento** apenas aparece,
     transmitiéndolo al navegador por SSE (Server-Sent Events).
4. Al terminar puedes **descargar** el audio y los archivos `.txt`, `.srt`,
   `.vtt`, `.tsv`, `.json` y `.lyrixa.json`, todos nombrados a partir del audio
   o video original.

El `.json` sigue siendo la salida raw de Whisper. El `.lyrixa.json` es una
derivación independiente: divide los segmentos largos usando exclusivamente los
timestamps reales por palabra y deja clips de letra manejables. Impórtalo en
Lyrixa mediante **More → Import project (.lyrixa.json)**, carga el mismo audio
y revisa la sincronización antes de exportar el bundle para LiveWallpaper.

Los resultados quedan en `webapp/output/<job_id>/`.

## Modelos
| Modelo | Calidad | Velocidad (CPU) |
|--------|---------|-----------------|
| tiny   | baja    | muy rápida      |
| base   | baja+   | rápida          |
| small  | media   | media           |
| medium | alta    | ~1x duración    |
| large-v3 | máxima | lenta          |
| turbo  | alta    | rápida          |

El primer uso de un modelo nuevo lo descarga una vez (~1–3 GB según el modelo).

## Notas
- Todo corre **local**: el audio no sale de tu máquina.
- Sin GPU, `medium` tarda aproximadamente lo que dura el audio. Para audios
  largos, `turbo` o `small` son bastante más rápidos.
