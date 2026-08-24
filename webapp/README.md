# 🎙️ Transcriptor local

App web para transcribir audio/video **en tu propia máquina** con Whisper.
Muestra en vivo: el **modelo** que se usa, el **dispositivo** (CPU/GPU), la **barra
de progreso**, y **cada segmento con su tiempo de inicio → fin** mientras avanza.

Tiene tres modos:
- **🎧 Transcribir** — texto con timestamps de un YouTube o archivo.
- **🎤 Letra de canción** — aísla la voz con BS-RoFormer o Demucs y la
  transcribe. Mucho mejor que transcribir la canción directa.
- **⬇️ Solo audio** — descarga un YouTube en MP3, M4A, Opus, OGG,
  FLAC, ALAC, WAV o en su formato original, sin transcribir.

## Requisitos (ya instalados en esta máquina)
- `python3`
- `yt-dlp` y `ffmpeg` (Homebrew) — para YouTube y para leer audio/video
- `openai-whisper` (pip) — el motor de transcripción
- BS-RoFormer (opcional, recomendado) — se instala con `./setup_separator.sh`
- `demucs` (pip, opcional) — perfiles vocales equilibrado/rápido
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

Para habilitar una vez el perfil vocal **Preciso**:

```bash
cd webapp
./setup_separator.sh
```

Se instala en `.separator-venv`, separado de Whisper y Demucs. El modelo de
aproximadamente 610 MB se descarga automáticamente al usarlo por primera vez.

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

En el modo **Letra de canción** aparecen además dos reproductores: voz aislada
y pista instrumental. Son los WAV originales creados por el separador elegido
y permiten detectar si una frase incorrecta viene de una mala separación o de
una alucinación de Whisper.

El perfil **Preciso** es el recomendado para mixes: usa BS-RoFormer y procesa
los audios de más de diez minutos por bloques para contener el uso de memoria.
**Equilibrado** ejecuta dos shifts de `htdemucs` con solapamiento `0.5` y
**Rápido** usa una sola pasada con solapamiento `0.25`.

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
