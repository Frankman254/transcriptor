# 🎙️ Transcriptor local

App web para transcribir audio/video y sacar letras de canciones **100% en tu
propia máquina** con [Whisper](https://github.com/openai/whisper), BS-RoFormer y
[Demucs](https://github.com/facebookresearch/demucs). El audio nunca sale de tu
equipo: no usa ninguna API en la nube.

Progreso en vivo: modelo, dispositivo (CPU/GPU), barra de avance y cada segmento
con su tiempo `inicio → fin` conforme se genera (vía SSE).

## Modos

| Modo | Qué hace |
|------|----------|
| 🎧 **Transcribir** | Texto con timestamps de un link de YouTube o un archivo. |
| 🎤 **Letra de canción** | Aísla la voz con BS-RoFormer o Demucs y luego la transcribe. |
| ⬇️ **Solo audio** | Descarga audio de YouTube en MP3, M4A, Opus, OGG, FLAC, ALAC, WAV u original. |

## Puente hacia Lyrixa

Además de conservar las salidas originales de Whisper, cada transcripción
genera un proyecto de autoría `.lyrixa.json`. Los segmentos largos se dividen
con los timestamps reales de `words[]`, priorizando puntuación y pausas, para
crear clips editables de aproximadamente 2–6 segundos (8 segundos como límite
normal). Si los timestamps por palabra faltan o están incompletos, se conserva
el segmento original completo como fallback.

En Lyrixa usa **More → Import project (.lyrixa.json)** y después carga el mismo
audio para revisar el timeline. Tras editar, usa **Export lyrics bundle** para
obtener el `.lyrixa-lyrics.json` que consume LiveWallpaperAnimeGlitch.

Al terminar el modo de letras, el resultado también muestra reproductores y
descargas para la voz aislada y el instrumental. Se sirven directamente desde
la salida del separador elegido, sin recomprimir ni duplicar los WAV, para poder
auditar qué material recibió Whisper.

La separación ofrece tres perfiles. **Preciso** (predeterminado) usa
BS-RoFormer y recupera mejor voces débiles; **Equilibrado** usa Demucs con dos
shifts y 50% de solapamiento; **Rápido** usa una sola pasada de Demucs.

## Arrancar

```bash
cd webapp
./run.sh
```

Abre **http://127.0.0.1:8000** en el navegador.

## Requisitos

- `python3`
- `ffmpeg` y `yt-dlp` (para leer audio/video y descargar de YouTube)
- `openai-whisper` (motor de transcripción)
- BS-RoFormer *(opcional, recomendado para letras)*: `cd webapp && ./setup_separator.sh`
- `demucs` *(opcional)* — perfiles equilibrado/rápido: `pip3 install demucs`

Las dependencias del backend (`fastapi`, `uvicorn`, `python-multipart`, `mutagen`) las
instala `run.sh` la primera vez.

Las descargas desde YouTube conservan el título original del video. Cuando el
formato lo permite, también incluyen título, artista, álbum, fecha y portada
incrustada. WAV no admite portada.

Más detalles en [`webapp/README.md`](webapp/README.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
```
