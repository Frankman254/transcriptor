# 🎙️ Transcriptor local

App web para transcribir audio/video y sacar letras de canciones **100% en tu
propia máquina** con [Whisper](https://github.com/openai/whisper) y
[Demucs](https://github.com/facebookresearch/demucs). El audio nunca sale de tu
equipo: no usa ninguna API en la nube.

Progreso en vivo: modelo, dispositivo (CPU/GPU), barra de avance y cada segmento
con su tiempo `inicio → fin` conforme se genera (vía SSE).

## Modos

| Modo | Qué hace |
|------|----------|
| 🎧 **Transcribir** | Texto con timestamps de un link de YouTube o un archivo. |
| 🎤 **Letra de canción** | Aísla la voz de la música con Demucs y luego la transcribe. |
| ⬇️ **Solo audio** | Descarga el mejor audio de un YouTube (sin recodificar), sin transcribir. |

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
- `demucs` *(opcional)* — solo para el modo "letra de canción": `pip3 install demucs`

Las dependencias del backend (`fastapi`, `uvicorn`, `python-multipart`) las
instala `run.sh` la primera vez.

Más detalles en [`webapp/README.md`](webapp/README.md).
