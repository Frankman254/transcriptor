#!/usr/bin/env bash
# Arranca el Transcriptor local en http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"

# Instala dependencias del backend la primera vez.
python3 -c "import fastapi, uvicorn, multipart, mutagen, numpy, certifi" 2>/dev/null || \
  pip3 install -r requirements.txt

# El modo ultra prefiere faster-whisper; si no está, cae a openai-whisper.
python3 -c "import faster_whisper" 2>/dev/null || \
  echo "Aviso: 'pip3 install faster-whisper' hace el modo ultra varias veces más rápido."

# Apple Silicon: evita que Turbo y large-v3 queden limitados al CPU de
# CTranslate2. Igual que el resto de dependencias, se instala una sola vez.
if [[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]]; then
  python3 -c "import mlx_whisper" 2>/dev/null || pip3 install mlx-whisper
fi

# La traducción y el romaji corren en Ollama, también local.
curl -sf -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || \
  echo "Aviso: Ollama no responde. Sin él el modo ultra transcribe igual, pero no traduce."

if [[ ! -x ".separator-venv/bin/audio-separator" ]]; then
  echo "Aviso: instala el modo vocal Preciso una vez con ./setup_separator.sh"
fi

exec python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
