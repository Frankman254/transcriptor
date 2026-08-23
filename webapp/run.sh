#!/usr/bin/env bash
# Arranca el Transcriptor local en http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"

# Instala dependencias del backend la primera vez.
python3 -c "import fastapi, uvicorn, multipart, mutagen" 2>/dev/null || \
  pip3 install -r requirements.txt

exec python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
