#!/usr/bin/env bash
# Instala BS-RoFormer en un entorno aislado para no alterar Whisper/Demucs.
set -euo pipefail
cd "$(dirname "$0")"

SEPARATOR_VENV=".separator-venv"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

if [[ ! -x "$SEPARATOR_VENV/bin/python" ]]; then
  "$PYTHON_COMMAND" -m venv "$SEPARATOR_VENV"
fi

"$SEPARATOR_VENV/bin/python" -m pip install --upgrade pip
"$SEPARATOR_VENV/bin/python" -m pip install \
  "audio-separator[cpu]==0.44.5" \
  "librosa==0.10.2.post1" \
  "audioread>=3.0,<4"

echo "BS-RoFormer listo. El modelo de ~610 MB se descargará al elegir el perfil Preciso."
