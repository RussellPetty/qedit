#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /qedit.log) 2>&1
echo "=== qedit start $(date -u +%FT%TZ) ==="
nvidia-smi || true
export HF_HOME="/models/hf"
export PIP_CACHE_DIR="/models/pip"
export TOKENIZERS_PARALLELISM=false

mkdir -p /models/hf /app
if [ ! -x /models/venv/bin/python ]; then python3 -m venv /models/venv; fi

echo "--- fetching code ---"
rm -rf /tmp/qc
git clone --depth 1 https://github.com/RussellPetty/qedit /tmp/qc
cp -a /tmp/qc/. /app/
rm -rf /tmp/qc

if [ ! -f /models/venv/.deps-ok ] || [ /app/requirements.txt -nt /models/venv/.deps-ok ]; then
  echo "--- installing deps (cached on volume after first boot) ---"
  /models/venv/bin/pip install --upgrade -q pip
  /models/venv/bin/pip install -q -r /app/requirements.txt
  touch /models/venv/.deps-ok
fi

cd /app
echo "--- launching server on :8000 ---"
exec /models/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75
