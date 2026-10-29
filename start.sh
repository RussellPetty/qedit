#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /qedit.log) 2>&1
echo "=== qedit start $(date -u +%FT%TZ) ==="
nvidia-smi || true
export HF_HOME="${HF_HOME:-/models}"
export PIP_CACHE_DIR="$HF_HOME/pip-cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p /models /app
if [ ! -x /venv/bin/python ]; then python3 -m venv /venv; fi

echo "--- fetching code ---"
rm -rf /tmp/qc
git clone --depth 1 https://github.com/RussellPetty/qedit /tmp/qc
cp -a /tmp/qc/. /app/
rm -rf /tmp/qc

if [ ! -f /venv/.deps-ok ] || [ /app/requirements.txt -nt /venv/.deps-ok ]; then
  echo "--- installing deps (cached after first boot) ---"
  /venv/bin/pip install --upgrade -q pip
  /venv/bin/pip install -q -r /app/requirements.txt
  touch /venv/.deps-ok
fi

cd /app
echo "--- launching server on :8000 ---"
exec /venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75
