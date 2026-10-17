#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"

command -v nvidia-smi >/dev/null
nvidia-smi
command -v nvcc >/dev/null
nvcc --version
python3.12 - <<'PY'
from pathlib import Path
import sysconfig
if not (Path(sysconfig.get_path("include")) / "Python.h").exists():
    raise SystemExit("Нужны заголовки Python: sudo apt-get install python3.12-dev")
PY
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install 'setuptools>=61' wheel packaging ninja
cd "$REPO_DIR/hafvillai"
../.venv/bin/python -m pip install -r requirements.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128
cd "$REPO_DIR"
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python -m pip check
# run_pipeline.sh runs code and GPU checks once before smoke tests and SFT.

