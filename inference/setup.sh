#!/usr/bin/env bash
# Optional native Linux/WSL setup; Docker builds install the same dependencies.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc >/dev/null || { echo "CUDA Toolkit 12.8 with nvcc is required." >&2; exit 1; }
python3.12 - <<'PY'
from pathlib import Path
import sysconfig
if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
    raise SystemExit("Install Python 3.12 development headers and venv support first.")
PY
cd "$PROJECT_DIR"
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install 'setuptools>=61' wheel packaging ninja
.venv/bin/python -m pip install -r inference/requirements.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ./mamba-source --no-build-isolation
.venv/bin/python -m pip check
