#!/usr/bin/env bash
set -euo pipefail
STUDIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_PYTHON="$STUDIO_DIR/.venv/bin/python"
if [[ ! -x "$STUDIO_PYTHON" ]]; then
    echo "Сначала создай окружение dataset-studio по инструкции в README.md." >&2
    exit 1
fi
cd "$STUDIO_DIR"
# One process owns HER state files and the single active operation.
exec "$STUDIO_PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "${STUDIO_PORT:-8765}" --workers 1
