#!/usr/bin/env bash
# Uses the same env file for Compose interpolation and the API container.
set -euo pipefail
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$INFERENCE_DIR/.env" ]] || { echo "Copy inference/.env.example to inference/.env and set HAFVILLAI_API_KEY." >&2; exit 1; }
exec docker compose --env-file "$INFERENCE_DIR/.env" -f "$INFERENCE_DIR/deploy/compose.yaml" "$@"
