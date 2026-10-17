"""Prepare or verify local inference artifacts without a GPU or cloud access."""

import argparse
import json
import os
from pathlib import Path

from hafvillai_runtime.config import ModelSpec
from .artifact_store import ROOT, prepare_artifacts, verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--destination", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "outputs/brev-coser")
    cache = Path(os.getenv("HF_HUB_CACHE", ROOT.parent / "hafvillai/.cache/huggingface/hub"))
    spec = ModelSpec()
    parser.add_argument("--base-directory", type=Path,
                        default=cache / ("models--" + spec.model_id.replace("/", "--")) / "snapshots" / spec.model_revision)
    args = parser.parse_args()
    result = (verify_artifacts(args.destination) if args.action == "verify" else
              prepare_artifacts(args.source_root, args.base_directory, args.destination))
    print(json.dumps({"status": "verified", "roles": result["roles"],
                      "files": len(result["files"]), "directory": str(args.destination)}, indent=2))


if __name__ == "__main__":
    main()
