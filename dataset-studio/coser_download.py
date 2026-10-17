"""Download pinned original CoSER books; no model calls or generated training."""

import argparse
import json
from pathlib import Path

from data_utils import atomic_json, sha256

DATASET = "Neph0s/CoSER"
REVISION = "7cc80430f92532cda85df45015a4aca8ecc068d0"
DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "coser"


def download(root=DEFAULT_ROOT):
    from huggingface_hub import HfApi, snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    selected = sorted(s.rfilename for s in info.siblings
                      if s.rfilename.startswith(("full/", "test/")) or s.rfilename == "README.md")
    if info.sha != REVISION or not selected:
        raise ValueError("Unexpected CoSER revision or empty remote file list")
    snapshot_download(DATASET, repo_type="dataset", revision=REVISION,
                      allow_patterns=selected, local_dir=root / "raw", max_workers=4)
    files = {}
    metadata = {s.rfilename: s for s in info.siblings}
    for name in selected:
        path = root / "raw" / name
        size = path.stat().st_size
        if size != metadata[name].size:
            raise ValueError(f"Incomplete download: {name}")
        if name.endswith(".json"):
            json.loads(path.read_text(encoding="utf-8"))
        files[name] = {"bytes": size, "sha256": sha256(path)}
    manifest = {"dataset": DATASET, "revision": REVISION, "files": files,
                "book_files": sum(n.startswith("full/") for n in files),
                "bytes": sum(f["bytes"] for f in files.values()),
                "excluded": ["train/sft_conversations_sharegpt.json"],
                "reason": "Prepare role dialogues with book/scene provenance from original full files."}
    atomic_json(root / "download_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    download(parser.parse_args().root)
