"""Export private inference code and artifacts without training or local secrets."""

import argparse
from pathlib import Path
import shutil
import tempfile

from .artifact_store import ROOT, verify_artifacts

EXCLUDED = {"artifacts", "dist", ".cache", ".venv", "__pycache__", "tests",
            "test-results", ".git", ".github", "build", "benchmarks", "evals",
            "secrets", ".aws", ".ssh", ".codex", ".agents"}


def ignore(directory, names):
    return [name for name in names if name in EXCLUDED or name.endswith((".pyc", ".egg-info", ".pem", ".key", ".p12", ".pfx"))
            or (name.startswith(".env") and name != ".env.example")]


def build_bundle(repository, artifacts, destination):
    repository, artifacts, destination = map(Path, (repository, artifacts, destination))
    if destination.exists():
        raise FileExistsError(f"Bundle already exists: {destination}; use another destination")
    manifest = verify_artifacts(artifacts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".bundle-") as temporary:
        ready = Path(temporary) / "ready"
        for name in ("inference", "hafvillai_runtime", "mamba-source"):
            shutil.copytree(repository / name, ready / name, ignore=ignore)
        shutil.copyfile(repository / ".dockerignore", ready / ".dockerignore")
        shutil.copytree(artifacts, ready / "inference/artifacts")
        verify_artifacts(ready / "inference/artifacts")
        (ready / "README.md").write_text(
            "# Hafvillai private inference bundle\n\n"
            "See [inference/README.md](inference/README.md) for setup and deployment.\n"
            "This directory includes model weights and tokenizers. Transfer it privately; "
            "do not publish it as a source repository or public image.\n", encoding="utf-8")
        ready.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/hafvillai-inference")
    args = parser.parse_args()
    result = build_bundle(ROOT.parent, args.artifacts, args.destination)
    print(f"Bundle ready: {args.destination}; roles: {', '.join(result['roles'])}")


if __name__ == "__main__":
    main()
