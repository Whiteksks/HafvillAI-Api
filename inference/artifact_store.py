"""Portable, checksummed inference artifacts without training state or data."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import tempfile

from hafvillai_runtime.config import ModelSpec
from hafvillai_runtime.files import file_sha256
from hafvillai_runtime.config import SERIALIZATION

ROOT = Path(__file__).resolve().parent
ROLES = ("consciousness", "subconsciousness")
FORMAT = "hafvillai-inference-artifacts-v1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def adapter_metadata(directory, role, spec):
    metadata = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    expected_base = {"model_id": spec.model_id, "revision": spec.model_revision,
                     "config_sha256": spec.model_config_sha256}
    expected_tokenizer = {"id": spec.tokenizer_id, "revision": spec.tokenizer_revision,
                          "format": SERIALIZATION}
    if (metadata.get("format") != "hafvillai-mamba3-lora-v1"
            or metadata.get("role") != role or metadata.get("base") != expected_base
            or metadata.get("tokenizer") != expected_tokenizer):
        raise ValueError(f"{role}: incompatible adapter format, base model, or tokenizer")
    if type(metadata.get("step")) is not int or metadata["step"] < 0:
        raise ValueError(f"{role}: invalid adapter step")
    return metadata


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("model") != asdict(ModelSpec()):
        raise ValueError("Artifact manifest does not match the pinned runtime")
    expected = {"base/config.json", "base/pytorch_model.bin"}
    if set(manifest.get("roles", {})) != set(ROLES):
        raise ValueError("Both role artifacts are required")
    for role in ROLES:
        expected.update({f"{role}/adapter.safetensors", f"{role}/adapter_config.json"})
        expected.update(f"{role}/tokenizer/{name}" for name in TOKENIZER_FILES)
    checksums = manifest.get("files", {})
    if set(checksums) != expected:
        raise ValueError("Artifact manifest has missing or unexpected files")
    for name, digest in checksums.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid artifact checksum: {name}")
        if file_sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    spec = ModelSpec()
    if checksums["base/config.json"] != spec.model_config_sha256:
        raise ValueError("Base configuration checksum does not match the pinned model")
    if checksums["base/pytorch_model.bin"] != spec.model_weights_sha256:
        raise ValueError("Base weights checksum does not match the pinned model")
    for role in ROLES:
        metadata = adapter_metadata(directory / role, role, spec)
        if manifest["roles"][role] != {"step": metadata["step"]}:
            raise ValueError(f"{role}: manifest and adapter steps differ")
    return manifest


def prepare_artifacts(source_root, base_directory, destination):
    """Copy selected adapters and cached base files atomically; never overwrite."""
    source_root, base_directory, destination = map(Path, (source_root, base_directory, destination))
    if destination.exists():
        raise FileExistsError(f"Artifacts already exist: {destination}; use another destination")
    spec = ModelSpec()
    files = {f"base/{name}": base_directory / name for name in ("config.json", "pytorch_model.bin")}
    if file_sha256(files["base/config.json"]) != spec.model_config_sha256:
        raise ValueError("Cached base configuration does not match the pinned model")
    roles = {}
    for role in ROLES:
        run = (source_root / role).resolve()
        selected = json.loads((run / "best.json").read_text(encoding="utf-8"))["checkpoint"]
        if not isinstance(selected, str):
            raise ValueError(f"{role}: invalid checkpoint selection")
        checkpoint = (run / selected).resolve()
        if not checkpoint.is_relative_to(run):
            raise ValueError(f"{role}: checkpoint escapes its run directory")
        metadata = adapter_metadata(checkpoint, role, spec)
        roles[role] = {"step": metadata["step"]}
        for name in ("adapter.safetensors", "adapter_config.json"):
            files[f"{role}/{name}"] = checkpoint / name
        for name in TOKENIZER_FILES:
            files[f"{role}/tokenizer/{name}"] = run / "tokenizer" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".artifacts-") as temporary:
        ready = Path(temporary) / "ready"
        checksums = {}
        for name, source in files.items():
            target = ready / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            checksums[name] = file_sha256(target)
            if file_sha256(source) != checksums[name]:
                raise ValueError(f"Source changed while copying: {name}")
        manifest = {"format": FORMAT, "model": asdict(spec), "roles": roles, "files": checksums}
        (ready / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_artifacts(ready)
        ready.rename(destination)
    return manifest
