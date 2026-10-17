"""Atomic checkpoints for adapters, AdamW, scheduler, and RNG state."""

import json
from pathlib import Path
import random
import tempfile

from safetensors.torch import load_file
import torch

from hafvillai_runtime.lora import restore_adapter, save_adapter


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(output, model, optimizer, scheduler, progress, manifest):
    output = Path(output)
    name = f"checkpoint-{progress['step']:06d}"
    target = output / name
    if target.exists():
        raise FileExistsError(f"Checkpoint уже существует: {target}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix=".checkpoint-") as tmp:
        ready = Path(tmp) / "ready"
        save_adapter(model, ready, {"base": manifest["base"], "role": manifest["config"]["role"],
                                   "tokenizer": manifest["tokenizer"],
                                   "source_fingerprint": manifest["source_fingerprint"],
                                   "step": progress["step"]})
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "progress": progress, "rng": rng_state(), "manifest": manifest}, ready / "training.pt")
        ready.rename(target)
    write_json(output / "latest.json", {"checkpoint": name})
    return target


def load_checkpoint(directory, model, optimizer, scheduler, manifest):
    directory = Path(directory)
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    if state["manifest"] != manifest:
        raise ValueError("Checkpoint создан с другими данными, кодом, версиями или конфигурацией")
    restore_adapter(model, load_file(str(directory / "adapter.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state["progress"]
