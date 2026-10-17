"""LoRA for nn.Linear without assuming Transformer or PEFT APIs."""

import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA требует nn.Linear и корректные rank/alpha/dropout")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        # Mamba3.allocate_inference_cache reads dtype/device from in_proj.weight.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        result = self.base(x)
        # Keep small adapters and gradients in FP32; return the original dtype to MIMO.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(self.dropout(x.float()), self.lora_A), self.lora_B)
        return result + (delta * (self.alpha / self.rank)).to(result.dtype)


def attach_lora(model, rank=16, alpha=32.0, dropout=0.0):
    """Adapt mixer.in_proj/out_proj only, preserving MIMO rank and SSM parameters."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("К модели уже подключён адаптер")
    targets = []
    for i, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        if not getattr(mixer, "is_mimo", False):
            raise ValueError(f"Слой {i}: ожидается Mamba-3 MIMO")
        for name in ("in_proj", "out_proj"):
            linear = getattr(mixer, name)
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"Слой {i}.{name}: ожидается nn.Linear")
            targets.append((mixer, name, linear))
    if not targets:
        raise ValueError("Не найдены слои Mamba-3")
    model.requires_grad_(False)
    for mixer, name, linear in targets:
        setattr(mixer, name, LoRALinear(linear, rank, alpha, dropout))
    return [name for name, m in model.named_modules() if isinstance(m, LoRALinear)]


def adapter_parameters(model):
    return {name: p for name, p in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}


def adapter_state(model):
    return {name: p.detach().cpu().contiguous().clone() for name, p in adapter_parameters(model).items()}


def restore_adapter(model, state):
    params = adapter_parameters(model)
    if not params or set(params) != set(state):
        raise ValueError("Набор параметров адаптера не совпадает с моделью")
    for name, p in params.items():
        if state[name].shape != p.shape or state[name].dtype != torch.float32 or not torch.isfinite(state[name]).all():
            raise ValueError(f"Некорректный тензор адаптера: {name}")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state[name])


def save_adapter(model, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not modules:
        raise ValueError("Нет адаптеров для сохранения")
    first = next(iter(modules.values()))
    config = dict(metadata, format="hafvillai-mamba3-lora-v1", rank=first.rank,
                  alpha=first.alpha, dropout=first.dropout.p, target_modules=list(modules))
    save_file(adapter_state(model), str(directory / "adapter.safetensors"))
    (directory / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_adapter(model, directory, *, expected_base):
    """Load the native format onto a fresh base; expected_base comes from load_base()."""
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "hafvillai-mamba3-lora-v1" or config.get("base") != expected_base:
        raise ValueError("Формат адаптера или версия базовой модели не совпадают")
    names = attach_lora(model, config["rank"], config["alpha"], config["dropout"])
    if names != config["target_modules"]:
        raise ValueError("Целевые слои адаптера не совпадают")
    restore_adapter(model, load_file(str(directory / "adapter.safetensors")))
    return config
