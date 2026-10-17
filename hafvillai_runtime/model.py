"""Pinned base loading and Mamba kernel dtype invariants."""

import json
from pathlib import Path

import torch

from .files import file_sha256


def load_base(config, *, device="cuda", offline=False, local_dir=None):
    # Upstream from_pretrained forwards kwargs to __init__ and cannot accept revision.
    # Download a pinned snapshot, then use the standard config and constructor.
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    names = ("config.json", "pytorch_model.bin")
    if local_dir is None:
        from huggingface_hub import hf_hub_download
        paths = {name: Path(hf_hub_download(config.model_id, name, revision=config.model_revision,
                                           local_files_only=offline)) for name in names}
    else:
        paths = {name: Path(local_dir) / name for name in names}
    if file_sha256(paths["config.json"]) != config.model_config_sha256:
        raise ValueError("Base configuration does not match the pinned model")
    if file_sha256(paths["pytorch_model.bin"]) != config.model_weights_sha256:
        raise ValueError("Base weights do not match the pinned model")
    raw = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    ssm = raw.get("ssm_cfg", {})
    if ssm.get("layer") != "Mamba3" or not ssm.get("is_mimo") or raw.get("attn_layer_idx"):
        raise ValueError("Ожидается чистая Mamba-3 MIMO без attention-слоёв")
    state = torch.load(paths["pytorch_model.bin"], map_location="cpu", weights_only=True)
    model = MambaLMHeadModel(MambaConfig(**raw), device=device, dtype=torch.bfloat16)
    # Copy weights while preserving FP32 dt_bias/B_bias/C_bias/D/mimo_* from initialization.
    # Do not call model.to(bfloat16): auxiliary parameters require their original precision.
    model.load_state_dict(state, strict=True)
    del state
    base = {"model_id": config.model_id, "revision": config.model_revision,
            "config_sha256": file_sha256(paths["config.json"])}
    assert_kernel_dtypes(model)
    return model, base


def assert_kernel_dtypes(model):
    for i, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        if mixer.in_proj.weight.dtype != torch.bfloat16 or mixer.out_proj.weight.dtype != torch.bfloat16:
            raise ValueError(f"Слой {i}: проекции должны быть BF16")
        for name in ("dt_bias", "B_bias", "C_bias", "D", "mimo_x", "mimo_z", "mimo_o"):
            if getattr(mixer, name).dtype != torch.float32:
                raise ValueError(f"Слой {i}.{name}: ожидается FP32")
