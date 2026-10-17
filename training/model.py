"""Official MambaLMHeadModel with a checkpointed backbone and chunked loss."""


import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hafvillai_runtime.serialization import IGNORE_INDEX


def forward_hidden(model, input_ids, *, gradient_checkpointing=True):
    if not gradient_checkpointing or not torch.is_grad_enabled():
        return model.backbone(input_ids)
    # Match the residual/add/norm order in the standard MixerModel.forward.
    # Preserve Block/Mamba3 ordering; local fixes are listed in LOCAL_CHANGES.md.
    backbone = model.backbone
    hidden = backbone.embedding(input_ids)
    residual = None
    for block in backbone.layers:
        hidden, residual = checkpoint(block, hidden, residual, use_reentrant=False)
    if not backbone.fused_add_norm:
        summed = hidden if residual is None else hidden + residual
        return backbone.norm_f(summed.to(backbone.norm_f.weight.dtype))
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
    return layer_norm_fn(hidden, backbone.norm_f.weight, backbone.norm_f.bias,
                         eps=backbone.norm_f.eps, residual=residual, prenorm=False,
                         residual_in_fp32=backbone.residual_in_fp32,
                         is_rms_norm=isinstance(backbone.norm_f, RMSNorm))


def causal_loss_sum(head, hidden, labels, *, chunk_tokens=128, checkpoint_loss=True):
    """Return summed cross-entropy and target count with one exact next-token shift.

    Project only positions preceding assistant targets, avoiding [B,S,128256] logits.
    Checkpoint head and cross-entropy to release each chunk's logits before backward.
    """
    if chunk_tokens < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("Некорректные формы hidden/labels или chunk_tokens")
    shifted = labels[:, 1:]
    selected = shifted != IGNORE_INDEX
    targets = shifted[selected]
    if not targets.numel():
        raise ValueError("Нет целевых токенов после causal shift")
    features = hidden[:, :-1, :][selected]

    def cross_entropy(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    losses = []
    for start in range(0, len(targets), chunk_tokens):
        x, y = features[start:start+chunk_tokens], targets[start:start+chunk_tokens]
        if checkpoint_loss and torch.is_grad_enabled():
            loss = checkpoint(cross_entropy, x, y, use_reentrant=False)
        else:
            loss = cross_entropy(x, y)
        losses.append(loss)
    return torch.stack(losses).sum(), len(targets)


def batch_loss(model, ids, labels, config):
    hidden = forward_hidden(model, ids, gradient_checkpointing=config.gradient_checkpointing)
    return causal_loss_sum(model.lm_head, hidden, labels, chunk_tokens=config.loss_chunk_tokens)
