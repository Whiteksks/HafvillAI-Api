"""Shared cached decoder for comparisons and the API; model weights are unchanged."""

import math
import torch


def stabilize_inference_norm():
    """Use the same RMSNorm configuration for prefill and step in every process."""
    import triton
    from mamba_ssm.ops.triton import layer_norm
    tuner = layer_norm._layer_norm_fwd_1pass_kernel
    tuner.configs = [triton.Config({}, num_warps=4)]
    tuner.cache.clear()


def repetition_cycle(tokens):
    """Require at least four exact block copies spanning 64 repeated tokens."""
    if len(tokens) < 128:
        return None
    for period in range(1, min(128, len(tokens)//4)+1):
        copies = max(4, math.ceil(64/period))
        size = copies*period
        if size <= len(tokens) and tokens[-size:] == tokens[-period:]*copies:
            start = len(tokens)-size
            while start > 0 and tokens[start-1] == tokens[start-1+period]:
                start -= 1
            return {"period":period,"copies":copies,"unit":tokens[-period:],"start":start}
    return None


@torch.inference_mode()
def generate_tokens(model, ids, budget, stop_ids, decoding, seed, repetition_penalty=1.0,
                    stop_on_repetition=False):
    """Use a fresh cache per request, multiple stop IDs, and an output-only penalty.

    With penalty=1, greedy prefixes match the library generator.
    Apply the penalty to the last 256 generated tokens before sampling.
    """
    from mamba_ssm.utils.generation import InferenceParams, sample
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or budget < 1:
        raise ValueError("Expected one nonempty prompt and a positive output budget")
    if decoding not in {"greedy", "sample"}:
        raise ValueError("Unknown decoding mode")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty must be finite and >= 1")
    if decoding == "sample":
        torch.manual_seed(seed)
        if ids.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    state = InferenceParams(max_seqlen=ids.shape[1] + budget, max_batch_size=1)
    current = ids
    result = []
    for _ in range(budget):
        positions = (torch.full_like(current, state.seqlen_offset)
                     if state.seqlen_offset > 0 else None)
        logits = model(current, position_ids=positions, inference_params=state,
                       num_last_tokens=1).logits.squeeze(1)
        if repetition_penalty != 1.0 and result:
            logits = logits.float().clone()
            seen = torch.tensor(sorted(set(result[-256:])), device=logits.device)
            values = logits[:, seen]
            logits[:, seen] = torch.where(values < 0, values * repetition_penalty,
                                          values / repetition_penalty)
        token = sample(logits, top_k=1 if decoding == "greedy" else 50,
                       top_p=0.0 if decoding == "greedy" else 0.9,
                       temperature=1.0 if decoding == "greedy" else 0.7)
        value = int(token.item())
        result.append(value)
        state.seqlen_offset += current.shape[1]
        if value in stop_ids or (stop_on_repetition and repetition_cycle(result) is not None):
            break
        current = token.reshape(1, 1)
    return result

