# CoSER SFT experiment: Mamba-3 MIMO 1.5B

Report date: October 15, 2026. Training completed on October 14; final API
verification followed on October 15. Measurements below describe the recorded
experiment, not a benchmark reproduced by a fresh source checkout. Model weights,
datasets, and raw run artifacts are not distributed in Git.

## Scope

Two independent LoRA adapters were trained from the same frozen base model:
**consciousness** for character dialogue and **subconsciousness** for third-person
conversation summaries. Persistent memory, first-person experience recording,
and personality changes through experience were not implemented or evaluated.

The dialogue adapter improved completion behavior on a small check suite. The
summary adapter still produced exact loops without decoding controls and could
invent facts or turn plans into completed events. Reliable termination and
factual accuracy remain separate requirements.

## Model and data

| Item | Recorded value |
|---|---|
| Base model | `state-spaces/mamba3-mimo-1.5b` |
| Base revision | `bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec` |
| Tokenizer | `meta-llama/Llama-3.1-8B` |
| Tokenizer revision | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| Serialization | Llama-3 headers, project format `llama3-headers-v1` |
| Training sequence limit | 2,048 tokens |
| CoSER revision | `7cc80430f92532cda85df45015a4aca8ecc068d0` |

The tokenizer does not change the model's Mamba architecture. Both adapters
started from the base model, independently of earlier HER adapters.

The dialogue dataset contains 5,197 training, 870 validation, and 1,574 test
examples from 85 selected books. Splits are by book, with the 11 official
out-of-distribution books held out for testing. Test examples were excluded
from training and checkpoint selection. The summary export contains 1,933
training and 330 validation records with matching book splits.

Bracketed thoughts were removed, while speech and parenthesized actions were
retained. Dialogue examples use a character/scene system context. Summary
targets describe facts, decisions, promises, and unresolved issues in the third
person. See [CoSER preparation](../dataset-studio/COSER.md) for selection,
provenance limits, token counts, and file layout.

## Training procedure

Both jobs ran sequentially on one NVIDIA H200. Combined SFT and validation took
38 minutes 14 seconds, excluding environment setup and preliminary checks.

| Setting | Both roles |
|---|---|
| Training | SFT, one epoch |
| LoRA | Rank 16, alpha 32, dropout 0 |
| Trainable modules | `mixer.in_proj`, `mixer.out_proj` |
| Trainable parameters | 6,770,688 FP32 parameters per adapter |
| Optimizer | AdamW, weight decay 0 |
| Learning rate | `1e-4`, 5% warmup, cosine decay |
| Microbatch / accumulation | 1 / 8 |
| Gradient norm limit | 1.0 |
| Padding | Right padding to a multiple of 256, masked from loss |
| Checkpoint and validation interval | Every 50 steps and at completion |

Base weights were frozen. Base projections used BF16; LoRA and designated Mamba
auxiliary parameters retained FP32. Non-reentrant gradient checkpointing and
chunked loss were enabled. Loss covered only assistant tokens and EOT, with
normalization over all target tokens in each accumulation group.

Before training, 14 CPU tests, two GPU regressions, and four short/long smoke
runs passed. Checks covered backward computation, finite gradients, LoRA matrix
updates, validation, checkpoint reload, and prepared-data consistency.

Recorded environment: Ubuntu 24.04, Python 3.12, CUDA Toolkit 12.8.93,
PyTorch 2.9.1+cu128, Transformers 4.57.6, TileLang 0.1.8, and Triton 3.5.1.
Run metadata records dependency versions, model revisions, and data/code hashes.
See the [training guide](../training/README.md) for the current workflow.

## Training results

The final checkpoint had the lowest validation loss in both runs.

| Role | Selected step | Token-weighted epoch train loss | Full validation loss | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| Consciousness | 650 | 2.287255 | 2.258817 | 3.98 GiB |
| Subconsciousness | 242 | 2.013634 | 1.877602 | 3.97 GiB |

Each adapter contains 96 tensors. Saved adapter tensors and optimizer states
were finite, and all LoRA B matrices changed during training.

A separate teacher-forced check used the same eight validation records per role:

| Role | Base loss | LoRA loss |
|---|---:|---:|
| Consciousness | 2.858389 | 2.362410 |
| Subconsciousness | 2.577748 | 2.005724 |

Loss measures next-token prediction, not an error percentage. Teacher forcing
supplies the correct preceding text; free generation instead conditions on the
model's own outputs and errors. Lower loss does not establish reliable dialogue
or summary accuracy.

## Generation checks

The initial comparison generated 32 responses: four inputs per role, base/LoRA,
and greedy/sampling decoding. Sampling used temperature 0.7, top-k 50, top-p
0.9, and seed 42. Responses were limited to 256 new tokens with EOT/EOS stopping;
repetition penalties and loop stopping were disabled.

| Role / model | Detected exact loops | End-marker completions |
|---|---:|---:|
| Consciousness base | 5/8 | 2/8 EOS |
| Consciousness LoRA | 0/8 | 8/8 EOT |
| Subconsciousness base | 5/8 | 1/8 EOS |
| Subconsciousness LoRA | 5/8 | 1/8 EOT |

Greedy prefixes were compared with the library generator, and repeated runs used
fresh caches. Relative L2 differences between full and cached logits ranged from
0.0059 to 0.0108. Earlier base-model diagnostics also found repetition without
adapters; fine-tuning alone does not explain the observed loops.

The final API uses greedy decoding, a fresh cache per request, EOT/EOS stopping,
and repetition penalties of 1.0 for dialogue and 1.2 for summaries. Penalties
apply to recent generated tokens. Exact-loop detection starts at 128 generated
tokens and looks for at least four copies spanning at least 64 tokens, with
periods up to 128 tokens. When detected, generation stops, extra copies are
removed, and `finish_reason` is `repetition`; usage counts all generated tokens.

With the final policy, four H200 summary checks had no detected exact loops;
two ended with EOT and two reached 512 tokens. On RTX 3060, nine API inputs were
each requested twice with matching text and usage: all five dialogue inputs
completed; two of four summaries completed and two reached the 256-token limit.
These small checks do not establish general determinism or eliminate
paraphrased repetition. The full 1,574-example dialogue test was not evaluated.

## Implementation fixes

| Area | Change |
|---|---|
| Single-token Mamba forward | Correct the shape passed to `step` and restore the sequence axis |
| Checkpointed MIMO backward | Read saved tensors once for non-reentrant checkpointing |
| Grouped Q/K reduction | Use int64 offsets before address multiplication |
| Min-p sampling | Use the correct row-wise relative threshold after temperature |
| Inference RMSNorm | Fix `num_warps=4` to avoid first-input-dependent kernel selection |
| Generation | Stop on both EOT and EOS; add repetition penalty and loop stopping |

The address-overflow fix concerns earlier, much longer sequences; current CoSER
examples fit within 2,048 tokens. Min-p was disabled in the looping checks, so
its fix does not explain those loops. Kernel and dtype checks did not identify
a gross full/step mismatch, but do not prove the absence of all implementation
errors. See [Mamba changes](../mamba-source/LOCAL_CHANGES.md) for maintained patches.

## Resources and limitations

Both loaded roles used about 5.69 GiB of allocated tensor memory after warmup
on an RTX 3060 12 GiB. This excludes other process/device memory and is not a
bound for every input length. An earlier backward check on that GPU requested
223,904 bytes of shared memory per block against 101,376 available; inference
success therefore does not establish training compatibility.

Remaining limitations include invented summary facts, changed participants,
plans reported as events, and outputs that finish only at the token limit.
Literary SFT does not establish general assistant quality. BF16 continuations
can differ between GPUs, and one epoch is an initial experiment rather than an
optimized training recipe. Experience-driven behavior remains an untested
project hypothesis.

Useful next evaluations are dialogue performance on the held-out test, summary
accuracy for participants and promises, plan/event distinction, and repetition
checks across more inputs and seeds. Generated summaries should not be treated
as verified experience automatically.

## Artifact layout

The original training artifact root is `outputs/brev-coser/`, with one directory
per role. Each role has `best.json`, a saved `tokenizer/`, run metadata, and
checkpoints. In this experiment, the selected checkpoints were
`consciousness/checkpoint-000650/` and `subconsciousness/checkpoint-000242/`.
Adapters are approximately 26 MiB each and use the native
`hafvillai-mamba3-lora-v1` format. The base model is stored separately.

Back up the entire run directory outside Git. Inference needs adapter weights,
configuration, tokenizer, and a compatible base model. Training recovery also
needs optimizer state, prepared data, matching code/configuration, and recorded
paths. Local evaluation outputs and run metadata are not part of the public
source tree. See the [API guide](../inference/README.md) and
[comparison guide](../hafvillai/README.md) for usage.

The standalone service uses a separate, checksummed copy under
`inference/artifacts/`. See the [inference guide](../inference/README.md) for
preparation and the private Brev deployment bundle.
