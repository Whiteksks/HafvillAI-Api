# LoRA training

This directory implements supervised fine-tuning (SFT) for two independent
Mamba-3 MIMO 1.5B adapters: character dialogue (**consciousness**) and conversation
summaries (**subconsciousness**). The base model remains frozen. Adapter files
use the native `hafvillai-mamba3-lora-v1` format loaded by `hafvillai_runtime/lora.py`, not PEFT.

## Responsibilities

| File | Responsibility |
|---|---|
| `config.py`, `configs/` | Training settings and data paths; model identity is inherited from the shared runtime |
| `data.py`, `prepare.py` | Chat serialization, assistant-only labels, splits, and prepared tensors |
| `model.py` | Training-only gradient checkpointing and chunked loss |
| `../hafvillai_runtime/` | Shared base loader, adapter modules, serialization, and decoder |
| `train.py`, `checkpoints.py` | Training, evaluation, atomic checkpoints, and resume |
| `run_pipeline.sh` | Regression checks, disposable smoke runs, and sequential role training |
| `setup_brev.sh` | GPU environment bootstrap on a compatible Linux machine |

## Environment

Run commands from the repository root. The recorded setup uses Linux, Python
3.12 with `venv` and development headers, an NVIDIA driver, and CUDA Toolkit
12.8 with `nvcc`. Dependencies are pinned in
[hafvillai/requirements.txt](../hafvillai/requirements.txt) and
[requirements.txt](requirements.txt). The editable Mamba install uses
[project-specific fixes](../mamba-source/LOCAL_CHANGES.md).

On a machine with these prerequisites:

```bash
bash training/setup_brev.sh
```

Ensure access to the configured base model and tokenizer. If authentication is
required, use `.venv/bin/hf auth login` locally. Download/cache the configured
revisions before offline preparation or training. Do not transfer virtual
environments or compiled GPU caches between machines; recreate the environment.

Training was verified on an H200. Inference was verified on an RTX 3060 12 GiB,
but an earlier MIMO backward check failed there because of per-block shared
memory requirements. Available VRAM alone does not establish training compatibility.

## Data and configurations

| Role | Configuration | Recorded train / validation examples |
|---|---|---:|
| Consciousness | `configs/coser_sft.json` | 5,197 / 870 |
| Subconsciousness | `configs/coser_subconsciousness.json` | 1,933 / 330 |

The 1,574 dialogue test examples are excluded from the trainer. Both roles use
matching book-level splits and a 2,048-token limit. See the
[CoSER guide](../dataset-studio/COSER.md) for preparation and split rules.

Datasets and prepared tensors are not included in Git. Generate the dialogue
export and summary targets first. Update the summary configuration's `dataset`
path to your generated `CoSER_summaries.jsonl`. Configuration paths resolve
from the repository root. For each new experiment, choose separate
`prepared_dir` and `output_dir` paths.

With exports and the tokenizer cache available:

```bash
bash training/run.sh prepare --config training/configs/coser_sft.json --offline
bash training/run.sh prepare --config training/configs/coser_subconsciousness.json --offline
bash dataset-studio/coser.sh check
.venv/bin/python -m unittest discover -s training/tests -v
```

## Training

The current CoSER configurations use rank 16, alpha 32, dropout 0, learning
rate `1e-4`, AdamW without weight decay, 5% warmup followed by cosine decay,
one epoch, microbatch 1, and gradient accumulation 8. Only LoRA matrices on
`mixer.in_proj` and `mixer.out_proj` are trained. LoRA parameters are FP32;
base projections use BF16 while Mamba's designated auxiliary parameters retain
FP32. These precision choices matter for the custom kernels.

Loss covers assistant tokens and EOT. System/user messages, headers, and padding
are masked. Accumulation is normalized by target tokens across the entire group.
Validation loss is not an error percentage or a guarantee of generation quality.

To run regression checks, short/long smoke tests, and both SFT jobs sequentially:

```bash
bash training/run_pipeline.sh training/configs/coser_sft.json training/configs/coser_subconsciousness.json
```

The pipeline uses offline model access. It skips completed runs and resumes
unfinished ones, so use new configuration/output paths for a new experiment.
For unattended runs, execute it inside `tmux`. Logs and pipeline status are
stored under `outputs/`.

## Checkpoints and inference

Each role's output directory contains `run.json`, `tokenizer/`, `metrics.jsonl`,
`best.json`, `latest.json`, `completed.json`, and checkpoint directories containing:

- `adapter.safetensors`: trained LoRA weights.
- `adapter_config.json`: role, base/tokenizer identity, and adapter configuration.
- `training.pt`: optimizer, scheduler, and random-number state for resume.

The base weights are stored separately and are not merged into the adapters.
For inference, preserve the checkpoint's weights and configuration plus the
role's saved tokenizer. Use `inference.prepare_artifacts` to select and copy checkpoints into the private
inference artifact directory. The service reads that directory independently of
training output paths and `best.json` files.

For an unfinished run on its original setup:

```bash
bash training/run.sh train --config training/configs/coser_sft.json --resume latest --offline
```

Resume requires matching code, data, dependency versions, configuration, and
recorded paths. Comment translation and configuration cleanup change the source
fingerprint, so checkpoints from earlier revisions may require their original
source checkout for strict training resume. Inference does not compare that
training fingerprint. Preserve the whole run and prepared data for recovery. All
training artifacts are ignored by Git and need separate backups.

The recorded CoSER runs selected steps 650 and 242, with validation losses
2.2588 and 1.8776. See the [experiment report](../docs/COSER_SFT_EXPERIMENT_REPORT.md)
for measurements and limitations.
