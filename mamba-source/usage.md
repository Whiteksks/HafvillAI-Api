# Mamba integration in Hafvillai

The project uses the vendored library through the root Python environment.
[Training setup](../training/README.md) installs it in editable mode, so the
[documented fixes](LOCAL_CHANGES.md) are used by training and inference.

## Loading and training

`hafvillai_runtime/model.py` downloads the configured base-model revision, constructs
`MambaLMHeadModel`, loads its weights, and checks kernel dtypes. Base projections
use BF16; designated Mamba auxiliary parameters stay FP32. Do not convert the
entire model to BF16, because that changes parameters required by the kernels.

`hafvillai_runtime/lora.py` attaches native LoRA modules to `mixer.in_proj` and
`mixer.out_proj`. Each role starts independently from the frozen base model;
adapters are saved separately. Gradient checkpointing and chunked language-model
loss are implemented by the project training layer.

## Inference

`hafvillai_runtime/decoding.py` manages token generation and fresh inference state,
applies optional repetition penalties, and stops on EOT/EOS or an exact loop.
Both the comparison tools and `inference/api/engine.py` use this
shared decoder. The API loads a separate base-plus-adapter model for each enabled
role and serializes generation through one GPU queue.

For commands, see the [API guide](../inference/README.md) and
[comparison guide](../hafvillai/README.md). For upstream attribution and library
structure, see [README.md](README.md).
