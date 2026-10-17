# Shared model runtime

This package is the common model layer used by training, comparison tools, and
the standalone inference service. It does not depend on the training package.

| File | Responsibility |
|---|---|
| `config.py` | Pinned base/tokenizer identity and serialization format |
| `files.py` | File checksums |
| `serialization.py` | Llama-3 headers and assistant-only target labels |
| `model.py` | Local or pinned-cache base loading and kernel dtype checks |
| `lora.py` | Native LoRA modules, parameter selection, and save/load format |
| `decoding.py` | Cached generation, stop markers, repetition penalty, and loop detection |
| `requirements.txt` | Shared GPU dependency versions |

Adapters keep the `hafvillai-mamba3-lora-v1` format. Base projections remain BF16;
LoRA matrices and designated Mamba auxiliary parameters remain FP32. Moving code
to this package does not merge adapters or change weights. Training fingerprints
include this package, so strict resume from an earlier source layout requires
the corresponding archived code.
