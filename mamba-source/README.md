# Vendored Mamba library

This directory contains the Mamba state-space model implementation from
[state-spaces/mamba](https://github.com/state-spaces/mamba), based on upstream
commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`, with project-specific fixes.
Hafvillai uses its Mamba-3 MIMO layers, model loader, GPU kernels, and inference
state handling.

## Project integration

The root Python environment installs this directory in editable mode through
[hafvillai/requirements.txt](../hafvillai/requirements.txt). Follow the pinned
[environment setup](../training/README.md) rather than substituting a different
library version: training and inference depend on the local fixes and compatible
kernel dependencies.

| Directory | Responsibility |
|---|---|
| `mamba_ssm/models/` | Model configuration and language-model backbone |
| `mamba_ssm/modules/` | Mamba layers, including Mamba-3 |
| `mamba_ssm/ops/` | GPU operations implemented with TileLang, Triton, and CuTe |
| `mamba_ssm/utils/` | Generation, inference state, and loading utilities |
| `csrc/` | CUDA extension sources retained from upstream |
| `tests/` | Upstream tests and project regression cases |
| `benchmarks/`, `evals/` | Upstream benchmark and evaluation utilities |

See [LOCAL_CHANGES.md](LOCAL_CHANGES.md) for the maintained fixes and
[usage.md](usage.md) for how project components call this library. The vendored
`.github/` files belong to upstream tooling; project-level GitHub Actions are
not configured at the repository root.

## Attribution

The original implementation is by the Mamba authors and contributors, listed
in [AUTHORS](AUTHORS). The upstream Apache 2.0 [LICENSE](LICENSE) is retained.

Research references retained from the upstream documentation:

- Albert Gu and Tri Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Tri Dao and Albert Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060).
- Aakash Lahoti and collaborators, [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569).
