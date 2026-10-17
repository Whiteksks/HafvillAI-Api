# Hafvillai

An experimental AI companion built on Mamba-3 MIMO 1.5B, with two independent
LoRA adapters trained on English literary dialogue and conversation summaries.

The project explores whether accumulated experience can shape a character's
future responses. Memory is intended to provide experience rather than a fixed
personality instruction. Persistent memory and experience-driven behavior are
research goals; they are not implemented or demonstrated by the current prototype.

## Components

| Directory | Responsibility |
|---|---|
| [inference/](inference/README.md) | HTTP endpoints for dialogue, summarization, and model health |
| [hafvillai/](hafvillai/README.md) | Base-model/LoRA comparison tools and output metrics |
| [hafvillai_runtime/](hafvillai_runtime/README.md) | Shared model loading, native LoRA, serialization, and decoding |
| [training/](training/README.md) | Training data preparation, SFT, validation, and checkpoints |
| [dataset-studio/](dataset-studio/README.md) | Local dataset preparation, inspection, and summary generation |
| [mamba-source/](mamba-source/README.md) | Vendored Mamba library with project-specific fixes |
| [docs/](docs/COSER_SFT_EXPERIMENT_REPORT.md) | Experiment results and known limitations |

The **consciousness** role generates dialogue as a selected character. The
**subconsciousness** role summarizes conversations in the third person. These
are component names, not claims of demonstrated consciousness. First-person
experience recording, background memory processing, and memory retrieval are
not yet implemented.

## Running the API

Run commands from the repository root unless a component guide says otherwise.
GPU execution requires Linux or WSL2, Python 3.12, an NVIDIA GPU with CUDA and
BF16 support, and the pinned environment described in the
[inference guide](inference/README.md).

**A source checkout alone cannot serve requests.** Base-model files, the
tokenizer, and trained adapters are not distributed in Git. Prepare the private
`inference/artifacts/` directory from existing checkpoints and the pinned cached
base, following the [inference guide](inference/README.md). The service then runs
offline without importing training code. A standalone export and Brev deployment
configuration are included; no cloud resource starts automatically.

With the environment and model artifacts available:

```bash
bash inference/run.sh
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
bash inference/run.sh smoke --role all --max-new-tokens 128
```

The API returns complete JSON responses and uses one shared GPU queue. It does
not store conversation history; clients provide the relevant messages on every
request.

## Current results

The recorded CoSER experiment trained dialogue and summary adapters for 650
and 242 steps, respectively. Dialogue completed all eight small-sample checks
without detected exact loops. Summaries still repeat without decoding controls
and can invent facts or confuse plans with completed events. These results do
not establish reliable memory or general assistant quality.

See the [experiment report](docs/COSER_SFT_EXPERIMENT_REPORT.md) for measurements,
hardware requirements, and evaluation limits, and the
[CoSER guide](dataset-studio/COSER.md) for dataset preparation.

## Source and local artifacts

Git contains source code, configuration templates, small examples, and
documentation. Local credentials, datasets, conversations, memory databases,
model downloads, LoRA checkpoints, caches, and run outputs are ignored. Keep
separate backups of training artifacts; cloning this repository will not
restore them. See the [publication review](docs/PUBLICATION_REVIEW.md) for the
source boundaries, verification results, and publishing notes.

Vendored Mamba attribution and license are retained in
[mamba-source/AUTHORS](mamba-source/AUTHORS) and
[mamba-source/LICENSE](mamba-source/LICENSE).
