# Hafvillai inference

An inference-only HTTP service for the two native Mamba-3 MIMO 1.5B LoRA
adapters. Training, dataset preparation, and optimizer state are not required.
The service loads all model artifacts locally and does not contact a model API.

## Structure

| Path | Responsibility |
|---|---|
| `api/` | FastAPI routes, validation, model engine, and the shared GPU queue |
| `artifact_store.py`, `prepare_artifacts.py` | Select, copy, and verify portable model artifacts |
| `bundle.py` | Export a private, standalone deployment directory |
| `artifacts/` | Ignored local base weights, adapters, tokenizers, and checksums |
| `deploy/` | Docker image, Compose service, and optional HTTPS proxy |
| `examples/requests.json` | Dialogue and summary request examples |
| `tests/` | HTTP, artifact integrity, and standalone import checks |

The service uses `hafvillai_runtime/` and the patched `mamba-source/` library.
It has no imports from `training/`, `hafvillai/`, or Dataset Studio.

## Prepare local artifacts

Run commands from the repository root. Preparation and checksum verification use
Python's standard library and do not require a GPU. The source run directories
must contain `best.json`, selected checkpoints, and saved tokenizers. The pinned
base-model snapshot must already be in the local Hugging Face cache.

```bash
python3 -m inference.prepare_artifacts prepare
python3 -m inference.prepare_artifacts verify
```

Defaults read adapters from `outputs/brev-coser/` and the base snapshot from
`hafvillai/.cache/huggingface/hub/`. Use `--source-root`, `--base-directory`, and
`--destination` for other locations. Existing destinations are never overwritten;
prepare a new directory to replace a deployment safely.

The resulting ignored directory is self-contained:

```text
inference/artifacts/
  manifest.json
  base/
    config.json
    pytorch_model.bin
  consciousness/
    adapter.safetensors
    adapter_config.json
    tokenizer/
  subconsciousness/
    adapter.safetensors
    adapter_config.json
    tokenizer/
```

Files are copied without merging or changing weights. The manifest records the
pinned model identity, role steps, and SHA-256 checksums. It does not contain
original machine paths. Optimizer states, datasets, run logs, and credentials
are excluded. The full artifact directory is about 2.9 GiB for this model;
keep a private backup and transfer it outside Git.

## Run locally

Linux/WSL2, Python 3.12, CUDA Toolkit 12.8, and an NVIDIA GPU with CUDA and BF16
support are required for generation. Use the existing root `.venv`, or run
`bash inference/setup.sh` on a machine with these prerequisites. The setup
installs the pinned runtime and editable Mamba library, without training tools.

Copy `.env.example` to `inference/.env` if custom settings are needed. Then:

```bash
bash inference/run.sh
```

From another terminal:

```bash
curl http://127.0.0.1:8000/health
bash inference/run.sh smoke --role all --max-new-tokens 128
```

First startup verifies artifact checksums and warms GPU kernels. One process
loads both roles; a bounded queue serializes GPU access. Disabling the summary
role loads only the dialogue model. The artifact manifest still describes both
roles. No conversation history or persistent memory is stored by this service.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness, enabled roles, adapter steps, queue state, and limits |
| POST | `/chat` | Character dialogue using the consciousness adapter |
| POST | `/summarize` | Conversation summary using the subconsciousness adapter |
| GET | `/docs` | Interactive schema reference |

```bash
curl http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello."}],"max_new_tokens":128}'
```

`messages` contains `{role, content}` objects. An optional `system` message is
allowed only first; the last message must be `user`. Include relevant history
on each request. `request_id` identifies a request, not a stored chat.
`/summarize` accepts the same schema. Examples are in `examples/requests.json`.

The complete JSON response contains `content`, `role`, `adapter_step`,
`request_id`, `finish_reason`, token `usage`, and `generation_seconds`.
There is no streaming or OpenAI-compatible `choices` wrapper. End reasons are
`eot`, `eos`, `max_new_tokens`, and `repetition`. The latter removes extra copies
of an exact loop while usage still counts all generated tokens.

## Settings

Environment variables override `inference/.env`. Relative artifact paths resolve
from the repository or exported bundle root.

| Variable | Default |
|---|---|
| `HAFVILLAI_ARTIFACTS_DIR` | `inference/artifacts` |
| `HAFVILLAI_HOST` / `HAFVILLAI_PORT` | `127.0.0.1` / `8000` |
| `HAFVILLAI_API_KEY` | Empty for local loopback access |
| `HAFVILLAI_ENABLE_SUBCONSCIOUSNESS` | `1` |
| `HAFVILLAI_MAX_INPUT_TOKENS` / `HAFVILLAI_MAX_NEW_TOKENS` | `1536` / `512` |
| `HAFVILLAI_MAX_QUEUE_SIZE` | `4` waiting requests |
| `HAFVILLAI_MAX_BODY_BYTES` | `1048576` (1 MiB) |
| `HAFVILLAI_CONSCIOUSNESS_REPETITION_PENALTY` | `1.0` |
| `HAFVILLAI_SUBCONSCIOUSNESS_REPETITION_PENALTY` | `1.2` |

Non-loopback listeners require a nonempty API key, including when the app is
created directly rather than through the launcher. Send it in
`Authorization: Bearer <key>` on generation requests. Keep it on the Next.js
server, never in browser code or `NEXT_PUBLIC_` variables.

The combined configured input/output limit remains 2,048 tokens; inputs are
not truncated. Requests default to 256 new tokens, subject to the server limit.
Decoding controls reduce exact loops but do not establish factual accuracy;
summary outputs still need review before future memory integration.

HTTP errors: `401` authentication, `413` input/body limit, `422` validation or
output limit, `429` queue full with `Retry-After`, and `503` generation/role failure.

## Export and deploy

```bash
python3 -m inference.bundle
```

This creates `inference/dist/hafvillai-inference/`, containing only inference
source, the shared runtime, patched Mamba, and private model artifacts. Local
`.env` files and tests are excluded. Transfer that directory privately to Brev;
it can run without the rest of this repository. A fresh `.env` is configured on
the destination. See [deploy/README.md](deploy/README.md) for the deployment steps.

In a source checkout, run tests with:

```bash
bash inference/run.sh test
PYTHONPATH=. .venv/bin/python -m unittest discover -s hafvillai/tests -v
```

CPU tests do not replace a container build and GPU smoke test on the selected
Brev machine. Creating this package does not create, start, or charge an instance.
