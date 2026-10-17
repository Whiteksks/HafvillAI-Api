# Dataset Studio

A local tool for preparing literary dialogue datasets and target summaries.
The training pipeline uses [CoSER](COSER.md). Dataset Studio is not the
chatbot's runtime memory store.

## Setup and launch

From the repository root, create the Studio's separate CPU environment:

```bash
python3.12 -m venv dataset-studio/.venv
dataset-studio/.venv/bin/python -m pip install -r dataset-studio/requirements.txt
bash dataset-studio/run.sh
```

Open `http://localhost:8765`. To use another port:

```bash
STUDIO_PORT=8766 bash dataset-studio/run.sh
```

Optional summary-generation settings go in `dataset-studio/.env`, using
[.env.example](.env.example) as a template. Restart Studio after editing it.
Keep real API keys out of Git.

## Responsibilities

| Files | Responsibility |
|---|---|
| `server.py`, `static/` | Local HTTP server and browser interface |
| `coser_download.py`, `coser_catalog.py`, `coser_sft.py` | CoSER download, selection, and dialogue export |
| `coser_summaries.py`, `summaries.py` | Summary inputs, external generation, and resumable exports |
| `gutenberg.py` | Book catalog matching and cached Gutenberg edition screening |
| `data_utils.py` | Atomic JSON writes, hashing, and stable text identifiers |
| `tests/` | Dataset, split, export, and summary workflow checks |

## CoSER preparation

The CoSER CLI uses the repository's root `.venv`, described in the
[training guide](../training/README.md), rather than the Studio environment.
The Gutenberg catalog CSV must be available at
`dataset-studio/data/pg_catalog.csv`. Edition evidence is stored independently
under `dataset-studio/data/gutenberg/editions/`. From the repository root:

```bash
bash dataset-studio/coser.sh download
bash dataset-studio/coser.sh catalog --refresh-editions
bash dataset-studio/coser.sh prepare
bash dataset-studio/coser.sh check
```

Catalog refresh downloads missing or stale Gutenberg edition evidence.
Preparation includes offline tokenizer loading, so cache the configured
tokenizer first. `check` verifies existing exports and prepared tensors; it
does not download missing data or train a model. Selection rules, book-level
splits, and recorded dataset sizes are documented in [COSER.md](COSER.md).

After preparing the SFT export, create the scene-level summary inputs:

```bash
PYTHONPATH=. .venv/bin/python dataset-studio/coser_summaries.py
```

## Summary generation

The summary interface processes complete, clean CoSER scenes. It supports batches
of 100 conversations or the full dataset, configurable concurrency, and editable
generation instructions. External API generation consumes the provider's quota;
it is not needed to run existing adapters.

- Start creates a new run.
- Stop retains completed responses.
- Resume uses the saved source snapshot and settings and skips saved records.
- Changing generation rules requires a new run.

Run artifacts are stored under `dataset-studio/data/summaries/runs/`, including
JSONL exports, SQLite state, and recovery files. Preserve the whole run directory
when backing it up. CoSER exports use `CoSER_summaries.jsonl`; retain `example_id`,
`split`, `split_group`, and `source` metadata for training. Thinking blocks,
service tokens, and incomplete responses are excluded from summary targets.

Datasets, generated summaries, local state, and credentials are not distributed
in Git. The [experiment report](../docs/COSER_SFT_EXPERIMENT_REPORT.md) describes
the resulting adapters and their limitations.

## Tests

The unit suite uses temporary datasets and mocked external responses:

```bash
PYTHONPATH=.:dataset-studio .venv/bin/python -m unittest discover -s dataset-studio/tests -v
```

For the browser workflow, install `requirements-dev.txt` in the Studio environment
and install its Playwright Chromium browser, then run:

```bash
dataset-studio/.venv/bin/python dataset-studio/tests/browser_summaries_smoke.py
```
