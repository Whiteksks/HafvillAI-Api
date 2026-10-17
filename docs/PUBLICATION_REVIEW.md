# Source publication review

Review date: October 15, 2026.

## Included source

The repository contains the CoSER preparation and summary tools, the native LoRA
trainer, shared decoding and comparison utilities, the inference API, tests, and
English documentation. Obsolete HER code, the CoSER pilot, and frontend proxy
examples have been removed. Vendored Mamba source, attribution, and license are
retained. A repository-root project license has not been specified.

## Local artifact boundaries

Model downloads, trained adapters, optimizer states, datasets, conversations,
memory databases, caches, virtual environments, local credentials, and backup
archives are excluded by `.gitignore`. Blank `.env.example` templates and
relative VS Code settings are included. Ignoring files neither deletes them nor
backs them up, and does not prevent a forced add.

Current CoSER exports, prepared tensors, summary runs, and adapters are kept
locally. Gutenberg selection evidence is stored independently under
`dataset-studio/data/gutenberg/editions/`. Cloning the source does not restore
these artifacts.

## Verification

The source review found no recognizable provider tokens, private keys,
credential-bearing URLs, embedded secret values, model weights, or dataset
exports in the publication candidate. Secret checks used filename inspection
and common content patterns, with candidate matches reviewed; they do not cover
every possible secret format.

Cleanup verification passed 77 unit tests, the browser summary workflow, and
integrity checks for all 7,641 dialogue examples from 85 books. SHA-256 comparison
confirmed that 912 current CoSER/model artifact files remained unchanged; all
429 relocated Gutenberg evidence files matched their original hashes. No new
training or paid generation was performed during verification.

## Publishing

The cleaned source snapshot is committed as a single root commit, without the
earlier development commits as ancestors. The previous local Git history is
backed up separately, outside the source repository. Updating an existing remote
branch requires an explicit history replacement; creating the local snapshot
does not change remote branches or hosting visibility.

Before publishing future changes, review the staged file list and ignored-file
boundaries from the repository root:

```bash
git status --short
git diff --cached --stat
git ls-files --cached --ignored --exclude-standard
git ls-files --others --exclude-standard
```

`git ls-files --cached --ignored --exclude-standard` should produce no output.
Old remote branches, tags, releases, and attachments need separate review when
replacing an existing public repository history.

## Inference packaging follow-up: October 16, 2026

The API now lives in `inference/` and uses `hafvillai_runtime/`, shared with
training and comparison tools. The inference package does not import training
or Dataset Studio. Private artifacts under `inference/artifacts/` contain a
checksummed base model, the two selected adapters, and their tokenizers; the
standalone export under `inference/dist/` excludes local credentials and training
state. Both directories remain excluded from Git, and model artifacts are also
excluded from Docker build contexts.

Verification passed 82 unit tests and the real 7,641-example CoSER integrity
check. Compose configuration validates against the official schema. Docker
Engine and a CUDA GPU were not available in the preparation environment, so
container build, GPU generation, external networking, and TLS are not yet
verified on the future Brev machine. No cloud resources were created or started.
