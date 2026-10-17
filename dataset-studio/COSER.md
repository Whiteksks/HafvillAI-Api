# CoSER dataset preparation

The recorded experiment uses `Neph0s/CoSER` at revision
`7cc80430f92532cda85df45015a4aca8ecc068d0`. Of 773 downloaded book files, 85 books
passed the project's Gutenberg-based selection. This selection does not establish
the exact edition or translation provenance of every dialogue. Dataset files
and generated exports are not included in this repository.

## Dialogue dataset

| Measure | Train | Validation | Test |
|---|---:|---:|---:|
| Books | 66 | 8 | 11 |
| Scenes | 1,933 | 330 | 591 |
| SFT examples | 5,197 | 870 | 1,574 |
| Assistant turns | 22,028 | 3,764 | 6,416 |
| Target tokens, including EOT | 747,170 | 132,543 | 208,051 |

The 7,641 examples are character-specific views of 2,854 scenes, not independent
conversations. Bracketed thoughts are removed; speech and parenthesized actions
are retained. The system message identifies the character, book, and scene;
the assistant plays one character and user messages contain the other speakers.
Malformed scenes and duplicates are excluded. Examples fit within 2,048 tokens
without truncating exchanges. Median training length is 544 tokens, maximum
1,571; median assistant target length is 29 tokens including EOT.

Splitting is by book with seed 42. All characters, scenes, and windows from a
book stay in the same split. The 11 official out-of-distribution books are held
out for testing. `sft.jsonl` contains only train and validation examples; test
examples are excluded from training and checkpoint selection.

## Files

Paths below are relative to the repository root.

| Path | Purpose |
|---|---|
| `dataset-studio/data/coser/sft-v1/{train,validation,test}.jsonl` | Readable split exports |
| `dataset-studio/data/coser/sft-v1/sft.jsonl` | Dialogue trainer input |
| `dataset-studio/data/coser/sft-v1/` | Selection report, splits, catalog, hashes, and evaluation inputs |
| `training/data/coser-sft-v1/consciousness/` | Prepared tensors, labels, and manifest |
| `training/configs/coser_sft.json` | Dialogue training configuration |
| `dataset-studio/data/gutenberg/editions/` | Cached book cards and texts used for selection checks |
| `dataset-studio/data/summaries/runs/<run>/CoSER_summaries.jsonl` | Generated summary export |
| `training/configs/coser_subconsciousness.json` | Summary training configuration |

The summary configuration currently references the recorded experiment's run
directory. Set its `dataset` field to your own generated export before preparing
a new summary run; keep its split metadata intact.

## Summary dataset

The recorded export contains 1,933 training and 330 validation summaries, with
214,758 training target tokens. Book-level splits match the dialogue dataset.
Targets describe participants, facts, decisions, promises, and unresolved issues
in the **third person**. First-person experience recording and long-term memory
are future project work.

## Preparation and checks

With the root training environment, cached tokenizer, and Gutenberg catalog
CSV at `dataset-studio/data/pg_catalog.csv`, run from the repository root:

```bash
bash dataset-studio/coser.sh download
bash dataset-studio/coser.sh catalog --refresh-editions
bash dataset-studio/coser.sh prepare
bash dataset-studio/coser.sh check
```

Edition evidence is independent of generated dialogue data. Catalog refresh can
fetch missing or stale evidence; integrity checks use the saved evidence without
refreshing it.

Checks cover hashes, split groups, test exclusion, dialogue reconstruction,
length limits, tensor consistency, and assistant-only labels. Generate summary
targets separately through [Dataset Studio](README.md), then follow the
[training guide](../training/README.md). See the
[experiment report](../docs/COSER_SFT_EXPERIMENT_REPORT.md) for results.
