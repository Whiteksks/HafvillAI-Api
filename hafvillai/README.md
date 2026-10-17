# Decoding and model comparison

This directory provides the shared decoder used by the
[inference API](../inference/README.md) and tools for comparing the base
model with LoRA adapters.

| File | Responsibility |
|---|---|
| `../hafvillai_runtime/decoding.py` | Shared token generation, repetition controls, and inference normalization |
| `compare.py` | Base/LoRA comparisons, generation checks, and result export |
| `comparison_metrics.py` | Repetition and output-format metrics |
| `build_coser_eval_cases.py` | Evaluation case preparation from CoSER exports |
| `examples/coser_checks.json` | Small dialogue and summary check suite |
| `requirements.txt` | Pinned GPU runtime dependencies |

## Comparing models

With the root GPU environment, base model, tokenizer, and adapters available,
run from the repository root:

```bash
bash hafvillai/run.sh compare --role consciousness --suite --decode both --max-new-tokens 256
```

Defaults use the CoSER configuration, an adapter selected by
`outputs/brev-coser/<role>/best.json`, and `examples/coser_checks.json` in this
directory. Each comparison creates a directory under `outputs/comparison/`.
Use `--mode base|lora|both`, `--adapter`, and `--messages-file` to choose model
variants, checkpoints, and custom inputs. The combined token budget is limited
to 2,048; inputs are not truncated automatically.

To use the API's summary repetition controls:

```bash
bash hafvillai/run.sh compare --role subconsciousness --mode lora --suite \
  --repetition-penalty 1.2 --stop-on-repetition --max-new-tokens 512
```

Ordinary comparisons default to penalty 1.0 with repetition stopping disabled
to expose the underlying model behavior. `--verify` compares greedy prefixes
and full/cached logits only when these controls are disabled. Repetition and
format metrics do not measure factual accuracy.

## Checks

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s hafvillai/tests -v
```

See the [environment setup](../training/README.md) and
[experiment results](../docs/COSER_SFT_EXPERIMENT_REPORT.md).
