# Project-specific Mamba fixes

Upstream baseline: `e9594ce1c732d97440f0332fdc43170a2294dbfa`.
These modifications are used through the editable library install. They do not
change the base weights, model architecture, or MIMO rank.

Paths in the table are relative to this directory.

| File | Fix |
|---|---|
| `mamba_ssm/modules/mamba3.py` | Pass single-token input to `step` as `(batch, dim)` and restore the output sequence axis; fixes a shape error |
| `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py` | Read `ctx.saved_tensors` once and reuse the tuple for compatibility with non-reentrant gradient checkpointing |
| `mamba_ssm/ops/triton/mamba3/grouped_head_reduction.py` | Convert address offsets to int64 before multiplication to prevent overflow on long sequences |
| `mamba_ssm/utils/generation.py` | Apply the min-p threshold `max(logits) + log(min_p)` per row after temperature scaling, including top-k filtering |

Min-p was disabled in the earlier looping generations, so its fix does not
explain those repetitions. The shared project decoder additionally fixes
RMSNorm `num_warps=4` in `hafvillai_runtime/decoding.py`; that setting is applied outside
this library.

## Verification

From the repository root with the GPU environment available:

```bash
.venv/bin/python mamba-source/tests/modules/test_mamba3_decode_shapes.py
.venv/bin/python -m unittest discover -s training/tests -v
PYTHONPATH=. .venv/bin/python -m unittest discover -s hafvillai/tests -v
```

The recorded H200 run passed GPU reduction regressions, including addresses
above `2**31`, training smoke tests, and full SFT. RTX 3060 checks covered
inference; backward compatibility was limited by shared memory. See the
[experiment report](../docs/COSER_SFT_EXPERIMENT_REPORT.md).
