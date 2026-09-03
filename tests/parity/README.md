# Parity tests

GPU integration tests that compare engine output against a reference run. They need
model weights and CUDA, so they are excluded from CPU CI.

No `.py` sources are present in this snapshot — git history for `tests/parity/` did not
surface recoverable tracked sources, and any `__pycache__` here is stale bytecode.

If the sources are restored:

```bash
uv run pytest tests/parity -m gpu --model /path/to/voicingtts
```

Runtime code under test lives in `voicingtts_nanovllm.models.voicingtts2`.
