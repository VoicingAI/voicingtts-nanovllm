# VoicingTTS-NanoVLLM

Voicing AI's TTS inference engine — concurrent request batching, async API, GPU-only
(Triton + FlashAttention). Ships the `voicingtts_nanovllm` package.

```bash
pip install "git+https://github.com/VoicingAI/voicingtts-nanovllm.git"
```

Requires Linux/Windows + NVIDIA CUDA, Python >= 3.10, and `flash-attn` installed
separately (the package imports it at runtime; on Windows use a prebuilt wheel).

```python
from voicingtts_nanovllm import VoicingTTS
server = VoicingTTS.from_pretrained("/models/ttsv2", devices=[0])
```

`from_pretrained` returns an async pool inside an event loop, otherwise a sync pool,
and selects the runner from the `architecture` field of the checkpoint's `config.json`.

See `ARCHITECTURE.md` for internals and `CONTRIBUTING.md` to develop.
