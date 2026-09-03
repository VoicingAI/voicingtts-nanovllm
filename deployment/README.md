# VoicingTTS FastAPI Service

Optional HTTP wrapper around `voicingtts_nanovllm`. Not packaged for distribution.

```bash
uv sync                                    # from the repo root
uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
uv run pytest deployment/tests -q
```

Docker:

```bash
docker build -f deployment/Dockerfile -t voicingtts-nanovllm-deployment:latest .
docker run --rm --gpus all -p 8000:8000 voicingtts-nanovllm-deployment:latest
```

Endpoints: `/health`, `/info`, `/metrics`, and prompt-wav latent encoding.
Configuration is read from environment variables; see `app/core/config.py`.
Example client: `uv run python deployment/client.py`.
