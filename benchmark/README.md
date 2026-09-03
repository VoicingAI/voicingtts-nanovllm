# Benchmark

Throughput and latency harnesses for the engine. All need a real GPU and a checkpoint.

```bash
# single-shot inference
uv run python benchmark/bench_inference.py --model ~/voicingtts --concurrency 4 --iters 5 --warmup 1 --seed 42

# open-loop (fixed arrival rate)
uv run python benchmark/bench_open_loop_users.py --model ~/voicingtts --rps 30 --duration-s 60

# closed-loop (fixed user count)
uv run python benchmark/bench_closed_loop_users.py --model ~/voicingtts --num-users 60 --duration-s 60 --warmup-s 5
```

Pass `--seed` for reproducible runs. Results vary with GPU, batch size and `inference_timesteps`.
