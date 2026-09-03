"""voicingtts_nanovllm.engine

Model-agnostic inference runtime core.

This package contains the components that make inference work end-to-end:

- :mod:`voicingtts_nanovllm.engine.sequence`: per-request state machine + KV mapping.
- :mod:`voicingtts_nanovllm.engine.block_manager`: KV-cache block pool + prefix cache.
- :mod:`voicingtts_nanovllm.engine.scheduler`: batching policy + preemption.
- :mod:`voicingtts_nanovllm.engine.model_runner`: GPU execution abstraction.
- :mod:`voicingtts_nanovllm.engine.llm_engine`: orchestrates the engine step loop.

The intent is that model implementations only need to provide a thin adapter
layer ("preprocess" and "postprocess") while the runtime handles scheduling,
memory management, and execution.

Reference implementation
------------------------
For a complete, working example of how to plug a model family into this runtime,
see ``voicingtts_nanovllm/models/voicingtts``.
"""
