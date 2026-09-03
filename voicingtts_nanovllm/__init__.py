from voicingtts_nanovllm.llm import VoicingTTS

try:
    from voicingtts_nanovllm._version import version as __version__
except Exception:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("voicingtts-nanovllm")
    except Exception:
        __version__ = "0.0.0"

__all__ = [
    "VoicingTTS",
    "__version__",
]
