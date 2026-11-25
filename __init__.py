"""English-only GPT project.

The repository provides a configurable GPT-style model, an English BPE
tokenizer, a resumable FineWeb-Edu streaming pipeline, pretraining, SFT,
inference, RAG and web search.  All source code lives in this package and all
generated artifacts use the sibling ``*_en`` directories.
"""

from config import (
    ARCH_VERSION,
    BASE_CONFIG,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MEMORY_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_TOKENIZER_DIR,
    ModelConfig,
    get_preset,
)

__all__ = [
    "ARCH_VERSION",
    "BASE_CONFIG",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MEMORY_DIR",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_TOKENIZER_DIR",
    "ModelConfig",
    "get_preset",
]
