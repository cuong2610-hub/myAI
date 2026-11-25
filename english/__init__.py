"""English edition of the myAI project.

This package is completely separate from the original Vietnamese modules at the
repository root.  It provides a configurable GPT-style model, an English BPE
tokenizer, a resumable FineWeb-Edu streaming pipeline, pretraining, SFT,
inference, RAG and web search.

Only this package and the English artifact directories are created by the
English edition.  Original Vietnamese source files, tokenizers, checkpoints,
datasets and memory stores are never modified.
"""

from .config import (
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
