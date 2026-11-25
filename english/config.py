"""Configuration objects and default paths for the English edition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

ARCH_VERSION = 1
"""Architecture version for English checkpoints.  Independent of the Vietnamese
edition's ``ARCH_VERSION`` so the two checkpoint families cannot be confused."""

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOKENIZER_DIR = REPO_ROOT / "tokenizer_en"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints_en"
DEFAULT_DATA_DIR = REPO_ROOT / "data_en"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs_en"
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory_en"

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"

# Special token ids are fixed by english.tokenizer.  Kept here so the model
# config can validate them without importing sentencepiece.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
SYSTEM_ID = 6


@dataclass(frozen=True)
class ModelConfig:
    """Architecture description.

    The default ``base`` preset is exactly 124,278,528 parameters with tied
    embeddings.  ``param_count()`` is the authoritative count; training prints
    it at startup and tests assert it.
    """

    vocab_size: int = 32768
    block_size: int = 1024
    n_layer: int = 14
    n_head: int = 12
    n_embd: int = 768
    n_inner: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    tie_weights: bool = True
    bias: bool = False
    n_kv_head: Optional[int] = None  # None -> n_head (standard MHA)

    def __post_init__(self) -> None:
        if self.vocab_size <= SYSTEM_ID:
            raise ValueError("vocab_size must be larger than the special-token ids")
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_head < 1 or self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        kv = self.n_head if self.n_kv_head is None else self.n_kv_head
        if kv < 1 or self.n_head % kv != 0:
            raise ValueError("n_kv_head must divide n_head")
        if self.n_inner < 1:
            raise ValueError("n_inner must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_head(self) -> int:
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    def param_count(self) -> int:
        """Exact parameter count including tied embeddings exactly once."""
        emb = self.vocab_size * self.n_embd
        kv_dim = self.kv_head * self.head_dim
        attn = (
            self.n_embd * self.n_embd  # q
            + self.n_embd * kv_dim  # k
            + self.n_embd * kv_dim  # v
            + self.n_embd * self.n_embd  # o
        )
        if self.bias:
            attn += self.n_embd + kv_dim + kv_dim + self.n_embd
        mlp = 3 * self.n_embd * self.n_inner
        if self.bias:
            mlp += 2 * self.n_inner + self.n_embd
        norms = 2 * self.n_embd
        per_block = attn + mlp + norms
        final_norm = self.n_embd
        lm_head = 0 if self.tie_weights else self.vocab_size * self.n_embd
        if not self.tie_weights and self.bias:
            lm_head += self.vocab_size
        return emb + self.n_layer * per_block + final_norm + lm_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


TINY_CONFIG = ModelConfig(
    vocab_size=512,
    block_size=128,
    n_layer=2,
    n_head=4,
    n_embd=128,
    n_inner=256,
    dropout=0.0,
)

BASE_CONFIG = ModelConfig(
    vocab_size=32768,
    block_size=1024,
    n_layer=14,
    n_head=12,
    n_embd=768,
    n_inner=2048,
    dropout=0.0,
)

WIDE_CONFIG = ModelConfig(
    vocab_size=32768,
    block_size=1024,
    n_layer=14,
    n_head=14,
    n_embd=896,
    n_inner=2432,
    dropout=0.0,
)

PRESETS = {
    "tiny": TINY_CONFIG,
    "base": BASE_CONFIG,
    "wide": WIDE_CONFIG,
}


def get_preset(name: str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown preset {name!r}; choose one of {sorted(PRESETS)}"
        ) from exc


def with_overrides(cfg: ModelConfig, **overrides: Any) -> ModelConfig:
    """Return a copy with only non-None overrides applied."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean) if clean else cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
