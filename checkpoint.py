"""Atomic, fully resumable checkpoint management for the English edition.

Every English checkpoint carries an explicit ``edition: "en"`` marker plus the
architecture version, the model configuration, the tokenizer fingerprint, the
AMP scaler state, all RNG states and the resumable data-stream state.  This is
deliberately stricter than earlier checkpoint formats, which lacked several of
these fields.

Only trusted checkpoint files should be loaded: the payload includes Python
objects (NumPy RNG tuples), so ``weights_only=False`` is required.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import ARCH_VERSION, ModelConfig
from .utils import capture_rng_state, restore_rng_state, unwrap_model

EDITION = "en"
CHECKPOINT_FORMAT = 1
SUPPORTED_FORMATS = (1,)

# Fields that define weight compatibility.  ``dropout`` is excluded: it does
# not change parameter shapes or semantics of a trained model.
ARCH_KEYS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "n_inner",
    "norm_eps",
    "rope_base",
    "tie_weights",
    "bias",
    "n_kv_head",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt or incompatible."""


@dataclass
class CheckpointInfo:
    path: str
    edition: Optional[str]
    format: Optional[int]
    arch_version: Optional[int]
    mode: Optional[str]
    step: int = 0
    tokens_seen: int = 0
    best_val: float = float("inf")
    config: Dict[str, Any] = field(default_factory=dict)
    tokenizer_sha256: Optional[str] = None
    has_optimizer: bool = False
    has_scaler: bool = False
    has_rng: bool = False
    has_data_state: bool = False
    train_args: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoadedCheckpoint:
    path: str
    info: CheckpointInfo
    optimizer_loaded: bool = False
    scaler_loaded: bool = False
    rng_loaded: bool = False
    data_state: Any = None
    rng_by_rank: Optional[List[Dict[str, Any]]] = None
    train_args: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


def _torch_load(path: str | Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def atomic_save(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` via a temporary file in the same directory, then rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return p


def _tokenizer_sha(tokenizer_fp: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tokenizer_fp:
        return None
    sha = tokenizer_fp.get("sha256")
    return str(sha) if sha else None


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: ModelConfig,
    mode: str,
    step: int,
    tokens_seen: int = 0,
    best_val: float = float("inf"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    tokenizer_fp: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    data_state: Optional[Dict[str, Any]] = None,
    train_args: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    include_optimizer: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a complete resumable checkpoint.

    ``include_optimizer=False`` is useful for the "best" checkpoint: it halves
    the file size and is still enough for inference/fine-tuning.  Resuming a
    training run must use a checkpoint saved with the optimizer.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if mode not in {"pretrain", "sft"}:
        raise ValueError("mode must be 'pretrain' or 'sft'")
    state = unwrap_model(model).state_dict()
    obj: Dict[str, Any] = {
        "edition": EDITION,
        "format": CHECKPOINT_FORMAT,
        "arch_version": int(ARCH_VERSION),
        "mode": str(mode),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val": float(best_val),
        "config": config.to_dict(),
        "model": state,
        "optimizer": optimizer.state_dict() if (optimizer is not None and include_optimizer) else None,
        "scaler": (
            scaler.state_dict()
            if (scaler is not None and include_optimizer and getattr(scaler, "is_enabled", lambda: False)())
            else None
        ),
        "tokenizer_fingerprint": tokenizer_fp,
        "tokenizer_sha256": _tokenizer_sha(tokenizer_fp),
        "rng": rng_state if (rng_state is not None and include_optimizer) else None,
        "data": data_state if include_optimizer else None,
        "train_args": dict(train_args or {}),
        "history": list(history or [])[-40:],
    }
    if extra:
        obj.update(extra)
    return atomic_save(obj, path)


def peek_checkpoint(path: str | Path, map_location: Any = "cpu") -> CheckpointInfo:
    """Read checkpoint metadata without touching model weights validation."""
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"Checkpoint not found: {p}")
    try:
        obj = _torch_load(p, map_location)
    except Exception as exc:
        raise CheckpointError(f"Could not read checkpoint {p}: {exc}") from exc
    if not isinstance(obj, dict):
        raise CheckpointError(f"Checkpoint {p} is not a dictionary")
    config = obj.get("config") or {}
    raw_best = obj.get("best_val", float("inf"))
    best_val = float("inf") if raw_best is None else float(raw_best)
    return CheckpointInfo(
        path=str(p),
        edition=obj.get("edition"),
        format=obj.get("format"),
        arch_version=obj.get("arch_version"),
        mode=obj.get("mode"),
        step=int(obj.get("step", 0) or 0),
        tokens_seen=int(obj.get("tokens_seen", 0) or 0),
        best_val=best_val,
        config=dict(config) if isinstance(config, dict) else {},        tokenizer_sha256=obj.get("tokenizer_sha256")
        or _tokenizer_sha(obj.get("tokenizer_fingerprint")),
        has_optimizer=obj.get("optimizer") is not None,
        has_scaler=obj.get("scaler") is not None,
        has_rng=obj.get("rng") is not None,
        has_data_state=obj.get("data") is not None,
        train_args=dict(obj.get("train_args") or {}),
        history=list(obj.get("history") or []),
    )


def config_mismatches(saved: Dict[str, Any], current: ModelConfig) -> List[str]:
    cur = current.to_dict()
    out: List[str] = []
    for key in ARCH_KEYS:
        if key not in saved:
            continue
        if saved[key] != cur[key]:
            out.append(f"{key}: checkpoint={saved[key]!r} current={cur[key]!r}")
    return out


def _check_identity(info: CheckpointInfo, expected_mode: Optional[str], tokenizer_fp: Optional[Dict[str, Any]]) -> None:
    if info.edition != EDITION:
        raise CheckpointError(
            f"{info.path} is not an English-edition checkpoint "
            f"(edition={info.edition!r}); refusing to load it."
        )
    if info.format not in SUPPORTED_FORMATS:
        raise CheckpointError(
            f"Unsupported checkpoint format {info.format}; supported: {SUPPORTED_FORMATS}"
        )
    if info.arch_version != ARCH_VERSION:
        raise CheckpointError(
            f"Architecture version mismatch: checkpoint={info.arch_version} "
            f"current={ARCH_VERSION}"
        )
    if expected_mode is not None and info.mode != expected_mode:
        raise CheckpointError(
            f"Checkpoint mode is {info.mode!r}, expected {expected_mode!r}. "
            + (
                "Use --resume with a matching checkpoint."
                if expected_mode == "pretrain"
                else "Start SFT from a matching SFT checkpoint."
            )
        )
    current_sha = _tokenizer_sha(tokenizer_fp)
    if current_sha and info.tokenizer_sha256 and current_sha != info.tokenizer_sha256:
        raise CheckpointError(
            "Tokenizer mismatch: checkpoint sha256="
            f"{info.tokenizer_sha256[:12]}..., current sha256={current_sha[:12]}... . "
            "Token ids would be meaningless; use the tokenizer that produced this checkpoint."
        )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: Optional[ModelConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device | str = "cpu",
    expected_mode: Optional[str] = None,
    tokenizer_fp: Optional[Dict[str, Any]] = None,
    load_optimizer: bool = True,
    load_data: bool = True,
    load_rng: bool = True,
    strict_arch: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint into ``model`` (and optionally optimizer/scaler).

    Raises :class:`CheckpointError` on any incompatibility instead of silently
    continuing with garbage weights.
    """
    p = Path(path)
    info = peek_checkpoint(p, map_location="cpu")
    _check_identity(info, expected_mode, tokenizer_fp)

    cfg = config
    if cfg is None and info.config:
        cfg = ModelConfig.from_dict(info.config)
    if cfg is not None and strict_arch:
        problems = config_mismatches(info.config, cfg)
        if problems:
            raise CheckpointError(
                "Checkpoint architecture does not match the requested config:\n  "
                + "\n  ".join(problems)
            )

    try:
        # Always read on CPU: model/optimizer state dicts are then copied to the
        # target device by ``load_state_dict``, which avoids a second GPU copy.
        obj = _torch_load(p, map_location="cpu")
    except Exception as exc:
        raise CheckpointError(f"Could not load checkpoint {p}: {exc}") from exc
    state = obj.get("model")
    if not isinstance(state, dict):
        raise CheckpointError(f"Checkpoint {p} has no model state dict")
    target = unwrap_model(model)
    try:
        target.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CheckpointError(f"Model weights are incompatible with {p}: {exc}") from exc

    result = LoadedCheckpoint(
        path=str(p),
        info=info,
        data_state=obj.get("data"),
        rng_by_rank=obj.get("rng_by_rank"),
        train_args=dict(obj.get("train_args") or {}),
        history=list(obj.get("history") or []),
    )

    if load_optimizer and optimizer is not None and obj.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(obj["optimizer"])
            result.optimizer_loaded = True
        except Exception as exc:
            raise CheckpointError(
                f"Optimizer state is incompatible with {p}: {exc}. "
                "Resume with a freshly created optimizer only if you accept a "
                "non-exact restart."
            ) from exc
    if (
        load_optimizer
        and scaler is not None
        and obj.get("scaler")
        and getattr(scaler, "is_enabled", lambda: False)()
    ):
        try:
            scaler.load_state_dict(obj["scaler"])
            result.scaler_loaded = True
        except Exception as exc:
            raise CheckpointError(f"AMP scaler state is incompatible with {p}: {exc}") from exc

    if load_rng and obj.get("rng"):
        restore_rng_state(obj["rng"])
        result.rng_loaded = True
    if load_data and not load_optimizer:
        # A model-only load must not restore the data stream.
        result.data_state = None
    return result


def load_model_weights(
    path: str | Path,
    model: nn.Module,
    *,
    config: Optional[ModelConfig] = None,
    device: torch.device | str = "cpu",
    tokenizer_fp: Optional[Dict[str, Any]] = None,
    expected_mode: Optional[str] = None,
) -> CheckpointInfo:
    """Load weights only (no optimizer/RNG/data) for evaluation or SFT init."""
    loaded = load_checkpoint(
        path,
        model,
        config=config,
        device=device,
        expected_mode=expected_mode,
        tokenizer_fp=tokenizer_fp,
        load_optimizer=False,
        load_data=False,
        load_rng=False,
    )
    return loaded.info


def find_latest(directory: str | Path, pattern: str = "*_last.pth") -> Optional[Path]:
    d = Path(directory)
    if not d.exists():
        return None
    candidates = [p for p in d.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_checkpoint(src: str | Path, dst: str | Path) -> Path:
    """Copy a checkpoint atomically (used for ``last`` -> ``best`` promotions)."""
    s = Path(src)
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_name(f"{d.name}.tmp.{os.getpid()}")
    shutil.copyfile(s, tmp)
    os.replace(tmp, d)
    return d


def checkpoint_summary(path: str | Path) -> str:
    info = peek_checkpoint(path)
    cfg = info.config
    return json.dumps(
        {
            "path": info.path,
            "edition": info.edition,
            "mode": info.mode,
            "step": info.step,
            "tokens_seen": info.tokens_seen,
            "best_val": info.best_val,
            "vocab_size": cfg.get("vocab_size"),
            "n_layer": cfg.get("n_layer"),
            "n_embd": cfg.get("n_embd"),
            "tokenizer_sha256": (info.tokenizer_sha256 or "")[:16],
            "has_optimizer": info.has_optimizer,
            "has_data_state": info.has_data_state,
        },
        indent=2,
    )
