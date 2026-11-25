"""Shared training utilities for the English edition."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# Devices and AMP
# ---------------------------------------------------------------------------


def is_bf16_supported(device: Optional[torch.device] = None) -> bool:
    """True only when the GPU has native BF16 support.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True``
    on recent PyTorch versions, which incorrectly reports Turing/T4 as capable.
    Emulated BF16 is slow and disables GradScaler, so emulation must be
    excluded.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        try:
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
        except Exception:
            return False
    except Exception:
        return False


def resolve_amp(mode: str, device: torch.device) -> tuple[bool, torch.dtype, bool]:
    """Return ``(use_amp, dtype, scaler_enabled)``.

    FP16 is the correct choice on T4.  BF16 is used only when the hardware
    supports it natively.  ``mode='auto'`` chooses FP16 on CUDA Turing and BF16
    on Ampere or newer.
    """
    mode = (mode or "auto").lower()
    if mode not in {"auto", "fp16", "bf16", "off"}:
        raise ValueError("amp must be one of: auto, fp16, bf16, off")
    if mode == "off" or device.type != "cuda":
        return False, torch.float32, False
    bf16 = is_bf16_supported(device)
    if mode == "bf16":
        if not bf16:
            raise RuntimeError(
                "BF16 is not natively supported on this GPU; use --amp fp16 or auto."
            )
        return True, torch.bfloat16, False
    if mode == "fp16":
        return True, torch.float16, True
    if bf16:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def make_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler across PyTorch API versions."""
    if not enabled:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except (TypeError, AttributeError):  # pragma: no cover
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=True)


@contextmanager
def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        yield
        return
    try:
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            yield
    except (TypeError, AttributeError):  # pragma: no cover - older PyTorch
        with torch.cuda.amp.autocast(dtype=dtype):
            yield


def get_dist_env() -> tuple[bool, int, int, int]:
    """Read torchrun environment variables."""
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return world > 1, rank, local_rank, world


def init_nccl(
    rank: int,
    local_rank: int,
    world: int,
    timeout_minutes: int = 30,
) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requires CUDA/NCCL. Enable a GPU runtime."
        )
    count = torch.cuda.device_count()
    if not 0 <= local_rank < count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid; only {count} CUDA device(s) visible."
        )
    torch.cuda.set_device(local_rank)
    kwargs = dict(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=timeout_minutes),
    )
    try:
        dist.init_process_group(device_id=local_rank, **kwargs)
    except TypeError:
        # Older PyTorch without device_id support.
        dist.init_process_group(**kwargs)
    current = torch.cuda.current_device()
    if current != local_rank:
        raise RuntimeError(
            f"Rank {rank} is on cuda:{current}, expected cuda:{local_rank}."
        )
    return torch.device(f"cuda:{local_rank}")


def init_gloo(rank: int, world: int, timeout_minutes: int = 120) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=timeout_minutes),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def gather_object(obj: Any) -> List[Any]:
    """Gather one JSON-able object from every rank.  Every rank must call this."""
    if not is_distributed():
        return [obj]
    world = dist.get_world_size()
    out: List[Any] = [None] * world  # type: ignore[list-item]
    dist.all_gather_object(out, obj)
    return out


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Seeding / RNG
# ---------------------------------------------------------------------------


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank) * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        # Keep CUDA RNG states on the CPU so the checkpoint loads on a
        # CPU-only machine (``torch.load`` cannot map CUDA tensors otherwise).
        state["cuda"] = [s.detach().cpu() for s in torch.cuda.get_rng_state_all()]
    else:
        state["cuda"] = None
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            restored = [
                s.to("cuda", non_blocking=False)
                for s in cuda_state[:device_count]
            ]
            if restored:
                torch.cuda.set_rng_state_all(restored)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Could not fully restore RNG state: {exc}")


# ---------------------------------------------------------------------------
# Optimizer and schedules
# ---------------------------------------------------------------------------


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in name.lower() and "embedding" not in name.lower():
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: Dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def cosine_lr(
    progress: float,
    peak_lr: float,
    min_lr: float,
) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def lr_at_tokens(
    tokens_seen: int,
    plan_tokens: int,
    warmup_tokens: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        return peak_lr * float(tokens_seen + 1) / float(warmup_tokens)
    if plan_tokens <= warmup_tokens:
        return min_lr
    progress = (tokens_seen - warmup_tokens) / float(plan_tokens - warmup_tokens)
    return cosine_lr(progress, peak_lr, min_lr)


def lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    return cosine_lr(progress, peak_lr, min_lr)


# ---------------------------------------------------------------------------
# Logging / signals / formatting
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


class JsonlLogger:
    """Append-only JSONL logger; rank 0 only by convention."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("time", utc_now())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class StopFlag:
    requested: bool = False

    def install(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.requested = True
            print(
                f"\n[INFO] Received signal {signum}; will save and stop after the current step.",
                flush=True,
            )

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


@contextmanager
def nullcontext():
    yield


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def gpu_memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / (1024 ** 3),
        torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    )
