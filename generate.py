"""Token sampling and KV-cached streaming generation.

The implementation keeps the full prompt context: generation refuses to exceed
``block_size`` instead of silently sliding the cache, because sliding unrotated
RoPE keys would corrupt positions.  Chat truncates the prompt to fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, SYSTEM_ID, USER_ID
from .tokenizer import decode as decode_ids

STOP_IDS = (PAD_ID, EOS_ID, BOS_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    repetition_window: int = 256
    seed: Optional[int] = None
    stop_ids: tuple[int, ...] = STOP_IDS
    min_new_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.repetition_window < 0:
            raise ValueError("repetition_window must be >= 0")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 1e-5


def make_generator(cfg: GenerationConfig, device: torch.device) -> Optional[torch.Generator]:
    if cfg.greedy or cfg.seed is None:
        return None
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(int(cfg.seed))
    return gen


def apply_repetition_penalty(
    logits: torch.Tensor,
    recent_tokens: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    """CTRL-style repetition penalty.  Positive logits are divided, negative multiplied."""
    if penalty <= 1.0 or not recent_tokens:
        return logits
    ids = sorted({int(t) for t in recent_tokens if 0 <= int(t) < logits.numel()})
    if not ids:
        return logits
    idx = torch.tensor(ids, dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, idx)
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    out = logits.clone()
    out.index_copy_(0, idx, adjusted)
    return out


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    kth = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability >= top_p."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative - probs >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out


def sample_token(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    recent_tokens: Sequence[int] = (),
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one token id from a 1-D logits vector."""
    if logits.dim() != 1:
        raise ValueError("logits must be a 1-D tensor")
    work = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    work = apply_repetition_penalty(work, recent_tokens[-cfg.repetition_window:], cfg.repetition_penalty)
    if cfg.greedy:
        return int(torch.argmax(work).item())

    work = work / float(cfg.temperature)
    work = filter_top_k(work, int(cfg.top_k))
    work = filter_top_p(work, float(cfg.top_p))
    probs = torch.softmax(work, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        return int(torch.argmax(work).item())
    tok = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(tok.item())


def _normalize_prompt(prompt_ids: Sequence[int], block_size: int, max_new: int) -> List[int]:
    ids = [int(t) for t in prompt_ids]
    if not ids:
        ids = [BOS_ID]
    if len(ids) + max_new > block_size:
        raise ValueError(
            f"Prompt has {len(ids)} tokens but block_size={block_size} leaves room for "
            f"only {block_size - len(ids)} new tokens (requested {max_new}). "
            "Shorten the prompt/history or lower --max-new-tokens."
        )
    return ids


@torch.inference_mode()
def generate_stream(
    model: nn.Module,
    prompt_ids: Sequence[int],
    cfg: GenerationConfig,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Iterator[int]:
    """Yield newly generated token ids until EOS/stop id or the token budget.

    The stop token itself is not yielded.  ``model`` must expose ``block_size``
    and accept ``(idx, past_key_values=..., use_cache=True)``.
    """
    device = torch.device(device)
    model.eval()
    block_size = int(getattr(model, "block_size"))
    prompt = _normalize_prompt(prompt_ids, block_size, cfg.max_new_tokens)
    generator = generator if generator is not None else make_generator(cfg, device)

    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    logits, _, past = model(ids, past_key_values=None, use_cache=True)
    generated: List[int] = []
    stop = set(int(t) for t in cfg.stop_ids)

    for step in range(cfg.max_new_tokens):
        recent = prompt + generated
        logits_step = logits[0, -1, :]
        tok = sample_token(logits_step, cfg, recent, generator)
        if tok in stop and step < cfg.min_new_tokens:
            # Forbid an early stop token and resample from the remaining mass.
            masked = logits_step.clone()
            stop_idx = torch.tensor(sorted(stop), dtype=torch.long, device=masked.device)
            masked.index_fill_(0, stop_idx, float("-inf"))
            tok = sample_token(masked, cfg, recent, generator)
        if tok in stop:
            return
        generated.append(tok)
        yield tok
        if step + 1 >= cfg.max_new_tokens:
            return
        nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
        logits, _, past = model(nxt, past_key_values=past, use_cache=True)


def generate_ids(
    model: nn.Module,
    prompt_ids: Sequence[int],
    cfg: GenerationConfig,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    return list(generate_stream(model, prompt_ids, cfg, device=device, generator=generator))


def generate_text(
    model: nn.Module,
    prompt_ids: Sequence[int],
    cfg: GenerationConfig,
    sp,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> str:
    ids = generate_ids(model, prompt_ids, cfg, device=device, generator=generator)
    return decode_ids(ids, sp=sp, skip_special=True).strip()


class TextStreamer:
    """Turn a token stream into incremental text deltas.

    SentencePiece can emit replacement characters while a multi-byte UTF-8
    sequence is incomplete; deltas are therefore only emitted when the decoded
    prefix is stable.
    """

    def __init__(self, sp):
        self.sp = sp
        self.ids: List[int] = []
        self.emitted = ""

    def push(self, token: int) -> str:
        self.ids.append(int(token))
        text = decode_ids(self.ids, sp=self.sp, skip_special=True)
        if not text.startswith(self.emitted):
            return ""
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def flush(self) -> str:
        text = decode_ids(self.ids, sp=self.sp, skip_special=True)
        if text.startswith(self.emitted):
            delta = text[len(self.emitted):]
        else:
            delta = text[len(self.emitted):]
        self.emitted = text
        return delta
