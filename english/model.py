"""Configurable decoder-only GPT model for the English edition.

Components: RMSNorm, RoPE, SwiGLU, tied embeddings, PyTorch SDPA attention with
an incremental KV cache.  The default configuration is exactly 124,278,528
parameters.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    head_dim: int,
    max_seq: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    inv_freq = 1.0 / (
        float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    """Multi-head (or grouped-query) causal self-attention with KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.kv_head
        self.head_dim = cfg.head_dim
        self.max_seq = cfg.block_size
        self.dropout = float(cfg.dropout)

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        kv_dim = self.n_kv_head * self.head_dim
        self.k_proj = nn.Linear(cfg.n_embd, kv_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, kv_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        cos, sin = build_rope_cache(self.head_dim, self.max_seq, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _apply_rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        n = x.size(2)
        end = offset + n
        if offset < 0 or end > self.max_seq:
            raise ValueError(
                f"RoPE positions {offset}..{end - 1} exceed max_seq={self.max_seq}"
            )
        cos = self.rope_cos[offset:end].to(dtype=x.dtype).view(1, 1, n, -1)
        sin = self.rope_sin[offset:end].to(dtype=x.dtype).view(1, 1, n, -1)
        return x * cos + rotate_half(x) * sin

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            if not isinstance(past_kv, (tuple, list)) or len(past_kv) != 2:
                raise ValueError("past_kv must be a (k, v) tuple")
            pk, pv = past_kv
            if pk.size(0) != b or pv.size(0) != b:
                raise ValueError("KV cache batch size does not match input")
            if pk.size(1) != self.n_kv_head or pv.size(1) != self.n_kv_head:
                raise ValueError("KV cache head count does not match model config")
            if pk.size(3) != self.head_dim or pv.size(3) != self.head_dim:
                raise ValueError("KV cache head dimension does not match model config")
            past_len = pk.size(2)
            if past_len + t > self.max_seq:
                raise ValueError(
                    f"Sequence length {past_len + t} exceeds block_size={self.max_seq}"
                )
            k_all = torch.cat((pk, k), dim=2)
            v_all = torch.cat((pv, v), dim=2)
        else:
            if t > self.max_seq:
                raise ValueError(f"Sequence length {t} exceeds block_size={self.max_seq}")
            k_all, v_all = k, v

        q = self._apply_rope(q, past_len)
        # The cache stores *unrotated* keys; RoPE is applied to the whole
        # concatenated sequence on every forward.  Rotating before caching would
        # rotate cached keys twice on the next call.
        k_rot = self._apply_rope(k_all, 0)

        # Optional GQA: repeat KV heads to match query heads.
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k_attn = k_rot.repeat_interleave(rep, dim=1)
            v_attn = v_all.repeat_interleave(rep, dim=1)
        else:
            k_attn, v_attn = k_rot, v_all

        dropout_p = self.dropout if self.training else 0.0
        total = k_attn.size(2)

        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}, got {tuple(attention_mask.shape)}"
                )
            key_mask = attention_mask.to(torch.bool)
            if past_len:
                key_mask = torch.cat(
                    (
                        torch.ones(b, past_len, dtype=torch.bool, device=x.device),
                        key_mask,
                    ),
                    dim=1,
                )
            key_mask = key_mask[:, None, None, :]
            if t > 1:
                q_pos = torch.arange(past_len, past_len + t, device=x.device)[:, None]
                k_pos = torch.arange(total, device=x.device)[None, :]
                causal = k_pos <= q_pos
                mask = causal[None, None, :, :] & key_mask
            else:
                mask = key_mask
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=mask, dropout_p=dropout_p
            )
        elif past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, dropout_p=dropout_p, is_causal=t > 1
            )
        elif t == 1:
            # A single query sees the entire cache; no causal mask is needed.
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, dropout_p=dropout_p, is_causal=False
            )
        else:
            q_pos = torch.arange(past_len, past_len + t, device=x.device)[:, None]
            k_pos = torch.arange(total, device=x.device)[None, :]
            mask = k_pos <= q_pos
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=mask, dropout_p=dropout_p
            )

        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        y = self.o_proj(y)
        if self.training and self.dropout:
            y = F.dropout(y, p=self.dropout)
        cache = (k_all, v_all) if use_cache else None
        return y, cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.n_inner, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, cfg.n_inner, bias=cfg.bias)
        self.down = nn.Linear(cfg.n_inner, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        a, cache = self.attn(
            self.norm1(x), past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask
        )
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, cache


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.vocab_size = cfg.vocab_size

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # GPT-2 style residual scaling for the output projections.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape [batch, seq], got {tuple(idx.shape)}")
        b, t = idx.shape
        if t < 1:
            raise ValueError("input sequence must not be empty")
        if t > self.block_size:
            raise ValueError(f"input length {t} exceeds block_size={self.block_size}")

        if past_key_values is not None:
            if not isinstance(past_key_values, (list, tuple)):
                raise ValueError("past_key_values must be a list/tuple per layer")
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    f"past_key_values has {len(past_key_values)} layers, expected {len(self.blocks)}"
                )

        x = self.token_embedding(idx)
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, cache = block(
                x, past_kv=past, use_cache=use_cache, attention_mask=attention_mask
            )
            if use_cache:
                if cache is None:
                    raise RuntimeError("use_cache=True but attention returned no cache")
                new_cache.append(cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} does not match idx {tuple(idx.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, (new_cache if use_cache else None)


def build_model(cfg: ModelConfig, device: torch.device | str = "cpu") -> GPT:
    model = GPT(cfg)
    return model.to(device)
