import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspired by ideas used in DeepSeek-family models, but intentionally kept small
# enough for a 2x Tesla T4 training setup.
BLOCK_SIZE = 768
N_EMBD = 512
N_HEAD = 8
N_LAYER = 12
DROPOUT = 0.05
NORM_EPS = 1e-5
ROPE_BASE = 10000.0

# DeepSeek-inspired components.
KV_LATENT = 128       # compressed KV cache representation
N_EXPERTS = 2         # small MoE
MOE_TOP_K = 1         # sparse top-1 routing
MOE_HIDDEN = 512
MOE_AUX_COEF = 0.01

ARCH_VERSION = 8

block_size = BLOCK_SIZE
n_embd = N_EMBD
n_head = N_HEAD
n_layer = N_LAYER


def _round_up(x, multiple=64):
    return multiple * ((x + multiple - 1) // multiple)


def model_config():
    """Metadata used to reject incompatible checkpoints early."""
    return {
        "arch_version": ARCH_VERSION,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "kv_latent": KV_LATENT,
        "n_experts": N_EXPERTS,
        "moe_top_k": MOE_TOP_K,
        "moe_hidden": _round_up(MOE_HIDDEN),
    }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


def rotate_half(x):
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE table for positions 0..max_position-1.

    The KV cache is stored *before* RoPE is applied. K is rotated again on every
    forward using positions relative to the current cache window. This keeps
    training (full sequence, positions 0..t-1) and inference (incremental, with
    a sliding window) consistent and avoids rotating keys twice.
    """

    def __init__(self, dim: int, max_position: int = BLOCK_SIZE, base: float = ROPE_BASE):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head_dim phải là số chẵn.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def apply_rotary(self, x, offset: int):
        n = x.size(2)
        end = offset + n
        if offset < 0 or end > self.cos.size(2):
            raise ValueError(
                f"RoPE vị trí {offset}..{end - 1} vượt quá max_position={self.cos.size(2)}."
            )
        cos = self.cos[:, :, offset:end].to(dtype=x.dtype)
        sin = self.sin[:, :, offset:end].to(dtype=x.dtype)
        return x * cos + rotate_half(x) * sin

    def forward(self, q, k, start: int):
        # Kept for backwards compatibility with external callers.
        return self.apply_rotary(q, start), self.apply_rotary(k, start)


class MultiHeadLatentAttentionLite(nn.Module):
    """MLA-inspired attention with compressed KV state.

    Keys/values are first compressed to a small latent dimension and the
    latent tensors are cached during generation. They are reconstructed only
    when attention is executed. This keeps the cache much smaller than a
    conventional full K/V cache while remaining simple enough for AI9.

    Cache layout: (k_latent, v_latent), each [batch, seq, KV_LATENT]. The cache
    never exceeds BLOCK_SIZE; when it is full the oldest positions are dropped
    so the model always sees the most recent window.
    """

    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.head_dim = N_EMBD // N_HEAD
        self.q_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.kv_down = nn.Linear(N_EMBD, KV_LATENT, bias=False)
        self.v_down = nn.Linear(N_EMBD, KV_LATENT, bias=False)
        self.k_up = nn.Linear(KV_LATENT, N_EMBD, bias=False)
        self.v_up = nn.Linear(KV_LATENT, N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_dropout = DROPOUT
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.rope = RotaryEmbedding(self.head_dim)

    def _expand_kv(self, k_latent, v_latent):
        b, t, _ = k_latent.shape
        k = self.k_up(k_latent).view(b, t, N_HEAD, self.head_dim).transpose(1, 2)
        v = self.v_up(v_latent).view(b, t, N_HEAD, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, x, past_kv=None, use_cache=False):
        b, t, c = x.shape
        if not 1 <= t <= BLOCK_SIZE:
            raise ValueError(f"Độ dài attention phải trong 1..{BLOCK_SIZE}, nhận {t}.")

        q = self.q_proj(x).view(b, t, N_HEAD, self.head_dim).transpose(1, 2)

        # Cache latent states in float32. They are position-independent, so RoPE
        # is applied to the expanded K at every forward. float32 also avoids
        # dtype mismatches if autocast is enabled on one call and disabled on
        # the next (common when generate() is called outside a training loop).
        k_latent = self.kv_down(x).float()
        v_latent = self.v_down(x).float()

        if past_kv is not None:
            if not isinstance(past_kv, (tuple, list)) or len(past_kv) != 2:
                raise ValueError("past_kv phải là tuple (k_latent, v_latent).")
            pk, pv = past_kv
            if pk.size(0) != b or pv.size(0) != b:
                raise ValueError(
                    f"Batch của KV cache ({pk.size(0)}) khác batch input ({b})."
                )
            if pk.size(1) != pv.size(1):
                raise ValueError("Độ dài K/V cache không khớp nhau.")
            if pk.size(2) != KV_LATENT or pv.size(2) != KV_LATENT:
                raise ValueError("Chiều latent của KV cache không khớp cấu hình.")

            past_len = pk.size(1)
            # Sliding window: keep only the most recent BLOCK_SIZE positions.
            # The whole window is shifted together, so relative RoPE distances
            # are unchanged (RoPE is invariant to a global position shift).
            drop = max(0, past_len + t - BLOCK_SIZE)
            if drop:
                pk = pk[:, drop:]
                pv = pv[:, drop:]
            all_k_latent = torch.cat((pk, k_latent), dim=1)
            all_v_latent = torch.cat((pv, v_latent), dim=1)
            past_len = all_k_latent.size(1) - t
        else:
            all_k_latent = k_latent
            all_v_latent = v_latent
            past_len = 0

        k, v = self._expand_kv(all_k_latent, all_v_latent)

        # Q uses positions past_len..past_len+t-1. K uses 0..total_len-1.
        q = self.rope.apply_rotary(q, past_len)
        k = self.rope.apply_rotary(k, 0)

        dropout_p = self.attn_dropout if self.training else 0.0
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        else:
            q_pos = torch.arange(past_len, past_len + t, device=x.device)[:, None]
            k_pos = torch.arange(k.size(2), device=x.device)[None, :]
            mask = k_pos <= q_pos
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False
            )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.proj(y))
        cache = (all_k_latent, all_v_latent) if use_cache else None
        return y, cache


class SwiGLU(nn.Module):
    def __init__(self, hidden=MOE_HIDDEN):
        super().__init__()
        hidden = _round_up(hidden)
        self.gate = nn.Linear(N_EMBD, hidden, bias=False)
        self.up = nn.Linear(N_EMBD, hidden, bias=False)
        self.down = nn.Linear(hidden, N_EMBD, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseMoE(nn.Module):
    """Small DeepSeek-style sparse MoE: routed experts + one shared expert."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(N_EMBD, N_EXPERTS, bias=False)
        self.experts = nn.ModuleList([SwiGLU(MOE_HIDDEN) for _ in range(N_EXPERTS)])
        self.shared = SwiGLU(MOE_HIDDEN)

    def forward(self, x):
        b, t, c = x.shape
        flat = x.reshape(-1, c)

        router_logits = self.router(flat)
        probs = torch.softmax(router_logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=MOE_TOP_K, dim=-1)
        if MOE_TOP_K > 1:
            # Chỉ chuẩn hóa lại khi top_k > 1. Với top-1, việc chia cho chính nó
            # sẽ làm weight luôn bằng 1.0 và triệt tiêu gradient của router từ loss chính.
            top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = self.shared(flat)
        routed = torch.zeros_like(flat)

        # General top-k dispatch. Token positions come from top_idx, not from
        # the flattened (token, slot) index, which is what makes top-k > 1 safe.
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.nonzero(
                top_idx == expert_id, as_tuple=True
            )
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_idx))
            weight = top_prob[token_idx, slot_idx].unsqueeze(-1).to(expert_out.dtype)
            routed.index_add_(0, token_idx, (expert_out * weight).float())

        out = out + routed

        # Switch-style auxiliary load-balancing loss.
        assign = F.one_hot(top_idx, num_classes=N_EXPERTS).float().sum(dim=1)
        fraction = assign.mean(dim=0)
        mean_prob = probs.mean(dim=0)
        aux = N_EXPERTS * torch.sum(fraction * mean_prob)
        return out.view(b, t, c), aux


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(N_EMBD)
        self.attn = MultiHeadLatentAttentionLite()
        self.norm2 = RMSNorm(N_EMBD)
        self.moe = SparseMoE()

    def forward(self, x, past_kv=None, use_cache=False):
        a, cache = self.attn(self.norm1(x), past_kv, use_cache)
        x = x + a
        ffn, aux = self.moe(self.norm2(x))
        x = x + ffn
        return x, cache, aux


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size phải > 0.")
        self.vocab_size = vocab_size
        self.block_size = BLOCK_SIZE
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, MultiHeadLatentAttentionLite):
            nn.init.normal_(module.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
            nn.init.normal_(module.v_up.weight, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
        if isinstance(module, SwiGLU):
            nn.init.normal_(module.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
        if isinstance(module, SparseMoE):
            # Small random values break the initial top-1 tie. Zero init sends
            # every token to expert 0 and delays expert 1 from receiving grads.
            nn.init.normal_(module.router.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        if idx.dim() != 2:
            raise ValueError(f"idx phải có shape [batch, seq], nhận {tuple(idx.shape)}.")
        b, t = idx.shape
        if not 1 <= t <= BLOCK_SIZE:
            raise ValueError(f"Độ dài input phải trong 1..{BLOCK_SIZE}.")

        if past_key_values is not None:
            if not isinstance(past_key_values, (list, tuple)):
                raise ValueError("past_key_values phải là list/tuple theo từng layer.")
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    f"past_key_values có {len(past_key_values)} layer, "
                    f"cần {len(self.blocks)}."
                )

        x = self.token_embedding_table(idx)
        new_cache = []
        moe_aux = []

        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, cache, aux = block(x, past, use_cache)
            if use_cache:
                new_cache.append(cache)
            moe_aux.append(aux)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} khác idx {tuple(idx.shape)}."
                )
            main_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            aux_loss = torch.stack(moe_aux).mean().to(dtype=main_loss.dtype)
            loss = main_loss + MOE_AUX_COEF * aux_loss

        return logits, loss, new_cache if use_cache else None

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.08,
        eos_id=3,
        min_new_tokens=4,
        pad_id=0,
        use_cache=True,
    ):
        """Autoregressive sampling.

        use_cache=True uses the compressed sliding KV cache. use_cache=False
        recomputes the full context window every step; it is slower but useful
        for verification and as a fallback.
        """
        if idx.dim() != 2:
            raise ValueError("idx phải có shape [batch, seq].")
        if idx.size(1) == 0:
            raise ValueError("Prompt rỗng.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens phải >= 0.")

        was_training = self.training
        self.eval()
        try:
            result = idx.clone()
            past = None

            for step in range(max_new_tokens):
                if use_cache:
                    x = result[:, -BLOCK_SIZE:] if past is None else result[:, -1:]
                    logits, _, new_past = self(
                        x, past_key_values=past, use_cache=True
                    )
                    past = new_past
                else:
                    x = result[:, -BLOCK_SIZE:]
                    logits, _, _ = self(x, use_cache=False)

                logits = logits[:, -1, :].float()

                if repetition_penalty and repetition_penalty != 1.0:
                    for row in range(logits.size(0)):
                        seen = result[row].unique()
                        vals = logits[row, seen]
                        logits[row, seen] = torch.where(
                            vals < 0,
                            vals * repetition_penalty,
                            vals / repetition_penalty,
                        )

                if pad_id is not None and 0 <= pad_id < logits.size(-1):
                    logits[:, pad_id] = -float("inf")
                if eos_id is not None and step < min_new_tokens:
                    logits[:, eos_id] = -float("inf")

                if temperature <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k and 0 < top_k < logits.size(-1):
                        v, _ = torch.topk(logits, top_k)
                        logits[logits < v[:, [-1]]] = -float("inf")
                    if 0 < top_p < 1:
                        vals, ids = torch.sort(logits, descending=True, dim=-1)
                        probs = torch.softmax(vals, dim=-1)
                        cumulative = torch.cumsum(probs, dim=-1)
                        remove = cumulative > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        vals[remove] = -float("inf")
                        logits = torch.full_like(logits, -float("inf")).scatter(1, ids, vals)
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)

                result = torch.cat((result, next_token), dim=1)

                if eos_id is not None and step + 1 >= min_new_tokens:
                    if bool((next_token == eos_id).all()):
                        break

            return result
        finally:
            self.train(was_training)


vocab_size = 16384
