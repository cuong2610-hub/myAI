"""Benchmark cached single-token attention for MLA and reference GQA."""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from model import BLOCK_SIZE, KV_LATENT, N_EMBD, N_HEAD, MultiHeadLatentAttentionLite


def benchmark(fn, warmup, iterations, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, BLOCK_SIZE])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    mla = MultiHeadLatentAttentionLite().to(device).eval()
    head_dim = N_EMBD // N_HEAD
    # Reference GQA with 2 KV heads; this is a timing baseline, not a trained model.
    gqa_k = torch.nn.Linear(N_EMBD, 2 * head_dim, bias=False).to(device).eval()
    gqa_v = torch.nn.Linear(N_EMBD, 2 * head_dim, bias=False).to(device).eval()
    gqa_q = torch.nn.Linear(N_EMBD, N_EMBD, bias=False).to(device).eval()

    print(f"device={device} | dtype=float32 | median ms, batch=1, query=1")
    for length in args.lengths:
        if not 1 <= length <= BLOCK_SIZE:
            raise ValueError(f"length must be in 1..{BLOCK_SIZE}: {length}")
        x = torch.randn(1, 1, N_EMBD, device=device)
        context = torch.randn(1, length, N_EMBD, device=device)
        with torch.inference_mode():
            kl = mla.kv_down(context).float()
            vl = mla.v_down(context).float()
            q = mla.q_proj(x).view(1, 1, N_HEAD, head_dim).transpose(1, 2)
            q = mla.rope.apply_rotary(q, length)

            def mla_decode():
                k, v = mla._expand_kv(kl, vl)
                k = mla.rope.apply_rotary(k, 0)
                F.scaled_dot_product_attention(q, k, v, is_causal=False)

            # For one query, causal masking has no effect because the entire cache is past.
            gk = gqa_k(context).view(1, length, 2, head_dim).transpose(1, 2)
            gv = gqa_v(context).view(1, length, 2, head_dim).transpose(1, 2)
            gq = gqa_q(x).view(1, 1, N_HEAD, head_dim).transpose(1, 2)
            gq = mla.rope.apply_rotary(gq, length)

            def gqa_decode():
                F.scaled_dot_product_attention(gq, gk, gv, is_causal=False, enable_gqa=True)

            mla_ms = benchmark(mla_decode, args.warmup, args.iterations, device)
            gqa_ms = benchmark(gqa_decode, args.warmup, args.iterations, device)
        cache_mla = 2 * length * KV_LATENT * 4
        cache_gqa = 2 * length * 2 * head_dim * 4
        print(
            f"len={length:4d} | MLA expand+attn={mla_ms:8.3f} ms "
            f"cache={cache_mla / 1024:.1f} KiB | "
            f"GQA attn={gqa_ms:8.3f} ms cache={cache_gqa / 1024:.1f} KiB"
        )


if __name__ == "__main__":
    main()
