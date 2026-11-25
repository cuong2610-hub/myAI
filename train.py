"""Resumable English pretraining on streaming FineWeb-Edu.

Single-GPU example::

    python train.py --preset base --amp auto

Dual-T4 (Kaggle) example::

    torchrun --nproc_per_node=2 -m train \
        --preset base --plan-tokens 5000000000 --amp fp16

The script never downloads the full corpus: Hugging Face ``streaming=True`` is
used and the shuffle buffer lives in this process (and in the checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from config import (
    BASE_CONFIG,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_RUNS_DIR,
    DEFAULT_SPLIT,
    DEFAULT_TOKENIZER_DIR,
    TINY_CONFIG,
    ModelConfig,
    ensure_dir,
    get_preset,
    with_overrides,
)
from data import (
    DocumentStream,
    PackedTokenStream,
    RetryConfig,
    SyntheticDocumentStream,
    batch_from_bin,
    build_val_tokens,
    load_val_tokens,
)
from model import GPT
from tokenizer import (
    BOS_ID,
    EOS_ID,
    load_tokenizer,
    tokenizer_fingerprint,
    train_tokenizer,
)
from utils import (
    JsonlLogger,
    StopFlag,
    autocast_context,
    barrier,
    build_optimizer,
    capture_rng_state,
    cleanup_distributed,
    configure_stdio,
    distributed_mean,
    gather_object,
    get_dist_env,
    gpu_memory,
    human_bytes,
    init_gloo,
    init_nccl,
    is_distributed,
    lr_at_tokens,
    make_grad_scaler,
    resolve_amp,
    restore_rng_state,
    seed_everything,
    unwrap_model,
)

BASE_PARAMS = 124_278_528
SMOKE_TEXTS = (
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Machine learning models learn patterns from large collections of text.",
    "A transformer uses attention to connect distant words in a sentence.",
    "Python is a popular programming language for data science and research.",
    "The training loop updates model weights with gradient descent.",
    "SentencePiece converts text into a sequence of subword tokens.",
    "Careful evaluation prevents overfitting to the training distribution.",
    "A small model can still produce coherent English paragraphs.",
    "Graphics processing units accelerate matrix multiplication.",
    "The validation loss should decrease as the model learns.",
    "Data pipelines stream documents without storing the whole dataset.",
    "Checkpoints preserve the optimizer, random state and data position.",
    "A good tokenizer balances vocabulary size against sequence length.",
    "Temperature and top-p sampling control the diversity of generated text.",
    "Retrieval augmented generation grounds answers in retrieved documents.",
    "The English edition is self-contained and independent.",
)

SMOKE_WORDS = (
    "the quick brown fox jumps over lazy dog machine learning transformer "
    "attention python code computer science model language data training neural "
    "network gradient token embedding sentence english text example value system "
    "user assistant river bank matrix vector probability sampling temperature "
    "validation checkpoint streaming dataset retrieval document paragraph"
).split()


# ---------------------------------------------------------------------------
# Arguments and plan
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="English GPT pretraining (streaming FineWeb-Edu).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--preset", choices=["tiny", "base", "wide"], default="base")
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--n-embd", type=int, default=None)
    ap.add_argument("--n-inner", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)

    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR / "tokenizer.model"))
    ap.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--data-source", default=None, help="Local JSONL file or dataset id.")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--min-chars", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--shuffle-buffer", type=int, default=1000)
    ap.add_argument("--max-doc-tokens", type=int, default=8192)

    ap.add_argument("--batch-size", type=int, default=8, help="per GPU")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--plan-tokens", type=int, default=5_000_000_000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup-tokens", type=int, default=100_000_000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)

    ap.add_argument("--resume", default="", help="Checkpoint path, or 'auto'.")
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--allow-data-restart", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expect-params", type=int, default=0)
    return ap


def resolve_config(args: argparse.Namespace) -> ModelConfig:
    cfg = get_preset(args.preset)
    cfg = with_overrides(
        cfg,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_inner=args.n_inner,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
    )
    return cfg


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.preset = "tiny"
    args.batch_size = 2
    args.grad_accum = 1
    args.plan_tokens = 100_000
    if args.max_steps <= 0:
        args.max_steps = 12
    args.lr = 3e-3
    args.min_lr = 3e-4
    args.warmup_tokens = 512
    args.eval_every = 2
    args.save_every = 2
    args.log_every = 1
    args.eval_iters = 2
    args.min_chars = 10
    args.max_chars = 20_000
    args.shuffle_buffer = 8
    args.max_doc_tokens = 256
    args.val_tokens = 20_000
    args.val_fraction = 0.1
    args.amp = "auto"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def resolve_source(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    source = args.data_source or args.dataset
    if os.path.exists(str(source)):
        return str(source), None
    return str(source), args.dataset_config


def val_asset_key(args: argparse.Namespace, cfg: ModelConfig, tokenizer_sha: str) -> str:
    source, config = resolve_source(args)
    stat_key: Dict[str, Any] = {}
    if os.path.exists(source):
        st = os.stat(source)
        stat_key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    key = {
        "format": 1,
        "source": os.path.abspath(source) if os.path.exists(source) else source,
        "config": config,
        "split": args.split,
        "val_fraction": args.val_fraction,
        "block_size": cfg.block_size,
        "val_tokens": args.val_tokens,
        "tokenizer_sha": tokenizer_sha,
        "stat": stat_key,
    }
    raw = json.dumps(key, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def smoke_tokenizer_corpus(path: Path, seed: int = 1337, lines: int = 20_000) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(lines):
            n = rng.randint(8, 28)
            words = [rng.choice(SMOKE_WORDS) for _ in range(n)]
            f.write(" ".join(words) + ".\n")
    return path


def ensure_smoke_tokenizer(directory: Path, vocab_size: int, seed: int) -> Path:
    model_path = directory / "tokenizer.model"
    if model_path.exists():
        try:
            sp = load_tokenizer(model_path, expected_vocab=vocab_size)
            if int(sp.vocab_size()) == vocab_size:
                return model_path
        except Exception:
            pass
    corpus = directory / "corpus_smoke.txt"
    if not corpus.exists():
        smoke_tokenizer_corpus(corpus, seed=seed)
    path = train_tokenizer(
        corpus_path=corpus,
        out_dir=directory,
        vocab_size=vocab_size,
        force=True,
        num_threads=2,
    )
    load_tokenizer(path, expected_vocab=vocab_size)
    return path


def smoke_documents(seed: int, count: int = 400) -> List[str]:
    rng = random.Random(seed + 17)
    docs: List[str] = []
    for _ in range(count):
        n = rng.randint(80, 220)
        words = [rng.choice(SMOKE_WORDS) for _ in range(n)]
        docs.append(" ".join(words) + ".")
    return docs


def build_synthetic_val_tokens(sp, cfg: ModelConfig, seed: int) -> np.ndarray:
    if int(sp.vocab_size()) > 65535:
        raise ValueError("synthetic validation tokens use uint16")
    ids: List[int] = []
    for text in smoke_documents(seed, count=120):
        ids.append(BOS_ID)
        ids.extend(int(i) for i in sp.encode(text, out_type=int))
        ids.append(EOS_ID)
    minimum = cfg.block_size * 64 + 2
    while len(ids) < minimum:
        ids.extend(ids[:minimum - len(ids)])
    return np.asarray(ids, dtype=np.uint16)


def prepare_assets(
    args: argparse.Namespace,
    cfg: ModelConfig,
    rank: int,
    world: int,
) -> Tuple[Path, Optional[Path], Optional[Path], Optional[np.ndarray]]:
    """Create tokenizer/validation assets before NCCL starts (rank 0 only work)."""
    ddp = world > 1
    if ddp:
        init_gloo(rank, world, timeout_minutes=180)
    tokenizer_path = Path(args.tokenizer)
    if args.smoke_test:
        # Every rank must agree on the smoke tokenizer path.
        tokenizer_path = Path(args.tokenizer_dir) / "smoke" / "tokenizer.model"
    val_bin: Optional[Path] = None
    val_meta: Optional[Path] = None
    synthetic_val: Optional[np.ndarray] = None
    try:
        if rank == 0:
            if args.smoke_test:
                tokenizer_path = ensure_smoke_tokenizer(
                    Path(args.tokenizer_dir) / "smoke", cfg.vocab_size, args.seed
                )
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"English tokenizer not found: {tokenizer_path}\n"
                    "Train it first:\n"
                    "  python tokenizer.py --train --sample-mb 200 --vocab-size 32768"
                )
            sp = load_tokenizer(tokenizer_path, expected_vocab=cfg.vocab_size)
            if args.smoke_test:
                synthetic_val = build_synthetic_val_tokens(sp, cfg, args.seed)
            else:
                sha = tokenizer_fingerprint(tokenizer_path)["sha256"]
                digest = val_asset_key(args, cfg, sha)
                val_dir = ensure_dir(Path(args.data_dir) / "val")
                val_bin = val_dir / f"val_{digest}.bin"
                val_meta = val_dir / f"val_{digest}.json"
                source, config = resolve_source(args)
                print(f"[data] Preparing validation tokens ({args.val_tokens:,}) ...", flush=True)
                started = time.time()
                meta = build_val_tokens(
                    val_bin,
                    val_meta,
                    sp,
                    block_size=cfg.block_size,
                    max_tokens=args.val_tokens,
                    source=source,
                    config=config,
                    split=args.split,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                    tokenizer_sha=sha,
                )
                print(
                    f"[data] Validation tokens ready: {meta['tokens']:,} tokens "
                    f"from {meta['documents']:,} held-out documents "
                    f"({time.time() - started:.1f}s)",
                    flush=True,
                )
        if ddp:
            barrier()
        if rank != 0 and not args.smoke_test:
            sha = tokenizer_fingerprint(tokenizer_path)["sha256"]
            digest = val_asset_key(args, cfg, sha)
            val_dir = Path(args.data_dir) / "val"
            val_bin = val_dir / f"val_{digest}.bin"
            val_meta = val_dir / f"val_{digest}.json"
    finally:
        if ddp and is_distributed():
            cleanup_distributed()
    return tokenizer_path, val_bin, val_meta, synthetic_val


# ---------------------------------------------------------------------------
# Data / evaluation
# ---------------------------------------------------------------------------


def build_packed_stream(args: argparse.Namespace, sp, cfg: ModelConfig, rank: int, world: int) -> PackedTokenStream:
    if args.smoke_test:
        docs = SyntheticDocumentStream(
            smoke_documents(args.seed),
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
            repeat=True,
            val_fraction=0.0,
            mode="train",
        )
    else:
        source, config = resolve_source(args)
        docs = DocumentStream(
            source=source,
            config=config,
            split=args.split,
            world_size=world,
            rank=rank,
            val_fraction=args.val_fraction,
            mode="train",
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            retry=RetryConfig(),
        )
    return PackedTokenStream(docs, sp, cfg.block_size, max_doc_tokens=args.max_doc_tokens)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: np.ndarray,
    cfg: ModelConfig,
    device: torch.device,
    iters: int,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    was_training = model.training
    model.eval()
    losses: List[float] = []
    for _ in range(max(1, int(iters))):
        x, y = batch_from_bin(data, cfg.block_size, batch_size, rng, device)
        with autocast_context(device, amp_dtype, use_amp):
            _, loss, _ = model(x, y)
        losses.append(float(loss.detach().float().item()))
    if was_training:
        model.train()
    value = sum(losses) / len(losses)
    return distributed_mean(value, device)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_smoke_defaults(args)
    cfg = resolve_config(args)

    ddp, rank, local_rank, world = get_dist_env()
    if world > 1 and not torch.cuda.is_available():
        raise SystemExit("Distributed training requires CUDA/NCCL; no CUDA device found.")

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0
    smoke = bool(args.smoke_test)

    ckpt_dir = ensure_dir(Path(args.checkpoint_dir) / ("smoke" if smoke else ""))
    runs_dir = ensure_dir(Path(args.runs_dir))
    last_path = ckpt_dir / "pretrain_last.pth"
    best_path = ckpt_dir / "pretrain_best.pth"

    # Parameter count check happens before any heavy work.
    formula = cfg.param_count()
    if args.preset == "base" and all(
        v is None
        for v in (args.n_layer, args.n_head, args.n_embd, args.n_inner, args.block_size, args.vocab_size)
    ):
        if formula != BASE_PARAMS:
            raise SystemExit(f"Internal error: base parameter formula {formula} != {BASE_PARAMS}")
    if args.expect_params and formula != args.expect_params:
        raise SystemExit(f"Parameter count {formula:,} != --expect-params {args.expect_params:,}")

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size * max(1, world)
    if args.plan_tokens < tokens_per_step:
        raise SystemExit("--plan-tokens must be at least one optimizer step of tokens")
    plan_steps = int(math.ceil(args.plan_tokens / tokens_per_step))
    total_steps = plan_steps if args.max_steps <= 0 else min(plan_steps, args.max_steps)

    if is_main:
        print("=" * 78)
        print("English edition pretraining")
        print(f"  preset              : {args.preset}")
        print(f"  config              : {cfg.n_layer}L / {cfg.n_embd}d / {cfg.n_head}h / "
              f"ctx {cfg.block_size} / vocab {cfg.vocab_size} / SwiGLU {cfg.n_inner}")
        print(f"  exact parameters    : {formula:,}"
              + ("  (base contract OK)" if formula == BASE_PARAMS else ""))
        print(f"  devices             : {world} x {device.type}"
              + (f" ({torch.cuda.get_device_name(local_rank)})" if device.type == "cuda" else ""))
        print(f"  batch / accumulation: {args.batch_size} x {args.grad_accum}"
              f" (global {tokens_per_step:,} tokens/step)")
        print(f"  token plan          : {args.plan_tokens:,} tokens "
              f"({plan_steps:,} steps" + (f", capped at {total_steps:,}" if total_steps != plan_steps else "") + ")")
        print(f"  learning rate       : {args.lr:g} -> {args.min_lr:g} "
              f"(warmup {args.warmup_tokens:,} tokens)")
        if not smoke:
            rough_hours = args.plan_tokens / 8000.0 / 3600.0
            print(f"  rough compute       : ~{rough_hours:.1f} GPU-hours of 2xT4 time "
                  "(exact speed varies; resumable)")
        print(f"  last checkpoint     : {last_path}")
        print("=" * 78, flush=True)

    if args.dry_run:
        if is_main:
            print("[dry-run] No tokenizer, dataset or checkpoint was touched.")
        return 0

    tokenizer_path, val_bin, val_meta, synthetic_val = prepare_assets(args, cfg, rank, world)
    sp = load_tokenizer(tokenizer_path, expected_vocab=cfg.vocab_size)
    tok_fp = tokenizer_fingerprint(tokenizer_path)

    if val_bin is not None:
        val_data = load_val_tokens(
            val_bin,
            val_meta,  # type: ignore[arg-type]
            tokenizer_sha=tok_fp["sha256"],
            block_size=cfg.block_size,
        )
    elif args.smoke_test:
        val_data = (
            synthetic_val
            if synthetic_val is not None
            else build_synthetic_val_tokens(sp, cfg, args.seed)
        )
    else:
        raise RuntimeError("No validation data was prepared")

    seed_everything(args.seed, rank)
    if world > 1:
        device = init_nccl(rank, local_rank, world)
    elif device.type == "cuda":
        torch.cuda.set_device(local_rank)

    use_amp, amp_dtype, scaler_enabled = resolve_amp(args.amp, device)
    if is_main:
        print(f"[amp] enabled={use_amp} dtype={amp_dtype} grad_scaler={scaler_enabled}", flush=True)

    model = GPT(cfg).to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    if actual_params != formula:
        raise SystemExit(f"Model has {actual_params:,} parameters, formula says {formula:,}")
    if is_main:
        print(f"[model] built {actual_params:,} parameters", flush=True)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(scaler_enabled)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    packed = build_packed_stream(args, sp, cfg, rank, world)
    logger = JsonlLogger(runs_dir / ("smoke.jsonl" if smoke else "pretrain.jsonl"))
    stop = StopFlag()
    stop.install()

    step = 0
    tokens_seen = 0
    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    data_mode = "fresh"
    resume_path = ""

    if not args.from_scratch:
        if args.resume:
            resume_path = str(last_path if args.resume == "auto" else args.resume)
        elif last_path.exists():
            resume_path = str(last_path)
        if resume_path and not Path(resume_path).exists() and args.resume not in ("", "auto"):
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        if resume_path and Path(resume_path).exists():
            loaded = load_checkpoint(
                resume_path,
                model,
                config=cfg,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                expected_mode="pretrain",
                tokenizer_fp=tok_fp,
                load_optimizer=True,
                load_data=True,
                load_rng=(world == 1),
            )
            step = int(loaded.info.step)
            tokens_seen = int(loaded.info.tokens_seen)
            best_val = float(loaded.info.best_val)
            history = list(loaded.history)
            if world > 1 and loaded.rng_by_rank and len(loaded.rng_by_rank) > rank:
                restore_rng_state(loaded.rng_by_rank[rank])
            states = loaded.data_state
            if states:
                if isinstance(states, dict):
                    states = [states]
                if not isinstance(states, list) or len(states) != world:
                    if not args.allow_data_restart:
                        raise CheckpointError(
                            f"Saved data state is for {len(states) if isinstance(states, list) else 1} "
                            f"rank(s), current world size is {world}. "
                            "Resume with the same GPU count or pass --allow-data-restart "
                            "(this re-reads already-seen documents and is not exact)."
                        )
                    print(
                        "[WARN] World size changed: restarting the data stream from the "
                        "beginning while keeping the optimizer/step state.",
                        flush=True,
                    )
                    data_mode = "restart"
                else:
                    packed.load_state_dict(states[rank])
                    data_mode = "exact"
            if is_main:
                print(
                    f"[resume] {resume_path} | step={step:,} tokens={tokens_seen:,} "
                    f"best_val={best_val:.4f} data={data_mode}",
                    flush=True,
                )

    if not is_main:
        logger = None  # keep JSONL single-writer

    if tokens_seen >= args.plan_tokens:
        if is_main:
            print(
                f"[done] Token plan already complete ({tokens_seen:,} >= {args.plan_tokens:,}).",
                flush=True,
            )
        cleanup_distributed()
        return 0

    model.train()
    started = time.time()
    start_step = step
    start_tokens = tokens_seen
    skipped_steps = 0
    micro_losses: List[float] = []
    grad_norm_value = 0.0
    last_log = time.time()
    running_loss: List[float] = []
    stopped_early = False

    def save_last() -> None:
        nonlocal history
        states = gather_object(packed.state_dict())
        rngs = gather_object(capture_rng_state())
        if is_main:
            save_checkpoint(
                last_path,
                model=model,
                config=cfg,
                mode="pretrain",
                step=step,
                tokens_seen=tokens_seen,
                best_val=best_val,
                optimizer=optimizer,
                scaler=scaler,
                tokenizer_fp=tok_fp,
                rng_state=rngs[0],
                data_state=states,
                train_args={
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "plan_tokens": args.plan_tokens,
                    "warmup_tokens": args.warmup_tokens,
                    "lr": args.lr,
                    "min_lr": args.min_lr,
                    "weight_decay": args.weight_decay,
                    "amp": args.amp,
                    "seed": args.seed,
                    "world_size": world,
                    "data_source": resolve_source(args)[0],
                    "val_fraction": args.val_fraction,
                    "shuffle_buffer": args.shuffle_buffer,
                    "preset": args.preset,
                },
                history=history,
                include_optimizer=True,
                extra={"rng_by_rank": rngs, "world_size": world},
            )
            if logger is not None:
                logger.log(
                    {
                        "event": "checkpoint",
                        "kind": "last",
                        "step": step,
                        "tokens": tokens_seen,
                        "best_val": best_val,
                        "data_mode": data_mode,
                    }
                )
        barrier()

    completed = False
    try:
        while step < total_steps:
            lr = lr_at_tokens(tokens_seen, args.plan_tokens, args.warmup_tokens, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            finite = True
            for micro in range(args.grad_accum):
                x_np, y_np = packed.next_batch(args.batch_size)
                x = torch.from_numpy(x_np).to(device, non_blocking=True)
                y = torch.from_numpy(y_np).to(device, non_blocking=True)
                sync_context = (
                    model.no_sync()
                    if world > 1 and micro < args.grad_accum - 1
                    else _NullContext()
                )
                with sync_context:
                    with autocast_context(device, amp_dtype, use_amp):
                        _, loss, _ = model(x, y)
                        scaled = loss / args.grad_accum
                    if not math.isfinite(float(loss.detach().float().item())):
                        finite = False
                    if scaler_enabled:
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                micro_losses.append(float(loss.detach().float().item()))

            tokens_seen += tokens_per_step
            if scaler_enabled:
                scaler.unscale_(optimizer)
            if finite:
                grad_norm_value = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                )
                if not math.isfinite(grad_norm_value):
                    finite = False
            if finite:
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                skipped_steps += 1
                if scaler_enabled:
                    scaler.update()
                if is_main:
                    print(f"[WARN] step {step + 1}: non-finite loss/gradient; optimizer step skipped.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            step += 1
            step_loss = sum(micro_losses) / max(1, len(micro_losses))
            running_loss.append(step_loss)

            if args.log_every and step % args.log_every == 0:
                now = time.time()
                elapsed = max(1e-6, now - started)
                done_tokens = max(1, tokens_seen - start_tokens)
                rate = done_tokens / elapsed
                remaining = max(0, total_steps - step)
                eta = remaining * tokens_per_step / max(1.0, rate)
                avg_loss = sum(running_loss) / len(running_loss)
                running_loss = []
                if is_main:
                    alloc, peak = gpu_memory(device)
                    print(
                        f"step {step:>7,}/{total_steps:,} | loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | gnorm {grad_norm_value:.2f} | "
                        f"{rate:,.0f} tok/s | ETA {_format_seconds(eta)} | "
                        f"gpu {alloc:.1f}/{peak:.1f} GiB",
                        flush=True,
                    )
                    logger.log(
                        {
                            "event": "train",
                            "step": step,
                            "tokens": tokens_seen,
                            "loss": avg_loss,
                            "lr": lr,
                            "grad_norm": grad_norm_value,
                            "tokens_per_sec": rate,
                            "skipped_steps": skipped_steps,
                        }
                    )

            should_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step == total_steps)
            if should_eval:
                val_loss = evaluate(
                    model,
                    val_data,
                    cfg,
                    device,
                    args.eval_iters,
                    args.batch_size,
                    use_amp,
                    amp_dtype,
                    seed=args.seed + step,
                )
                improved = val_loss < best_val - 1e-4
                if improved:
                    best_val = val_loss
                if is_main:
                    print(
                        f"[eval] step {step:,} | val_loss {val_loss:.4f} | best {best_val:.4f}"
                        + (" *" if improved else ""),
                        flush=True,
                    )
                    logger.log(
                        {
                            "event": "eval",
                            "step": step,
                            "tokens": tokens_seen,
                            "val_loss": val_loss,
                            "best_val": best_val,
                            "improved": improved,
                        }
                    )
                history.append(
                    {"step": step, "tokens": tokens_seen, "val_loss": val_loss, "best_val": best_val}
                )
                if improved and is_main:
                    save_checkpoint(
                        best_path,
                        model=model,
                        config=cfg,
                        mode="pretrain",
                        step=step,
                        tokens_seen=tokens_seen,
                        best_val=best_val,
                        optimizer=optimizer,
                        scaler=scaler,
                        tokenizer_fp=tok_fp,
                        rng_state=None,
                        data_state=None,
                        train_args={"plan_tokens": args.plan_tokens, "preset": args.preset},
                        history=history,
                        include_optimizer=False,
                    )
                    if logger is not None:
                        logger.log({"event": "checkpoint", "kind": "best", "step": step, "val_loss": val_loss})

            if args.save_every and step % args.save_every == 0:
                save_last()

            if stop.requested:
                if is_main:
                    print("[stop] Signal received: saving checkpoint and exiting.", flush=True)
                save_last()
                stopped_early = True
                break

        if not stopped_early:
            save_last()
        completed = True
    finally:
        if is_main and completed:
            elapsed = max(1e-6, time.time() - started)
            trained_steps = step - start_step
            trained_tokens = max(0, tokens_seen - start_tokens)
            print(
                f"[done] steps={step:,} (+{trained_steps:,}) tokens={tokens_seen:,} "
                f"(+{trained_tokens:,}) best_val={best_val:.4f} "
                f"rate={trained_tokens / elapsed:,.0f} tok/s",
                flush=True,
            )
            if logger is not None:
                logger.log(
                    {
                        "event": "finish",
                        "step": step,
                        "tokens": tokens_seen,
                        "best_val": best_val,
                        "skipped_steps": skipped_steps,
                    }
                )
        cleanup_distributed()
    return 0


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
