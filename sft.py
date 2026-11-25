"""Supervised fine-tuning (SFT) for the English edition.

Prompt/header tokens are masked with ``-100``; only assistant content and the
final EOS are trained.  The file supports resumable checkpoints (optimizer, AMP
scaler, RNG and the exact epoch/batch position).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_model_weights,
    peek_checkpoint,
    save_checkpoint,
)
from config import (
    BOS_ID,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_TOKENIZER_DIR,
    PAD_ID,
    ModelConfig,
    ensure_dir,
)
from model import GPT
from tokenizer import (
    build_chat_ids,
    load_tokenizer,
    normalize_messages,
    tokenizer_fingerprint,
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
    human_bytes,
    init_nccl,
    is_distributed,
    lr_at_step,
    make_grad_scaler,
    resolve_amp,
    restore_rng_state,
    seed_everything,
    unwrap_model,
)

DEFAULT_SFT_DATA = Path(__file__).resolve().parent / "data" / "sft_sample_en.jsonl"


@dataclass
class SftExample:
    ids: List[int]
    labels: List[int]
    prompt_key: str
    source: str = ""

    @property
    def trainable(self) -> int:
        return sum(1 for x in self.labels if x != -100)


def _prompt_key(messages: Sequence[Dict[str, str]]) -> str:
    payload = "|".join(f"{m['role']}:{m['content']}" for m in messages[:-1])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def truncate_example(
    ids: List[int],
    labels: List[int],
    block_size: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Keep the prompt tail and the answer prefix when an example is too long."""
    if len(ids) != len(labels):
        raise ValueError("ids and labels must have the same length")
    if not ids or all(x == -100 for x in labels):
        return None
    if len(ids) <= block_size:
        out_ids, out_labels = list(ids), list(labels)
    else:
        first_train = next(i for i, x in enumerate(labels) if x != -100)
        prompt_ids, prompt_labels = ids[:first_train], labels[:first_train]
        answer_ids, answer_labels = ids[first_train:], labels[first_train:]
        keep_prompt = max(0, block_size - len(answer_ids))
        keep_answer = min(len(answer_ids), block_size - keep_prompt)
        start = max(0, len(prompt_ids) - keep_prompt)
        out_ids = prompt_ids[start:] + answer_ids[:keep_answer]
        out_labels = prompt_labels[start:] + answer_labels[:keep_answer]
    # Keep a leading BOS for consistency with generation if there is room.
    if out_ids and out_ids[0] != BOS_ID and len(out_ids) < block_size:
        out_ids = [BOS_ID] + out_ids
        out_labels = [-100] + out_labels
    if sum(1 for x in out_labels if x != -100) < 1:
        return None
    return out_ids, out_labels


def load_examples(
    paths: Sequence[str | Path],
    sp,
    block_size: int,
    max_examples: int = 0,
) -> List[SftExample]:
    examples: List[SftExample] = []
    seen_prompts: Dict[str, int] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SFT data file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = normalize_messages(obj)
                    ids, labels = build_chat_ids(messages, sp)
                except Exception as exc:
                    raise ValueError(f"{p}:{line_no}: {exc}") from exc
                truncated = truncate_example(ids, labels, block_size)
                if truncated is None:
                    continue
                key = _prompt_key(messages)
                examples.append(SftExample(truncated[0], truncated[1], key, str(p)))
                seen_prompts[key] = seen_prompts.get(key, 0) + 1
                if max_examples and len(examples) >= max_examples:
                    return examples
    if not examples:
        raise ValueError("No usable SFT examples were loaded")
    return examples


def split_examples(
    examples: Sequence[SftExample],
    val_ratio: float = 0.05,
    seed: int = 1337,
) -> Tuple[List[SftExample], List[SftExample], List[str]]:
    if len(examples) < 2:
        raise ValueError("SFT needs at least 2 examples")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    groups: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.prompt_key, []).append(i)
    keys = sorted(groups)
    target = max(1, int(round(len(examples) * val_ratio)))
    rng = __import__("random").Random(seed)
    rng.shuffle(keys)
    val_idx: List[int] = []
    for key in keys:
        if len(val_idx) >= target:
            break
        if len(val_idx) + len(groups[key]) >= len(examples):
            continue
        val_idx.extend(groups[key])
    if not val_idx:
        key = keys[0]
        val_idx = list(groups[key])
    val_set = set(val_idx)
    if len(val_set) >= len(examples):
        raise ValueError("SFT split leaves no training examples; add more data")
    train = [ex for i, ex in enumerate(examples) if i not in val_set]
    val = [ex for i, ex in enumerate(examples) if i in val_set]
    val_keys = sorted({examples[i].prompt_key for i in val_set})
    return train, val, val_keys


class SftStream:
    """Resumable epoch/batch iterator (all DDP ranks see the same batches)."""

    def __init__(
        self,
        examples: Sequence[SftExample],
        batch_size: int,
        seed: int = 1337,
        shuffle: bool = True,
    ):
        if not examples:
            raise ValueError("SftStream requires at least one example")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.examples = list(examples)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._rng = __import__("random").Random(self.seed)
        self.epoch = 0
        self.pos = 0
        self.order = list(range(len(self.examples)))
        self._new_epoch()

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.pos = 0
        self.order = list(range(len(self.examples)))
        if self.shuffle:
            self._rng.shuffle(self.order)

    def next_batch(self) -> List[SftExample]:
        if self.pos >= len(self.order):
            self._new_epoch()
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += len(idx)
        return [self.examples[i] for i in idx]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": 1,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "pos": self.pos,
            "order": list(self.order),
            "rng": self._rng.getstate(),
            "count": len(self.examples),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if int(state.get("count", len(self.examples))) != len(self.examples):
            raise ValueError("SFT data state was saved for a different number of examples")
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError("SFT data state batch_size does not match --batch-size")
        self.epoch = int(state.get("epoch", self.epoch))
        self.pos = int(state.get("pos", 0))
        self.order = [int(i) for i in state.get("order", self.order)]
        if state.get("rng") is not None:
            self._rng.setstate(state["rng"])
        if len(self.order) != len(self.examples):
            raise ValueError("SFT data state order does not match the dataset")


def collate(batch: Sequence[SftExample], pad_id: int = PAD_ID):
    max_len = max(len(ex.ids) for ex in batch)
    n = len(batch)
    x = torch.full((n, max_len), int(pad_id), dtype=torch.long)
    y = torch.full((n, max_len), -100, dtype=torch.long)
    mask = torch.zeros((n, max_len), dtype=torch.bool)
    for i, ex in enumerate(batch):
        length = len(ex.ids)
        x[i, :length] = torch.tensor(ex.ids, dtype=torch.long)
        y[i, :length] = torch.tensor(ex.labels, dtype=torch.long)
        mask[i, :length] = True
    return x, y, mask


def _iter_batches(examples: Sequence[SftExample], batch_size: int) -> Iterable[List[SftExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start:start + batch_size])


@torch.no_grad()
def evaluate_sft(
    model: nn.Module,
    examples: Sequence[SftExample],
    cfg: ModelConfig,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in _iter_batches(examples, batch_size):
        x, y, mask = collate(batch)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with autocast_context(device, amp_dtype, use_amp):
            _, loss, _ = model(x, targets=y, attention_mask=mask)
        tokens = int((y != -100).sum().item())
        if tokens:
            total_loss += float(loss.detach().float().item()) * tokens
            total_tokens += tokens
    if was_training:
        model.train()
    value = total_loss / max(1, total_tokens)
    return distributed_mean(value, device)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="English SFT (assistant-only loss).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", nargs="+", default=[str(DEFAULT_SFT_DATA)])
    ap.add_argument("--init", default="", help="Pretrained checkpoint (default: auto).")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR / "tokenizer.model"))
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4, help="per GPU")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr", type=float, default=5e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = evaluate at epoch end")
    ap.add_argument("--resume", default="", help="SFT checkpoint path or 'auto'.")
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0)
    return ap


def _resolve_init(args: argparse.Namespace, ckpt_dir: Path) -> Path:
    if args.init:
        return Path(args.init)
    base_dir = Path(args.checkpoint_dir)
    if args.smoke_test:
        for name in ("pretrain_best.pth", "pretrain_last.pth"):
            candidate = ckpt_dir / name
            if candidate.exists():
                return candidate
        return ckpt_dir / "pretrain_last.pth"
    for name in ("pretrain_best.pth", "pretrain_last.pth"):
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return base_dir / "pretrain_best.pth"


def main(argv: Optional[List[str]] = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    smoke = bool(args.smoke_test)
    if smoke:
        args.batch_size = 2
        args.grad_accum = 1
        args.epochs = 2
        args.lr = 1e-3
        args.min_lr = 1e-4
        args.warmup_steps = 2
        args.eval_every = 1
        args.log_every = 1
        args.amp = "auto"
        default_tok = Path(DEFAULT_TOKENIZER_DIR) / "tokenizer.model"
        if Path(args.tokenizer) == default_tok:
            args.tokenizer = str(Path(DEFAULT_TOKENIZER_DIR) / "smoke" / "tokenizer.model")

    base_ckpt_dir = ensure_dir(Path(args.checkpoint_dir))
    ckpt_dir = ensure_dir(base_ckpt_dir / ("smoke" if smoke else ""))
    runs_dir = ensure_dir(Path(args.runs_dir))
    last_path = ckpt_dir / "sft_last.pth"
    best_path = ckpt_dir / "sft_best.pth"
    init_path = _resolve_init(args, ckpt_dir)
    if not init_path.exists():
        raise SystemExit(
            f"Pretrained checkpoint not found: {init_path}\n"
            "Run pretraining first, or pass --init path/to/checkpoint."
        )

    info = peek_checkpoint(init_path)
    if info.edition != "en":
        raise SystemExit(
            f"{init_path} is not an English-edition checkpoint (edition={info.edition!r})."
        )
    cfg = ModelConfig.from_dict(info.config)
    if info.mode not in (None, "pretrain"):
        raise SystemExit(f"--init must be a pretraining checkpoint, got mode={info.mode!r}")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise SystemExit(
            f"Tokenizer not found: {tokenizer_path}\n"
            "Train it with: python tokenizer.py --train --sample-mb 200"
        )
    sp = load_tokenizer(tokenizer_path, expected_vocab=cfg.vocab_size)
    tok_fp = tokenizer_fingerprint(tokenizer_path)

    examples = load_examples(args.data, sp, cfg.block_size, max_examples=args.max_examples)
    train_ex, val_ex, val_keys = split_examples(examples, args.val_ratio, args.seed)

    steps_per_epoch = max(1, math.ceil(len(train_ex) / (args.batch_size * args.grad_accum)))
    total_steps = steps_per_epoch * max(1, args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    if args.dry_run:
        print(json.dumps(
            {
                "init": str(init_path),
                "mode": info.mode,
                "parameters": cfg.param_count(),
                "tokenizer_vocab": int(sp.vocab_size()),
                "examples": len(examples),
                "train_examples": len(train_ex),
                "val_examples": len(val_ex),
                "val_prompt_keys": len(val_keys),
                "block_size": cfg.block_size,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "trainable_tokens": sum(ex.trainable for ex in train_ex),
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
            },
            indent=2,
        ))
        return 0

    ddp, rank, local_rank, world = get_dist_env()
    if world > 1 and not torch.cuda.is_available():
        raise SystemExit("Distributed SFT requires CUDA/NCCL")
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    seed_everything(args.seed, rank)
    if world > 1:
        device = init_nccl(rank, local_rank, world)
    elif device.type == "cuda":
        torch.cuda.set_device(local_rank)
    use_amp, amp_dtype, scaler_enabled = resolve_amp(args.amp, device)

    model = GPT(cfg).to(device)
    actual = sum(p.numel() for p in model.parameters())
    if actual != cfg.param_count():
        raise SystemExit(f"Model has {actual:,} parameters, config says {cfg.param_count():,}")
    load_model_weights(
        init_path,
        model,
        config=cfg,
        device=device,
        tokenizer_fp=tok_fp,
        expected_mode="pretrain",
    )
    model.train()

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(scaler_enabled)
    stream = SftStream(train_ex, args.batch_size, seed=args.seed)

    step = 0
    tokens_seen = 0
    best_val = float("inf")
    history: List[Dict[str, Any]] = []
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
                expected_mode="sft",
                tokenizer_fp=tok_fp,
                load_optimizer=True,
                load_data=True,
                load_rng=(world == 1),
            )
            step = int(loaded.info.step)
            tokens_seen = int(loaded.info.tokens_seen)
            best_val = float(loaded.info.best_val)
            history = list(loaded.history)
            saved_keys = sorted((loaded.train_args or {}).get("val_keys") or [])
            if saved_keys and saved_keys != val_keys:
                raise CheckpointError(
                    "SFT validation split changed since the checkpoint was written "
                    "(data files or --val-ratio/--seed changed). Start a new SFT run "
                    "with --from-scratch or restore the original data."
                )
            if world > 1 and loaded.rng_by_rank and len(loaded.rng_by_rank) > rank:
                restore_rng_state(loaded.rng_by_rank[rank])
            if loaded.data_state:
                state = loaded.data_state
                if isinstance(state, list):
                    if len(state) != world:
                        raise CheckpointError(
                            f"Saved SFT data state is for {len(state)} rank(s), "
                            f"current world size is {world}. Resume with the same GPU count."
                        )
                    state = state[rank]
                stream.load_state_dict(state)
            if is_main:
                print(
                    f"[resume] {resume_path} | step={step:,} tokens={tokens_seen:,} "
                    f"best_val={best_val:.4f}",
                    flush=True,
                )

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    logger = JsonlLogger(runs_dir / ("sft_smoke.jsonl" if smoke else "sft.jsonl")) if is_main else None
    if is_main:
        print("=" * 74)
        print("English edition SFT")
        print(f"  init          : {init_path} (mode={info.mode})")
        print(f"  parameters    : {actual:,}")
        print(f"  examples      : {len(examples)} (train {len(train_ex)}, val {len(val_ex)})")
        print(f"  steps/epoch   : {steps_per_epoch} | total steps: {total_steps}")
        print(f"  batch/accum   : {args.batch_size} x {args.grad_accum} per GPU, world={world}")
        print(f"  lr            : {args.lr:g} -> {args.min_lr:g}")
        print(f"  amp           : {use_amp} ({amp_dtype}, scaler={scaler_enabled})")
        print(f"  last/best     : {last_path.name} / {best_path.name}")
        print("=" * 74, flush=True)

    stop = StopFlag()
    stop.install()
    last_saved_step = -1

    def save_last() -> None:
        nonlocal last_saved_step
        states = gather_object(stream.state_dict())
        rngs = gather_object(capture_rng_state())
        if is_main:
            save_checkpoint(
                last_path,
                model=model,
                config=cfg,
                mode="sft",
                step=step,
                tokens_seen=tokens_seen,
                best_val=best_val,
                optimizer=optimizer,
                scaler=scaler,
                tokenizer_fp=tok_fp,
                rng_state=rngs[0],
                data_state=states[0] if world == 1 else states,
                train_args={
                    "val_keys": val_keys,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "seed": args.seed,
                    "data_files": [str(p) for p in args.data],
                    "val_ratio": args.val_ratio,
                    "world_size": world,
                    "init": str(init_path),
                },
                history=history,
                include_optimizer=True,
                extra={"rng_by_rank": rngs, "world_size": world},
            )
            if logger is not None:
                logger.log({"event": "checkpoint", "kind": "last", "step": step, "tokens": tokens_seen})
        barrier()
        last_saved_step = step

    started = __import__("time").time()
    running_loss: List[float] = []
    grad_norm_value = 0.0
    skipped = 0
    completed = False

    try:
        while step < total_steps:
            lr = lr_at_step(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            finite = True
            micro_losses: List[float] = []
            micro_tokens: List[int] = []
            for micro in range(args.grad_accum):
                batch = stream.next_batch()
                x, y, mask = collate(batch)
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                n_tokens = int((y != -100).sum().item())
                if n_tokens == 0:
                    continue
                sync_context = (
                    model.no_sync() if world > 1 and micro < args.grad_accum - 1 else _NullContext()
                )
                with sync_context:
                    with autocast_context(device, amp_dtype, use_amp):
                        _, loss, _ = model(x, targets=y, attention_mask=mask)
                        scaled = loss / args.grad_accum
                    if not math.isfinite(float(loss.detach().float().item())):
                        finite = False
                    if scaler_enabled:
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                micro_losses.append(float(loss.detach().float().item()))
                micro_tokens.append(n_tokens)

            step += 1
            tokens_seen += sum(micro_tokens)
            if not micro_losses:
                finite = False
            if scaler_enabled:
                scaler.unscale_(optimizer)
            if finite:
                grad_norm_value = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                if not math.isfinite(grad_norm_value):
                    finite = False
            if finite:
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                skipped += 1
                if scaler_enabled:
                    scaler.update()
                if is_main:
                    print(f"[WARN] step {step}: non-finite loss; skipped.", flush=True)
            optimizer.zero_grad(set_to_none=True)

            step_loss = (
                sum(l * t for l, t in zip(micro_losses, micro_tokens)) / max(1, sum(micro_tokens))
                if micro_losses else float("nan")
            )
            running_loss.append(step_loss)

            if args.log_every and step % args.log_every == 0 and is_main:
                avg = sum(running_loss) / len(running_loss)
                running_loss = []
                print(
                    f"step {step:>5,}/{total_steps:,} | loss {avg:.4f} | lr {lr:.2e} | "
                    f"gnorm {grad_norm_value:.2f} | tokens {tokens_seen:,}",
                    flush=True,
                )
                if logger is not None:
                    logger.log(
                        {
                            "event": "train",
                            "step": step,
                            "loss": avg,
                            "lr": lr,
                            "grad_norm": grad_norm_value,
                            "tokens": tokens_seen,
                        }
                    )

            at_epoch_end = step % steps_per_epoch == 0 or step == total_steps
            should_eval = (
                at_epoch_end if args.eval_every <= 0 else (step % args.eval_every == 0 or step == total_steps)
            )
            if should_eval:
                val_loss = evaluate_sft(
                    model, val_ex, cfg, device, args.batch_size, use_amp, amp_dtype
                )
                improved = val_loss < best_val - 1e-4
                if improved:
                    best_val = val_loss
                history.append(
                    {"step": step, "tokens": tokens_seen, "val_loss": val_loss, "best_val": best_val}
                )
                if is_main:
                    print(
                        f"[eval] step {step:,} | val_loss {val_loss:.4f} | best {best_val:.4f}"
                        + (" *" if improved else ""),
                        flush=True,
                    )
                    if logger is not None:
                        logger.log(
                            {"event": "eval", "step": step, "val_loss": val_loss, "best_val": best_val}
                        )
                if improved and is_main:
                    save_checkpoint(
                        best_path,
                        model=model,
                        config=cfg,
                        mode="sft",
                        step=step,
                        tokens_seen=tokens_seen,
                        best_val=best_val,
                        optimizer=optimizer,
                        scaler=scaler,
                        tokenizer_fp=tok_fp,
                        train_args={"val_keys": val_keys, "epochs": args.epochs},
                        history=history,
                        include_optimizer=False,
                    )

            if at_epoch_end:
                save_last()

            if stop.requested:
                if is_main:
                    print("[stop] Saving SFT checkpoint and exiting.", flush=True)
                break

        if last_saved_step != step:
            save_last()
        completed = True
    finally:
        if is_main and completed:
            elapsed = max(1e-6, __import__("time").time() - started)
            print(
                f"[done] SFT steps={step:,} tokens={tokens_seen:,} best_val={best_val:.4f} "
                f"skipped={skipped} ({tokens_seen / elapsed:,.0f} tok/s)",
                flush=True,
            )
            if logger is not None:
                logger.log(
                    {"event": "finish", "step": step, "tokens": tokens_seen, "best_val": best_val}
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
