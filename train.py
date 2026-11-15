import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import (
    GPTLanguageModel,
    ARCH_VERSION,
    BLOCK_SIZE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
)
from preprocess import preprocess
from tokenizer import (
    VOCAB_SIZE,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    train_tokenizer,
    load_tokenizer,
    encode,
)

DATA_PATH = "data.txt"
CLEAN_PATH = "cleaned_data.txt"
TRAIN_BIN = "train_tokens.bin"
VAL_BIN = "val_tokens.bin"
META_PATH = "data_meta.json"
PRETRAIN_LAST = "pretrain_last.pth"
PRETRAIN_BEST = "pretrain_best.pth"
SFT_LAST = "instruct_last.pth"
SFT_BEST = "instruct_best.pth"
INSTRUCTION_PATH = "instruction_data.jsonl"
DISTILL_PATH = "deepseek_distill.jsonl"

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_TRAIN_TOKENS = 500_000_000
SFT_STEPS = 2000
EVAL_INTERVAL = 250
EVAL_ITERS = 20
LEARNING_RATE = 1.5e-4
MIN_LR = 1.5e-5
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_RATIO = 0.01
SEED = 1337

# Long preprocessing happens before NCCL is initialized. Gloo is used only
# as a CPU coordination barrier while rank 0 prepares the pretraining files.
PREP_TIMEOUT = timedelta(hours=2)
DDP_TIMEOUT = timedelta(minutes=30)


def get_dist_env():
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    return (
        True,
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )


def initialize_nccl(rank, local_rank, world):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP training yêu cầu CUDA/NCCL, nhưng CUDA không khả dụng. "
            "Hãy bật GPU trong Kaggle Session Options."
        )

    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} không hợp lệ; chỉ nhìn thấy {device_count} GPU."
        )

    # Explicit mapping: rank 0 -> cuda:0, rank 1 -> cuda:1, ...
    torch.cuda.set_device(local_rank)

    # device_id prevents NCCL from guessing the process-to-GPU mapping.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=DDP_TIMEOUT,
        device_id=local_rank,
    )

    current = torch.cuda.current_device()
    if current != local_rank:
        raise RuntimeError(
            f"DDP mapping lỗi: rank {rank} đang ở cuda:{current}, "
            f"nhưng LOCAL_RANK={local_rank}."
        )


def prepare_pretrain_assets(rank, world, force=False):
    """Prepare preprocessing/tokenizer/bin files BEFORE NCCL starts.

    In multi-GPU runs rank 0 does all expensive file work while every other
    rank waits in a CPU Gloo process group. This avoids NCCL barrier timeouts.
    """
    ddp = world > 1

    if ddp:
        # Temporary CPU-only rendezvous. No CUDA/NCCL is touched here.
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=world,
            timeout=PREP_TIMEOUT,
        )

    try:
        if rank == 0:
            # Rebuild means start the entire data pipeline over. Otherwise,
            # reuse already prepared files whenever possible.
            if force:
                print("[1/3] Làm sạch data.txt...")
                if not os.path.exists(DATA_PATH):
                    raise FileNotFoundError(
                        "Không tìm thấy data.txt để rebuild. "
                        "Hãy thêm data.txt vào /kaggle/working/ hoặc bỏ --rebuild-data."
                    )
                preprocess(DATA_PATH, CLEAN_PATH)
            elif not os.path.exists(CLEAN_PATH):
                print("[1/3] cleaned_data.txt chưa có, đang làm sạch data.txt...")
                if not os.path.exists(DATA_PATH):
                    raise FileNotFoundError("Không tìm thấy data.txt.")
                preprocess(DATA_PATH, CLEAN_PATH)

            if force or not os.path.exists("tokenizer.model") or not os.path.exists("tokenizer.vocab"):
                print("[2/3] Huấn luyện SentencePiece...")
                train_tokenizer(CLEAN_PATH, VOCAB_SIZE, force=force)
            else:
                print("[2/3] Dùng tokenizer hiện có.")

            sp = load_tokenizer()

            # Build memmap bins here, still outside NCCL.
            build_bins(
                CLEAN_PATH,
                sp,
                force=force,
            )
        else:
            sp = None

        # Rank 1 waits on Gloo (CPU), not NCCL/GPU.
        if ddp:
            dist.barrier()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


def build_bins(data_path, sp, force=False):
    wanted = {
        "source": file_fingerprint(data_path),
        "vocab_size": sp.vocab_size(),
        "val_ratio": VAL_RATIO,
        "format": 3,
    }

    meta = None
    if not force and os.path.exists(META_PATH):
        try:
            meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
        except Exception:
            meta = None

    if meta == wanted and os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        print("[3/3] Dữ liệu tokenized đã tồn tại, dùng lại.")
        return

    for path in (TRAIN_BIN, VAL_BIN):
        if os.path.exists(path):
            os.remove(path)

    print("[3/3] Tokenize toàn bộ dataset theo streaming...")
    rng = random.Random(SEED)
    train_count = 0
    val_count = 0
    train_f = open(TRAIN_BIN, "wb", buffering=8 * 1024 * 1024)
    val_f = open(VAL_BIN, "wb", buffering=8 * 1024 * 1024)
    train_buf, val_buf = [], []

    try:
        with open(
            data_path,
            "r",
            encoding="utf-8",
            errors="replace",
            buffering=8 * 1024 * 1024,
        ) as f:
            for line_no, line in enumerate(f, 1):
                line = line.replace("\r", "").replace("\x00", "").strip()
                if len(line) < 2:
                    continue

                ids = sp.encode(line, out_type=int)
                if len(ids) < 2:
                    continue

                seq = [BOS_ID] + ids + [EOS_ID]
                if rng.random() < VAL_RATIO:
                    val_buf.extend(seq)
                    val_count += len(seq)
                else:
                    train_buf.extend(seq)
                    train_count += len(seq)

                if len(train_buf) >= 1_000_000:
                    np.asarray(train_buf, dtype=np.uint16).tofile(train_f)
                    train_buf.clear()

                if len(val_buf) >= 1_000_000:
                    np.asarray(val_buf, dtype=np.uint16).tofile(val_f)
                    val_buf.clear()

                if line_no % 250_000 == 0:
                    print(
                        f"{line_no:,} dòng | train {train_count:,} | val {val_count:,}"
                    )

        if train_buf:
            np.asarray(train_buf, dtype=np.uint16).tofile(train_f)
        if val_buf:
            np.asarray(val_buf, dtype=np.uint16).tofile(val_f)
    finally:
        train_f.close()
        val_f.close()

    Path(META_PATH).write_text(
        json.dumps(wanted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã tạo token: train {train_count:,} | val {val_count:,}")


def file_fingerprint(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def get_batch(data, device, rng):
    max_start = len(data) - BLOCK_SIZE - 1
    if max_start <= 0:
        raise RuntimeError("Dataset nhỏ hơn context window.")

    starts = rng.integers(0, max_start, size=BATCH_SIZE)
    x = np.stack(
        [np.asarray(data[i : i + BLOCK_SIZE], dtype=np.int64) for i in starts]
    )
    y = np.stack(
        [np.asarray(data[i + 1 : i + BLOCK_SIZE + 1], dtype=np.int64) for i in starts]
    )

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def load_instruction_data(paths, sp):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    rows = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL lỗi ở {path}, dòng {line_no}: {exc}"
                    ) from exc

                user = str(obj.get("user", "")).strip()
                assistant = str(obj.get("assistant", "")).strip()
                if not user or not assistant:
                    continue

                text = f"<|user|>\n{user}\n<|assistant|>\n{assistant}"
                tokens = encode(text, sp=sp, add_bos=True, add_eos=True)
                if len(tokens) < 3:
                    continue

                prefix = encode(
                    f"<|user|>\n{user}\n<|assistant|>\n",
                    sp=sp,
                    add_bos=True,
                )
                if len(prefix) >= BLOCK_SIZE or len(tokens) <= len(prefix):
                    continue

                # Trim prompt first; very long answers may also lose their earliest tokens.
                if len(tokens) > BLOCK_SIZE:
                    trim = len(tokens) - BLOCK_SIZE
                    tokens = [tokens[0]] + tokens[trim + 1:]
                    prefix = [prefix[0]] + prefix[trim + 1:]
                assistant_start = len(prefix)

                rows.append((tokens, assistant_start, tuple(prefix)))

    if not rows:
        raise RuntimeError("Không có mẫu SFT hợp lệ.")

    return rows

def get_sft_batch(rows, device, rng):
    batch = [rows[int(rng.integers(0, len(rows)))] for _ in range(BATCH_SIZE)]
    max_len = min(BLOCK_SIZE, max(len(tokens) for tokens, _, _ in batch))
    xs, ys = [], []

    for tokens, assistant_start, _ in batch:
        tokens = tokens[:max_len]
        x = tokens[:-1]
        y = tokens[1:]

        mask_start = max(0, assistant_start - 1)
        labels = [
            token if j >= mask_start else -100
            for j, token in enumerate(y)
        ]

        x += [PAD_ID] * (max_len - 1 - len(x))
        labels += [-100] * (max_len - 1 - len(labels))
        xs.append(x)
        ys.append(labels)

    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (
        1 + math.cos(math.pi * progress)
    )


def build_optimizer(model, device):
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim >= 2
            and "norm" not in name.lower()
            and "embedding" not in name.lower()
        ):
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    kwargs = {
        "lr": LEARNING_RATE,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    if device.type == "cuda":
        kwargs["fused"] = True

    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def save_checkpoint(path, model, optimizer, scaler, step, best_val, vocab_size, mode,
                   sft_val_indices=None):
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    obj = {
        "version": 8,
        "arch_version": ARCH_VERSION,
        "mode": mode,
        "sft_val_indices": sft_val_indices,
        "step": step,
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "best_val": best_val,
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
    }
    atomic_save(obj, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    device,
    mode,
    load_optimizer=True,
):
    if not os.path.exists(path):
        return 0, float("inf"), False

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        return 0, float("inf"), False

    expected = {
        "arch_version": ARCH_VERSION,
        "vocab_size": model.vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
    }

    if any(ckpt.get(key) != value for key, value in expected.items()):
        print(
            "Checkpoint không tương thích với kiến trúc hiện tại "
            f"(AI7-DeepSeek Lite arch {ARCH_VERSION})."
        )
        return 0, float("inf"), False

    ckpt_mode = ckpt.get("mode")
    if mode == "pretrain" and ckpt_mode not in (None, "pretrain"):
        print("Checkpoint SFT không thể dùng để resume pretrain.")
        return 0, float("inf"), False

    model.load_state_dict(ckpt["model"], strict=True)

    if load_optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"Không khôi phục được optimizer; dùng optimizer mới: {exc}")

    if load_optimizer and scaler.is_enabled() and ckpt.get("scaler"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    return (
        int(ckpt.get("step", -1)) + 1,
        float(ckpt.get("best_val", float("inf"))),
        True,
    )


@torch.no_grad()
def eval_pretrain(model, train_data, val_data, device, rng):
    model.eval()
    values = {}

    for name, data in (("train", train_data), ("val", val_data)):
        vals = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, device, rng)
            _, loss, _ = model(x, y)
            vals.append(loss.item())
        values[name] = sum(vals) / len(vals)

    model.train()
    return values


@torch.no_grad()
def eval_sft(model, rows, device, rng):
    model.eval()
    vals = []

    for _ in range(min(EVAL_ITERS, max(1, len(rows)))):
        x, y = get_sft_batch(rows, device, rng)
        _, loss, _ = model(x, y)
        vals.append(loss.item())

    model.train()
    return sum(vals) / len(vals)


def split_sft_rows(rows, seed=SEED, val_ratio=0.05, val_indices=None):
    if len(rows) < 2:
        raise RuntimeError("SFT cần ít nhất 2 mẫu để tách train/validation.")
    if val_indices is None:
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(row[2], []).append(i)
        keys = list(groups)
        np.random.default_rng(seed).shuffle(keys)
        val_indices = []
        target = max(1, int(len(rows) * val_ratio))
        for key in keys:
            if len(val_indices) + len(groups[key]) >= len(rows):
                continue
            val_indices.extend(groups[key])
            if len(val_indices) >= target:
                break
    val_idx = set(val_indices)
    if not val_idx or min(val_idx) < 0 or max(val_idx) >= len(rows) or len(val_idx) == len(rows):
        raise ValueError("Chỉ số validation trong checkpoint không khớp dữ liệu SFT hiện tại.")
    val_prompts = {rows[i][2] for i in val_idx}
    if any((row[2] in val_prompts) != (i in val_idx) for i, row in enumerate(rows)):
        raise ValueError("Checkpoint chia các câu hỏi trùng nhau giữa train và validation; hãy bắt đầu SFT mới.")
    if len(val_idx) == len(rows):
        raise ValueError("Không thể chia SFT: toàn bộ mẫu có cùng câu hỏi.")
    return ([row for i, row in enumerate(rows) if i not in val_idx],
            [row for i, row in enumerate(rows) if i in val_idx], sorted(val_idx))


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    seed += rank * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--rebuild-data", action="store_true")
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=MAX_TRAIN_TOKENS)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--sft-data", nargs="+", default=None, help="Một hoặc nhiều JSONL dùng cho SFT.")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    ddp, rank, local_rank, world = get_dist_env()

    if ddp and args.mode == "pretrain" and not args.smoke_test:
        # IMPORTANT: prepare all large files BEFORE initializing NCCL.
        # Only rank 0 touches the data; other ranks wait in CPU/Gloo.
        prepare_pretrain_assets(
            rank=rank,
            world=world,
            force=args.rebuild_data,
        )

    elif not ddp and args.mode == "pretrain" and not args.smoke_test:
        # Single-GPU / CPU preparation.
        prepare_pretrain_assets(rank=0, world=1, force=args.rebuild_data)

    # SFT intentionally does NOT touch data.txt or the Wikipedia token bins.
    if ddp:
        initialize_nccl(rank, local_rank, world)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")

    try:
        seed_everything(SEED, rank)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if args.smoke_test:
            model = GPTLanguageModel(VOCAB_SIZE).to(device)
            model.train()
            x = torch.randint(0, VOCAB_SIZE, (2, min(BLOCK_SIZE, 64)), device=device)
            y = torch.randint(0, VOCAB_SIZE, x.shape, device=device)
            logits, loss, _ = model(x, y)
            loss.backward()
            if rank == 0:
                params = sum(p.numel() for p in model.parameters())
                print(f"SMOKE TEST: loss={loss.item():.4f} | logits={tuple(logits.shape)} | params={params:,}")
                print("SMOKE TEST PASS")
            return

        # Tokenizer must exist for both modes.
        try:
            sp = load_tokenizer()
        except Exception as exc:
            raise RuntimeError(
                "Không tải được tokenizer. Hãy chạy pretrain/preprocess trước để "
                "tạo tokenizer.model và tokenizer.vocab."
            ) from exc

        if args.mode == "sft" and args.from_scratch:
            raise ValueError(
                "SFT phải bắt đầu từ checkpoint pretrained; không dùng --from-scratch."
            )

        if args.mode == "pretrain":
            if not os.path.exists(TRAIN_BIN) or not os.path.exists(VAL_BIN):
                # This normally means the preparation step was skipped or failed.
                if not os.path.exists(DATA_PATH):
                    raise FileNotFoundError(
                        "Không tìm thấy data.txt và cũng không có train_tokens.bin/val_tokens.bin."
                    )
                raise FileNotFoundError(
                    "Thiếu train_tokens.bin/val_tokens.bin. Hãy chạy lại với --rebuild-data."
                )

            train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
            val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
            total_tokens = args.max_tokens
            effective_tokens = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM * world
            total_steps = args.max_steps or max(1, total_tokens // effective_tokens)
            last_path = PRETRAIN_LAST
            best_path = PRETRAIN_BEST
            rng = np.random.default_rng(SEED + 17 * rank)
            rows = None

        else:
            sft_paths = list(args.sft_data) if args.sft_data else [INSTRUCTION_PATH]
            if args.sft_data is None and os.path.exists(DISTILL_PATH):
                sft_paths.append(DISTILL_PATH)
            rows = load_instruction_data(sft_paths, sp)
            resume_path = args.checkpoint or (SFT_LAST if os.path.exists(SFT_LAST) else PRETRAIN_BEST)
            saved_val_indices = None
            if os.path.basename(resume_path) in (SFT_LAST, SFT_BEST) and os.path.exists(resume_path):
                saved_val_indices = torch.load(resume_path, map_location="cpu", weights_only=False).get("sft_val_indices")
            rows, val_rows, val_indices = split_sft_rows(rows, val_indices=saved_val_indices)
            train_data = None
            val_data = None
            effective_tokens = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM * world
            total_steps = args.max_steps or SFT_STEPS
            last_path = SFT_LAST
            best_path = SFT_BEST
            rng = np.random.default_rng(SEED + 97 * rank)

        model = GPTLanguageModel(sp.vocab_size()).to(device)
        optimizer = build_optimizer(model, device)
        use_amp = device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and amp_dtype == torch.float16,
        )

        if args.checkpoint:
            checkpoint = args.checkpoint
        elif args.mode == "sft":
            checkpoint = SFT_LAST if os.path.exists(SFT_LAST) else PRETRAIN_BEST
        else:
            checkpoint = PRETRAIN_LAST

        if args.from_scratch:
            start_step = 0
            best_val = float("inf")
        else:
            # When starting SFT from pretrain_best.pth, load ONLY model weights
            # and use a fresh SFT optimizer/schedule. When resuming an SFT
            # checkpoint, restore optimizer/scaler/step as usual.
            sft_from_pretrain = (
                args.mode == "sft"
                and os.path.basename(checkpoint) in (PRETRAIN_BEST, PRETRAIN_LAST)
                and os.path.exists(checkpoint)
            )
            start_step, best_val, loaded = load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scaler,
                device,
                args.mode,
                load_optimizer=not sft_from_pretrain,
            )
            if not loaded:
                if args.mode == "sft":
                    raise RuntimeError(
                        f"Không thể tải checkpoint SFT/pretrain tương thích: {checkpoint}"
                    )
                start_step = 0
                best_val = float("inf")

            if sft_from_pretrain:
                start_step = 0
                best_val = float("inf")
                print("Đã nạp trọng số pretrained; bắt đầu SFT với optimizer mới.")

        if ddp:
            # MoE top-1 routing means some expert parameters may be unused on
            # a given rank. Let DDP detect/reduce those parameters instead of
            # hanging on an all-reduce.
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        if rank == 0:
            params = sum(
                p.numel()
                for p in (model.module if isinstance(model, DDP) else model).parameters()
            )
            print(f"Device: {device} | GPUs: {world}")
            print(f"Vocab: {sp.vocab_size()} | Parameters: {params:,}")
            print(
                f"Context: {BLOCK_SIZE} | Effective tokens/step: "
                f"{effective_tokens:,} | Max steps: {total_steps:,}"
            )
            print(f"Mode: {args.mode} | AMP: {amp_dtype if use_amp else 'off'}")
            if args.mode == "sft":
                print(f"Instruction examples: {len(rows):,} train | {len(val_rows):,} val")
                print("SFT data: " + ", ".join(sft_paths))
            elif args.rebuild_data:
                print("Data pipeline: rebuilt")
            else:
                print("Data pipeline: reused when compatible")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        last_step = max(start_step - 1, 0)

        for step in range(start_step, total_steps):
            last_step = step
            lr = get_lr(step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            for _ in range(GRAD_ACCUM):
                if args.mode == "pretrain":
                    x, y = get_batch(train_data, device, rng)
                else:
                    x, y = get_sft_batch(rows, device, rng)

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    _, loss, _ = model(x, y)
                    loss = loss / GRAD_ACCUM

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
                if args.mode == "pretrain":
                    metrics = eval_pretrain(model, train_data, val_data, device, rng)
                    train_loss = metrics["train"]
                    local_val_loss = metrics["val"]
                    ppl = math.exp(min(local_val_loss, 20))
                    metric_text = (
                        f"train {train_loss:.4f} | "
                        f"val {local_val_loss:.4f} | ppl {ppl:.2f}"
                    )
                else:
                    local_val_loss = eval_sft(model, val_rows, device, rng)
                    metric_text = f"sft_loss {local_val_loss:.4f}"

                val_loss = torch.tensor(
                    [local_val_loss],
                    dtype=torch.float32,
                    device=device,
                )
                if ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss = float(val_loss.item())

                elapsed = time.time() - t0
                if rank == 0:
                    print(
                        f"Step {step:6d} | lr {lr:.2e} | {metric_text} | "
                        f"elapsed {elapsed:.1f}s"
                    )

                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scaler,
                            step,
                            best_val,
                            sp.vocab_size(),
                            args.mode,
                            val_indices if args.mode == "sft" else None,
                        )
                        print(f"  -> lưu model tốt nhất: {best_path}")

                    save_checkpoint(
                        last_path,
                        model,
                        optimizer,
                        scaler,
                        step,
                        best_val,
                        sp.vocab_size(),
                        args.mode,
                        val_indices if args.mode == "sft" else None,
                    )

                if ddp:
                    dist.barrier()

        if rank == 0:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scaler,
                last_step,
                best_val,
                sp.vocab_size(),
                args.mode,
                val_indices if args.mode == "sft" else None,
            )
            print(f"Đã lưu checkpoint cuối: {last_path}")
            print(f"Model tốt nhất: {best_path}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
