"""AI9 audit suite.

Runs fast, deterministic checks against the current source tree and prints an
explicit PASS / FAIL / SKIP status for every check.

Usage:
    python test_audit.py
    python test_audit.py --full   # also run the heavy --smoke-test

Exit code is 0 only when no check FAILs. SKIP means the check could not run in
this environment (for example no CUDA GPU or no tokenizer.model yet), it is not
counted as a failure.
"""

import argparse
import importlib
import os
import py_compile
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch

from model import (
    ARCH_VERSION,
    BLOCK_SIZE,
    KV_LATENT,
    MOE_HIDDEN,
    MOE_TOP_K,
    N_EMBD,
    N_EXPERTS,
    N_HEAD,
    N_LAYER,
    GPTLanguageModel,
    RotaryEmbedding,
    model_config,
)
from train import (
    get_batch,
    get_lr,
    get_sft_batch,
    load_checkpoint,
    save_checkpoint,
)
from tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    USER_ID,
    VOCAB_SIZE,
    decode,
    encode,
    load_tokenizer,
    tokenizer_fingerprint,
)

PY_FILES = [
    "model.py",
    "tokenizer.py",
    "train.py",
    "chat.py",
    "preprocess.py",
    "rag.py",
    "agent.py",
    "web_search.py",
    "distill_generate.py",
    "test_audit.py",
]

RESULTS = []


class SkipTest(Exception):
    """Raised by a check that cannot run in the current environment."""


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


def check(name, fn):
    try:
        detail = fn() or ""
        record(name, "PASS", detail)
    except SkipTest as exc:
        record(name, "SKIP", str(exc))
    except Exception as exc:
        record(name, "FAIL", str(exc))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Source and imports
# ---------------------------------------------------------------------------

def test_compile():
    for name in PY_FILES:
        path = os.path.join(os.path.dirname(__file__), name)
        py_compile.compile(path, doraise=True)
    return f"{len(PY_FILES)} files"


def test_imports():
    modules = [
        "model",
        "tokenizer",
        "preprocess",
        "train",
        "chat",
        "rag",
        "agent",
        "web_search",
    ]
    for name in modules:
        importlib.import_module(name)
    return ", ".join(modules)


def test_distill_import():
    try:
        importlib.import_module("distill_generate")
    except ModuleNotFoundError as exc:
        if exc.name in ("transformers", "bitsandbytes", "accelerate"):
            raise SkipTest(f"missing optional dependency: {exc.name}") from exc
        raise
    return "distill_generate imported"


def test_cli_help():
    for script in ("train.py", "chat.py"):
        proc = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{script} --help failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-500:]}"
            )
    return "train.py --help, chat.py --help"


# ---------------------------------------------------------------------------
# 2. Configuration and model
# ---------------------------------------------------------------------------

def test_config():
    expected = {
        "arch_version": 8,
        "block_size": 768,
        "n_embd": 512,
        "n_head": 8,
        "n_layer": 12,
        "kv_latent": 128,
        "n_experts": 2,
        "moe_top_k": 1,
    }
    cfg = model_config()
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise AssertionError(f"{key}={cfg.get(key)}, expected {value}")
    constants = {
        "ARCH_VERSION": ARCH_VERSION,
        "BLOCK_SIZE": BLOCK_SIZE,
        "N_EMBD": N_EMBD,
        "N_HEAD": N_HEAD,
        "N_LAYER": N_LAYER,
        "KV_LATENT": KV_LATENT,
        "N_EXPERTS": N_EXPERTS,
        "MOE_TOP_K": MOE_TOP_K,
        "MOE_HIDDEN": MOE_HIDDEN,
    }
    expected_constants = {
        "ARCH_VERSION": 8,
        "BLOCK_SIZE": 768,
        "N_EMBD": 512,
        "N_HEAD": 8,
        "N_LAYER": 12,
        "KV_LATENT": 128,
        "N_EXPERTS": 2,
        "MOE_TOP_K": 1,
        "MOE_HIDDEN": 512,
    }
    if constants != expected_constants:
        raise AssertionError(f"module constants changed: {constants}")
    if cfg["moe_hidden"] < MOE_HIDDEN:
        raise AssertionError("model_config moe_hidden is smaller than MOE_HIDDEN")
    return f"arch={ARCH_VERSION}, ctx={BLOCK_SIZE}"


def test_forward_backward():
    torch.manual_seed(42)
    model = GPTLanguageModel(256)
    model.train()
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    logits, loss, cache = model(x, y)
    assert logits.shape == (2, 32, 256), logits.shape
    assert cache is None
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    assert grads > 100, f"only {grads} parameters received gradients"
    return f"loss={loss.item():.4f}, grads={grads}"


# ---------------------------------------------------------------------------
# 3. RoPE and KV cache
# ---------------------------------------------------------------------------

def test_rope_relative_invariance():
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=8, max_position=64)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    for start in (0, 1, 7, 33):
        for delta in (1, 9, 20):
            q0 = rope.apply_rotary(q, 0)
            k0 = rope.apply_rotary(k, delta)
            qp = rope.apply_rotary(q, start)
            kp = rope.apply_rotary(k, start + delta)
            a = float((q0 * k0).sum())
            b = float((qp * kp).sum())
            if abs(a - b) > 2e-5:
                raise AssertionError(
                    f"RoPE dot changed for start={start}, delta={delta}: {a} vs {b}"
                )
    return "relative positions preserved"


def test_rope_bounds():
    rope = RotaryEmbedding(dim=8, max_position=16)
    try:
        rope.apply_rotary(torch.randn(1, 1, 1, 8), 16)
    except ValueError:
        return "out-of-range position rejected"
    raise AssertionError("RoPE accepted an out-of-range position")


def test_kv_cache_matches_full():
    torch.manual_seed(7)
    model = GPTLanguageModel(256)
    model.eval()
    seq = torch.randint(0, 256, (1, 24))

    with torch.no_grad():
        full, _, _ = model(seq)
        part1, _, cache = model(seq[:, :7], use_cache=True)
        part2, _, cache = model(seq[:, 7:17], past_key_values=cache, use_cache=True)
        part3, _, cache = model(seq[:, 17:], past_key_values=cache, use_cache=True)

    assert part1.shape[1] == 7 and part2.shape[1] == 10 and part3.shape[1] == 7
    diff = float((full[:, 17:] - part3).abs().max())
    if diff > 1e-3:
        raise AssertionError(f"cached logits differ from full forward: {diff:.3e}")
    if any(t.dtype != torch.float32 for layer in cache for t in layer):
        raise AssertionError("KV cache is not float32")
    return f"max_diff={diff:.2e}, cache=float32"


def test_kv_cache_window():
    torch.manual_seed(11)
    model = GPTLanguageModel(256)
    model.eval()
    warmup = torch.randint(0, 256, (1, BLOCK_SIZE))

    with torch.no_grad():
        _, _, cache = model(warmup, use_cache=True)
        assert cache[0][0].size(1) == BLOCK_SIZE
        step, _, cache = model(
            torch.randint(0, 256, (1, 1)), past_key_values=cache, use_cache=True
        )

    assert step.shape == (1, 1, 256)
    assert torch.isfinite(step).all()
    if cache[0][0].size(1) != BLOCK_SIZE:
        raise AssertionError(
            f"cache length {cache[0][0].size(1)} after overflow, expected {BLOCK_SIZE}"
        )
    return f"cache clipped to {BLOCK_SIZE}"


def test_generate_cache_equivalence():
    torch.manual_seed(123)
    model = GPTLanguageModel(256)
    model.eval()
    prompt = torch.randint(4, 256, (1, 5))
    out_cache = model.generate(
        prompt, max_new_tokens=6, temperature=0, eos_id=None, use_cache=True
    )
    out_plain = model.generate(
        prompt, max_new_tokens=6, temperature=0, eos_id=None, use_cache=False
    )
    if not torch.equal(out_cache, out_plain):
        raise AssertionError("cached and non-cached greedy generation differ")
    if not torch.equal(out_cache[:, :5], prompt):
        raise AssertionError("generate modified the prompt")
    return f"shape={tuple(out_cache.shape)}"


def test_generate_eos_and_penalty():
    torch.manual_seed(5)
    model = GPTLanguageModel(256)
    model.eval()
    prompt = torch.randint(4, 256, (1, 3))
    out = model.generate(
        prompt,
        max_new_tokens=12,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_id=EOS_ID,
        min_new_tokens=2,
    )
    assert out.shape[1] >= 5
    assert out.shape[1] <= 15
    greedy = model.generate(prompt, max_new_tokens=5, temperature=0, eos_id=None)
    greedy2 = model.generate(prompt, max_new_tokens=5, temperature=0, eos_id=None)
    if not torch.equal(greedy, greedy2):
        raise AssertionError("greedy generation is not deterministic")
    return f"len={out.shape[1]}, greedy deterministic"


# ---------------------------------------------------------------------------
# 4. Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer():
    if not os.path.exists("tokenizer.model"):
        raise SkipTest("tokenizer.model not built yet")
    sp = load_tokenizer()
    if sp.vocab_size() < VOCAB_SIZE - 300 or sp.vocab_size() > VOCAB_SIZE:
        raise AssertionError(f"unexpected vocab size {sp.vocab_size()}")
    expected = {
        "pad": PAD_ID,
        "unk": UNK_ID,
        "bos": BOS_ID,
        "eos": EOS_ID,
        "user": USER_ID,
        "assistant": ASSISTANT_ID,
    }
    actual = {
        "pad": sp.pad_id(),
        "unk": sp.unk_id(),
        "bos": sp.bos_id(),
        "eos": sp.eos_id(),
        "user": sp.piece_to_id("<|user|>"),
        "assistant": sp.piece_to_id("<|assistant|>"),
    }
    if expected != actual:
        raise AssertionError(f"special IDs mismatch: {actual} != {expected}")

    text = "Xin chào, đây là kiểm thử tokenizer 123."
    ids = encode(text, sp=sp, add_bos=True, add_eos=True)
    if ids[0] != BOS_ID or ids[-1] != EOS_ID:
        raise AssertionError("add_bos/add_eos produced wrong boundaries")
    restored = decode(ids, sp)
    if restored.strip() != text:
        raise AssertionError(f"round-trip mismatch: {restored!r}")
    if encode("<|user|>", sp=sp) != [USER_ID]:
        raise AssertionError("<|user|> is not a single special token")
    if encode("<|assistant|>", sp=sp) != [ASSISTANT_ID]:
        raise AssertionError("<|assistant|> is not a single special token")

    fp1 = tokenizer_fingerprint()
    fp2 = tokenizer_fingerprint()
    if fp1 != fp2 or len(fp1.get("sha256", "")) != 64:
        raise AssertionError("tokenizer fingerprint is not stable")
    return f"vocab={sp.vocab_size()}, sha256={fp1['sha256'][:12]}..."


# ---------------------------------------------------------------------------
# 5. Batching, SFT masking, LR schedule
# ---------------------------------------------------------------------------

def test_batch_sharding():
    data = np.arange(10000, dtype=np.uint16)
    rng0 = np.random.default_rng(99)
    rng1 = np.random.default_rng(99)
    x0, y0 = get_batch(data, torch.device("cpu"), rng0, batch_size=2, rank=0, world=2)
    x1, y1 = get_batch(data, torch.device("cpu"), rng1, batch_size=2, rank=1, world=2)
    assert x0.shape == (2, BLOCK_SIZE) and y0.shape == x0.shape
    assert x1.shape == (2, BLOCK_SIZE)
    if torch.equal(x0, x1):
        raise AssertionError("rank 0 and rank 1 received the same batch")
    for row in range(2):
        if not torch.equal(y0[row, :-1], x0[row, 1:]):
            raise AssertionError("pretrain targets are not shifted inputs")
    return "disjoint rank slices, y=x+1"


def test_sft_batch_mask():
    tokens = [BOS_ID, 10, 11, 12, 13, 14]
    assistant_start = 3
    rows = [(tokens, assistant_start, tuple(tokens[:assistant_start]))]
    x, y = get_sft_batch(
        rows, torch.device("cpu"), np.random.default_rng(0), batch_size=1
    )
    expected_x = tokens[:-1]
    if x[0].tolist() != expected_x:
        raise AssertionError(f"x={x[0].tolist()} != {expected_x}")
    for j, label in enumerate(y[0].tolist()):
        should_mask = j < assistant_start - 1
        if should_mask and label != -100:
            raise AssertionError(f"prompt label at {j} was not masked: {label}")
        if not should_mask and label != tokens[j + 1]:
            raise AssertionError(f"answer label at {j} changed: {label}")
    return "prompt masked, answer preserved"


def test_lr_schedule():
    total = 1000
    values = [get_lr(step, total) for step in range(0, total, 97)]
    if any(v <= 0 for v in values):
        raise AssertionError("learning rate is not positive")
    if get_lr(0, total) >= get_lr(100, total):
        raise AssertionError("warmup does not increase the learning rate")
    if get_lr(total - 1, total) >= get_lr(100, total):
        raise AssertionError("cosine decay does not decrease the learning rate")
    return f"min={min(values):.2e}, max={max(values):.2e}"


# ---------------------------------------------------------------------------
# 6. Checkpoint round-trip and compatibility checks
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    torch.manual_seed(3)
    model = GPTLanguageModel(256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    x = torch.randint(0, 256, (1, 16))
    y = torch.randint(0, 256, (1, 16))
    _, loss, _ = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    data_rng = np.random.default_rng(123)
    data_rng.random()
    eval_rng = np.random.default_rng(456)
    eval_rng.random()
    fp = {"sha256": "a" * 64, "size": 123}
    state = data_rng.bit_generator.state

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "roundtrip.pth")
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            step=7,
            best_val=1.25,
            vocab_size=256,
            mode="pretrain",
            total_steps=10,
            rng=data_rng,
            eval_rng=eval_rng,
            tokenizer_fp=fp,
        )
        assert not any(name.endswith(".tmp") for name in os.listdir(td))

        model2 = GPTLanguageModel(256)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        rng2 = np.random.default_rng(0)
        loaded = load_checkpoint(
            path,
            model2,
            opt2,
            scaler,
            "pretrain",
            tokenizer_fp=fp,
            load_optimizer=True,
            rng=rng2,
        )
        if not loaded["loaded"]:
            raise AssertionError(loaded["reason"])
        if loaded["start_step"] != 7:
            raise AssertionError(f"start_step={loaded['start_step']}, expected 7")
        if not loaded["optimizer_loaded"]:
            raise AssertionError("optimizer state was not restored")
        if rng2.bit_generator.state != state:
            raise AssertionError("data RNG state was not restored")
        for (name, p1), (_, p2) in zip(
            model.state_dict().items(), model2.state_dict().items()
        ):
            if not torch.equal(p1, p2):
                raise AssertionError(f"weight mismatch after load: {name}")

        bad = torch.load(path, map_location="cpu", weights_only=False)
        bad["tokenizer_fingerprint"] = {"sha256": "b" * 64, "size": 123}
        rejected = load_checkpoint(
            path, model2, opt2, scaler, "pretrain", tokenizer_fp=fp, ckpt=bad
        )
        if rejected["loaded"]:
            raise AssertionError("checkpoint with mismatched tokenizer was accepted")
        if "tokenizer" not in rejected["reason"].lower():
            raise AssertionError(f"unclear rejection reason: {rejected['reason']}")

    return "weights/optimizer/RNG restored, fingerprint enforced"


def test_sft_starts_from_pretrain():
    torch.manual_seed(4)
    model = GPTLanguageModel(256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    fp = {"sha256": "c" * 64, "size": 1}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pretrain.pth")
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            step=9,
            best_val=2.0,
            vocab_size=256,
            mode="pretrain",
            total_steps=9,
            tokenizer_fp=fp,
        )
        model2 = GPTLanguageModel(256)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        loaded = load_checkpoint(
            path,
            model2,
            opt2,
            scaler,
            "sft",
            tokenizer_fp=fp,
            load_optimizer=False,
        )
        if not loaded["loaded"] or loaded["ckpt_mode"] != "pretrain":
            raise AssertionError(f"pretrain->SFT load failed: {loaded}")
        if loaded["optimizer_loaded"]:
            raise AssertionError("pretrain optimizer was loaded into a fresh SFT run")
        for (name, p1), (_, p2) in zip(
            model.state_dict().items(), model2.state_dict().items()
        ):
            if not torch.equal(p1, p2):
                raise AssertionError(f"weight mismatch after load: {name}")
    return "weights only, fresh optimizer"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="AI9 audit suite")
    parser.add_argument(
        "--full", action="store_true", help="also run train.py --smoke-test"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AI9 AUDIT")
    print("=" * 70)

    print("\n[1] Source and imports")
    check("compile all source files", test_compile)
    check("import application modules", test_imports)
    check("import distill_generate (optional)", test_distill_import)
    check("CLI --help", test_cli_help)

    print("\n[2] Model")
    check("architecture configuration", test_config)
    check("forward/backward", test_forward_backward)

    print("\n[3] RoPE and KV cache")
    check("RoPE relative invariance", test_rope_relative_invariance)
    check("RoPE position bounds", test_rope_bounds)
    check("KV cache == full forward", test_kv_cache_matches_full)
    check("sliding KV cache window", test_kv_cache_window)
    check("generate cache equivalence", test_generate_cache_equivalence)
    check("generate EOS/penalty/determinism", test_generate_eos_and_penalty)

    print("\n[4] Tokenizer")
    check("tokenizer special IDs and round-trip", test_tokenizer)

    print("\n[5] Batching")
    check("rank-sharded pretrain batch", test_batch_sharding)
    check("SFT prompt masking", test_sft_batch_mask)
    check("learning-rate schedule", test_lr_schedule)

    print("\n[6] Checkpoints")
    check("checkpoint round-trip", test_checkpoint_roundtrip)
    check("SFT from pretrain checkpoint", test_sft_starts_from_pretrain)

    if args.full:
        print("\n[7] Full smoke test")
        proc = subprocess.run(
            [sys.executable, "train.py", "--smoke-test"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if proc.returncode == 0 and "SMOKE TEST PASS" in proc.stdout:
            detail = [
                line for line in proc.stdout.splitlines() if "SMOKE TEST:" in line
            ]
            record("train.py --smoke-test", "PASS", detail[-1] if detail else "")
        else:
            record(
                "train.py --smoke-test",
                "FAIL",
                (proc.stderr or proc.stdout)[-800:],
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status, detail in RESULTS:
        print(f"  {status:4s}  {name}" + (f" | {detail}" if detail else ""))

    passed = sum(1 for _, status, _ in RESULTS if status == "PASS")
    failed = sum(1 for _, status, _ in RESULTS if status == "FAIL")
    skipped = sum(1 for _, status, _ in RESULTS if status == "SKIP")
    print(f"\nTotal: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
