import argparse
import os
import sys

import torch

from model import GPTLanguageModel, BLOCK_SIZE, ARCH_VERSION, model_config
from tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    USER_ID,
    ASSISTANT_ID,
    load_tokenizer,
    decode,
    encode,
    tokenizer_fingerprint,
)
from agent import retrieve_context
from rag import clear_memory

CHECKPOINTS = [
    "instruct_best.pth",
    "instruct_last.pth",
    "pretrain_best.pth",
    "pretrain_last.pth",
]


def select_checkpoint(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Checkpoint không tồn tại: {explicit}")
        return explicit
    for path in CHECKPOINTS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Không tìm thấy checkpoint. Hãy chạy pretrain + SFT trước, hoặc truyền "
        "--checkpoint /path/to/model.pth."
    )


def infer_mode(ckpt, path):
    mode = ckpt.get("mode")
    if mode:
        return mode
    base = os.path.basename(path).lower()
    return "sft" if base.startswith("instruct") else "pretrain"


def load_model(checkpoint_path, device):
    sp = load_tokenizer()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint không đúng định dạng: {checkpoint_path}")

    if ckpt.get("arch_version") is not None and ckpt.get("arch_version") != ARCH_VERSION:
        raise ValueError(
            f"Checkpoint arch_version={ckpt.get('arch_version')} không khớp "
            f"model hiện tại (arch {ARCH_VERSION})."
        )
    if ckpt.get("vocab_size") != sp.vocab_size():
        raise ValueError(
            f"Checkpoint vocab={ckpt.get('vocab_size')} không khớp "
            f"tokenizer vocab={sp.vocab_size()}."
        )
    if ckpt.get("block_size") not in (None, BLOCK_SIZE):
        raise ValueError(
            f"Checkpoint block_size={ckpt.get('block_size')} không khớp {BLOCK_SIZE}."
        )

    expected = {
        "n_embd": model_config()["n_embd"],
        "n_head": model_config()["n_head"],
        "n_layer": model_config()["n_layer"],
        "kv_latent": model_config()["kv_latent"],
        "n_experts": model_config()["n_experts"],
        "moe_top_k": model_config()["moe_top_k"],
        "moe_hidden": model_config()["moe_hidden"],
    }
    for key, value in expected.items():
        if key in ckpt and ckpt[key] != value:
            raise ValueError(
                f"Checkpoint {key}={ckpt[key]} không khớp cấu hình hiện tại ({value})."
            )
    saved_cfg = ckpt.get("config")
    if isinstance(saved_cfg, dict):
        for key, value in model_config().items():
            if key in saved_cfg and saved_cfg[key] != value:
                raise ValueError(
                    f"Checkpoint config {key}={saved_cfg[key]} không khớp ({value})."
                )

    saved_fp = ckpt.get("tokenizer_fingerprint")
    if saved_fp:
        current_fp = tokenizer_fingerprint()
        if saved_fp != current_fp:
            raise ValueError(
                "Tokenizer hiện tại không khớp tokenizer đã dùng để train checkpoint. "
                "Hãy dùng đúng tokenizer.model hoặc train lại từ đầu."
            )
    else:
        print("[WARN] Checkpoint cũ không có tokenizer fingerprint; "
              "không thể kiểm tra token ID.")

    model = GPTLanguageModel(sp.vocab_size())
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    mode = infer_mode(ckpt, checkpoint_path)
    info = {
        "mode": mode,
        "step": ckpt.get("step"),
        "best_val": ckpt.get("best_val"),
        "arch_version": ckpt.get("arch_version"),
    }
    return model, sp, info


def _current_block(user_text, context_text, mode):
    if mode == "sft":
        current = user_text
        if context_text:
            current = (
                "Dưới đây là thông tin tham khảo từ nguồn bên ngoài. Chỉ dùng nó khi phù hợp "
                "với câu hỏi. Không bịa thêm dữ kiện không có trong nguồn.\n\n"
                + context_text + "\n\nCâu hỏi của người dùng:\n" + user_text
            )
        return f"<|user|>\n{current}\n<|assistant|>\n"
    return user_text if not context_text else context_text + "\n\n" + user_text


def build_prompt(user_text, history, mode, context_text, sp):
    """Build a prompt that always keeps the current question.

    Old conversation turns are dropped first; if the current turn itself is
    still too long, the oldest tokens are trimmed but BOS is preserved.
    """
    current = _current_block(user_text, context_text, mode)
    blocks = []
    if mode == "sft":
        for u, a in history:
            if a:
                blocks.append(f"<|user|>\n{u}\n<|assistant|>\n{a}")

    while True:
        text = "\n".join(blocks + [current]) if mode == "sft" else current
        ids = encode(text, sp=sp, add_bos=True)
        if len(ids) <= BLOCK_SIZE:
            return ids
        if blocks:
            blocks.pop(0)
            continue
        # The current turn alone is too long. Keep the end (the actual
        # question) and restore exactly one BOS.
        return [BOS_ID] + ids[-(BLOCK_SIZE - 1):]


def _clean_answer(text):
    for marker in (
        "<|user|>",
        "<|assistant|>",
        "Nguồn 1",
        "Nguồn 2",
        "</s>",
        "<pad>",
    ):
        text = text.replace(marker, "")
    return text.strip()


def _configure_stdio():
    """Avoid UnicodeEncodeError when stdout is redirected on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    _configure_stdio()
    ap = argparse.ArgumentParser(description="Chat với checkpoint AI9.")
    ap.add_argument("--checkpoint", default="", help="Đường dẫn checkpoint cụ thể.")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Mặc định 0.58 khi có ngữ cảnh web/RAG, 0.68 khi không.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--repetition-penalty", type=float, default=1.10)
    ap.add_argument("--min-new-tokens", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true",
                    help="Tắt KV cache (chậm; dùng để kiểm tra tính nhất quán).")
    ap.add_argument("--no-web", action="store_true", help="Tắt web agent lúc khởi động.")
    ap.add_argument("--no-rag", action="store_true", help="Tắt truy xuất bộ nhớ RAG.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = select_checkpoint(args.checkpoint or None)
    model, sp, info = load_model(checkpoint, device)

    mode = info["mode"]
    print(f"Model đã load: {checkpoint}")
    print(
        f"Mode: {mode} | step: {info['step']} | best_val: {info['best_val']} | "
        f"arch: {info['arch_version']}"
    )
    print(f"Thiết bị: {device} | KV cache: {'off' if args.no_cache else 'on'}")
    if mode != "sft":
        print("Lưu ý: đây là base model; nên SFT trước khi chat.")
    print("Internet agent: " + ("OFF" if args.no_web else "AUTO"))
    print("RAG memory: " + ("OFF" if args.no_rag else "ON"))
    print("Lệnh: /web on | /web off | /web <câu hỏi> | /clear | /forget | exit\n")

    history = []
    web_enabled = not args.no_web

    while True:
        try:
            prompt = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() == "exit":
            break
        if prompt.lower() == "/clear":
            history.clear()
            print("Đã xóa lịch sử.\n")
            continue
        if prompt.lower() == "/forget":
            clear_memory()
            print("Đã xóa bộ nhớ web/RAG.\n")
            continue
        if prompt.lower() == "/web on":
            web_enabled = True
            print("Internet agent: ON\n")
            continue
        if prompt.lower() == "/web off":
            web_enabled = False
            print("Internet agent: OFF\n")
            continue

        force_web = False
        if prompt.startswith("/web "):
            prompt = prompt[5:].strip()
            force_web = True
        if not prompt:
            continue

        try:
            if args.no_rag and not web_enabled and not force_web:
                route = None
                context_text = ""
            else:
                route = retrieve_context(
                    prompt,
                    force_web=force_web,
                    use_memory=not args.no_rag,
                    allow_web=web_enabled,
                )
                context_text = route.context
        except Exception as exc:
            print(f"[WARN] Không lấy được ngữ cảnh web/RAG: {exc}")
            route = None
            context_text = ""

        ids = build_prompt(prompt, history, mode, context_text, sp)
        if len(ids) == 0:
            print("AI: (prompt rỗng)\n")
            continue
        context = torch.tensor([ids], dtype=torch.long, device=device)

        temperature = (
            args.temperature
            if args.temperature is not None
            else (0.58 if context_text else 0.68)
        )

        with torch.inference_mode():
            out = model.generate(
                context,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_id=EOS_ID,
                min_new_tokens=args.min_new_tokens,
                pad_id=PAD_ID,
                use_cache=not args.no_cache,
            )

        if out.dim() != 2 or out.size(0) != 1:
            raise RuntimeError(f"generate trả về shape bất thường: {tuple(out.shape)}")
        new_ids = out[0].tolist()[len(ids):]
        text = _clean_answer(
            decode(
                new_ids,
                sp,
                skip_ids=(PAD_ID, UNK_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID),
            )
        )
        if text:
            history.append((prompt, text))

        if route is not None:
            if route.web_docs:
                print(f"[Web] đã đọc {len(route.web_docs)} nguồn.")
            elif route.memory_docs:
                print(f"[RAG] dùng {len(route.memory_docs)} tài liệu đã lưu.")

        print(f"AI: {text if text else '(không sinh được nội dung)'}\n")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        sys.exit(1)
