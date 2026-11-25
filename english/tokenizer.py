"""English BPE tokenizer (SentencePiece) with fixed special tokens.

The tokenizer is deliberately independent of ``tokenizer.py`` at the repository
root (Vietnamese).  Files are written under ``tokenizer_en/`` by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sentencepiece as spm

from .config import (
    ASSISTANT_ID,
    BOS_ID,
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_SPLIT,
    DEFAULT_TOKENIZER_DIR,
    EOS_ID,
    PAD_ID,
    SYSTEM_ID,
    UNK_ID,
    USER_ID,
)

DEFAULT_VOCAB_SIZE = 32768
SPECIAL_PIECES = ("<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "<|system|>")
SPECIAL_IDS = {
    "pad": PAD_ID,
    "unk": UNK_ID,
    "bos": BOS_ID,
    "eos": EOS_ID,
    "user": USER_ID,
    "assistant": ASSISTANT_ID,
    "system": SYSTEM_ID,
}
TOKENIZER_FORMAT = 1


class TokenizerError(RuntimeError):
    pass


def _sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_fingerprint(model_path: str | Path) -> Dict[str, Any]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {p}")
    return {
        "sha256": _sha256_file(p),
        "size": int(p.stat().st_size),
        "path": str(p),
    }


def _atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _atomic_replace(src: str | Path, dst: str | Path) -> None:
    os.replace(src, dst)


def _validate_specials(sp: spm.SentencePieceProcessor, expected_vocab: Optional[int] = None) -> None:
    actual = int(sp.vocab_size())
    if expected_vocab is not None and actual != int(expected_vocab):
        raise TokenizerError(
            f"Tokenizer vocab={actual}, expected {expected_vocab}. "
            "Use the tokenizer that belongs to this checkpoint."
        )
    got = {
        "pad": int(sp.pad_id()),
        "unk": int(sp.unk_id()),
        "bos": int(sp.bos_id()),
        "eos": int(sp.eos_id()),
        "user": int(sp.piece_to_id("<|user|>")),
        "assistant": int(sp.piece_to_id("<|assistant|>")),
        "system": int(sp.piece_to_id("<|system|>")),
    }
    if got != SPECIAL_IDS:
        raise TokenizerError(
            f"Special token ids do not match the English contract: {got} != {SPECIAL_IDS}"
        )


def load_tokenizer(
    model_path: str | Path | None = None,
    expected_vocab: Optional[int] = None,
) -> spm.SentencePieceProcessor:
    path = Path(model_path) if model_path is not None else DEFAULT_TOKENIZER_DIR / "tokenizer.model"
    if not path.exists():
        raise FileNotFoundError(
            f"English tokenizer not found at {path}. Train it first with:\n"
            "  python -m english.tokenizer --train --sample-mb 200"
        )
    try:
        sp = spm.SentencePieceProcessor(model_file=str(path))
    except Exception as exc:
        raise TokenizerError(f"Could not load tokenizer {path}: {exc}") from exc
    _validate_specials(sp, expected_vocab)
    return sp


def encode(
    text: str,
    sp: spm.SentencePieceProcessor | None = None,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    sp = sp or load_tokenizer()
    ids = [int(i) for i in sp.encode(str(text).replace("\r", ""), out_type=int)]
    if add_bos:
        ids.insert(0, BOS_ID)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode(
    ids: Sequence[int],
    sp: spm.SentencePieceProcessor | None = None,
    skip_special: bool = True,
) -> str:
    sp = sp or load_tokenizer()
    special = set(SPECIAL_IDS.values())
    clean = [int(i) for i in ids if not (skip_special and int(i) in special)]
    return sp.decode(clean)


def vocab_size(sp: spm.SentencePieceProcessor | None = None) -> int:
    return int((sp or load_tokenizer()).vocab_size())


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------


def format_chat_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Prompt text ending with an assistant header, matching build_chat_ids()."""
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role {role!r}")
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def build_chat_ids(
    messages: Sequence[Dict[str, str]],
    sp: spm.SentencePieceProcessor,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Tuple[List[int], List[int]]:
    """Build input ids and labels for SFT.

    Headers and non-assistant content are masked with -100.  Assistant content
    and the final EOS are trained.
    """
    ids: List[int] = []
    labels: List[int] = []
    if add_bos:
        ids.append(BOS_ID)
        labels.append(-100)

    for i, msg in enumerate(messages):
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role {role!r}")
        header_ids = encode(f"<|{role}|>\n", sp=sp)
        content_ids = encode(content, sp=sp)
        ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))
        ids.extend(content_ids)
        labels.extend(content_ids if role == "assistant" else [-100] * len(content_ids))
        if i < len(messages) - 1:
            nl = encode("\n", sp=sp)
            ids.extend(nl)
            labels.extend([-100] * len(nl))

    if add_eos:
        ids.append(EOS_ID)
        labels.append(EOS_ID)
    return ids, labels


def normalize_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """Accept {'messages': [...]} or {'user': ..., 'assistant': ...}."""
    if "messages" in obj:
        raw = obj["messages"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("'messages' must be a non-empty list")
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each message must be an object")
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"invalid message role {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content must be a non-empty string")
            out.append({"role": role, "content": content.strip()})
        system = obj.get("system")
        if (
            isinstance(system, str)
            and system.strip()
            and not any(m["role"] == "system" for m in out)
        ):
            out.insert(0, {"role": "system", "content": system.strip()})
        return out

    user = obj.get("user", "")
    assistant = obj.get("assistant", "")
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("'user' and 'assistant' must be strings")
    if not user.strip() or not assistant.strip():
        raise ValueError("'user' and 'assistant' must be non-empty")
    out = []
    system = obj.get("system")
    if isinstance(system, str) and system.strip():
        out.append({"role": "system", "content": system.strip()})
    out.append({"role": "user", "content": user.strip()})
    out.append({"role": "assistant", "content": assistant.strip()})
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _sample_corpus(
    output_path: Path,
    sample_mb: int,
    seed: int,
    dataset: str,
    dataset_config: str,
    split: str,
    val_fraction: float,
    max_chars_per_line: int = 2048,
) -> Dict[str, Any]:
    """Stream a bounded English sample for tokenizer training (no full download)."""
    from .data import DocumentStream, RetryConfig  # lazy: avoids import cycle

    target = max(1, int(sample_mb)) * 1024 * 1024
    stream = DocumentStream(
        source=dataset,
        config=dataset_config,
        split=split,
        world_size=1,
        rank=0,
        val_fraction=val_fraction,
        mode="train",
        shuffle_buffer=10000,
        seed=seed,
        min_chars=64,
        max_chars=200_000,
        retry=RetryConfig(max_retries=6, base_delay=1.0, max_delay=60.0),
    )
    written = 0
    docs = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        while written < target:
            try:
                doc = stream.next_doc()
            except StopIteration:
                break
            text = " ".join(str(doc["text"]).split())
            if not text:
                continue
            docs += 1
            for start in range(0, len(text), max_chars_per_line):
                piece = text[start:start + max_chars_per_line]
                if len(piece) < 8:
                    continue
                f.write(piece + "\n")
                written += len(piece) + 1
                if written >= target:
                    break
    os.replace(tmp, output_path)
    if written < 1_000_000:
        raise TokenizerError(
            f"Tokenizer corpus is too small ({written} bytes). "
            "Use --sample-mb larger or provide a local --corpus file."
        )
    return {"bytes": written, "documents": docs, "sample_mb": sample_mb, "seed": seed}


def train_tokenizer(
    corpus_path: str | Path | None = None,
    out_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    force: bool = False,
    sample_mb: int = 200,
    seed: int = 1337,
    dataset: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
    val_fraction: float = 0.01,
    num_threads: int | None = None,
    keep_corpus: bool = False,
) -> Path:
    """Train the English SentencePiece BPE tokenizer.

    If ``corpus_path`` is given it is used directly (useful for tests and
    offline reproducibility).  Otherwise a bounded streamed sample of English
    text is written under ``out_dir/corpus.txt``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "tokenizer.model"
    vocab_path = out / "tokenizer.vocab"
    meta_path = out / "meta.json"

    if model_path.exists() and not force:
        try:
            sp = spm.SentencePieceProcessor(model_file=str(model_path))
            _validate_specials(sp, vocab_size)
            return model_path
        except Exception:
            pass  # incompatible -> rebuild

    if vocab_size < 320:
        raise ValueError("vocab_size must be >= 320 (256 byte fallback + specials)")

    generated_corpus = corpus_path is None
    corpus = Path(corpus_path) if corpus_path is not None else out / "corpus.txt"
    sample_info: Dict[str, Any] = {}
    if generated_corpus:
        if corpus.exists():
            corpus.unlink()
        sample_info = _sample_corpus(
            corpus,
            sample_mb=sample_mb,
            seed=seed,
            dataset=dataset,
            dataset_config=dataset_config,
            split=split,
            val_fraction=val_fraction,
        )
    elif not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}")

    tmp_prefix = out / "tokenizer.tmp"
    for p in (Path(str(tmp_prefix) + ".model"), Path(str(tmp_prefix) + ".vocab")):
        if p.exists():
            p.unlink()

    threads = int(num_threads) if num_threads else max(2, os.cpu_count() or 2)
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(tmp_prefix),
        model_type="bpe",
        vocab_size=int(vocab_size),
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_unicode_script=True,
        split_by_number=True,
        split_digits=False,
        byte_fallback=True,
        hard_vocab_limit=True,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        user_defined_symbols="<|user|>,<|assistant|>,<|system|>",
        num_threads=threads,
        minloglevel=2,
    )

    tmp_model = Path(str(tmp_prefix) + ".model")
    tmp_vocab = Path(str(tmp_prefix) + ".vocab")
    sp = spm.SentencePieceProcessor(model_file=str(tmp_model))
    _validate_specials(sp, vocab_size)

    # Keep the previous tokenizer until the new one is fully validated.
    _atomic_replace(tmp_model, model_path)
    _atomic_replace(tmp_vocab, vocab_path)

    fp = tokenizer_fingerprint(model_path)
    meta = {
        "format": TOKENIZER_FORMAT,
        "model_type": "bpe",
        "vocab_size": int(sp.vocab_size()),
        "specials": SPECIAL_IDS,
        "fingerprint": fp,
        "corpus": {
            "path": str(corpus),
            "bytes": int(corpus.stat().st_size),
            "generated": generated_corpus,
            **sample_info,
        },
        "sentencepiece": {
            "normalization": "identity",
            "add_dummy_prefix": False,
            "byte_fallback": True,
            "split_by_number": True,
        },
    }
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))

    for p in (Path(str(tmp_prefix) + ".model"), Path(str(tmp_prefix) + ".vocab")):
        if p.exists():
            p.unlink()
    if generated_corpus and not keep_corpus:
        try:
            corpus.unlink()
        except OSError:
            pass
    return model_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train or inspect the English tokenizer.")
    ap.add_argument("--train", action="store_true", help="Train a new tokenizer.")
    ap.add_argument("--corpus", default=None, help="Local text corpus (one chunk per line).")
    ap.add_argument("--out-dir", default=str(DEFAULT_TOKENIZER_DIR))
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--sample-mb", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-corpus", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--text", default="Hello world. This is the English tokenizer.")
    args = ap.parse_args(argv)

    if args.inspect:
        sp = load_tokenizer(Path(args.out_dir) / "tokenizer.model")
        print(f"vocab={sp.vocab_size()} path={Path(args.out_dir) / 'tokenizer.model'}")
        print(f"specials={SPECIAL_IDS}")
        ids = encode(args.text, sp=sp, add_bos=True, add_eos=True)
        print(f"ids={ids}")
        print(f"decoded={decode(ids, sp=sp)!r}")
        print(f"fingerprint={tokenizer_fingerprint(Path(args.out_dir) / 'tokenizer.model')}")
        return 0

    if args.train:
        path = train_tokenizer(
            corpus_path=args.corpus,
            out_dir=args.out_dir,
            vocab_size=args.vocab_size,
            force=args.force,
            sample_mb=args.sample_mb,
            seed=args.seed,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            val_fraction=args.val_fraction,
            keep_corpus=args.keep_corpus,
        )
        print(f"Tokenizer ready: {path}")
        print(f"Fingerprint: {tokenizer_fingerprint(path)}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
