import os
import random
from pathlib import Path

import sentencepiece as spm

VOCAB_SIZE = 16384
MODEL_PATH = "tokenizer.model"
VOCAB_PATH = "tokenizer.vocab"
META_PATH = "tokenizer_meta.json"
CORPUS_PATH = "tokenizer_corpus.txt"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
TOKENIZER_SEED = 1337
TOKENIZER_FORMAT = 2


def _clean_line(s):
    return s.replace("\r", "").replace("\x00", "").replace("\t", " ").strip()


def _file_fingerprint(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def tokenizer_fingerprint(model_path=MODEL_PATH):
    """Content hash of the tokenizer used by a checkpoint.

    mtime is intentionally not used: copying a tokenizer.model between Kaggle
    sessions changes mtime but not the token IDs.
    """
    import hashlib

    if not os.path.exists(model_path):
        return None

    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": os.path.getsize(model_path)}


def _wanted_meta(input_path, vocab_size):
    return {
        "format": TOKENIZER_FORMAT,
        "source": _file_fingerprint(input_path),
        "vocab_size": int(vocab_size),
    }


def _read_meta():
    try:
        import json

        return json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(obj):
    import json

    tmp_path = META_PATH + ".tmp"
    Path(tmp_path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, META_PATH)


def _write_sampled_corpus(input_path, output_path, target_chars=100_000_000, max_piece_chars=4096):
    total_lines = 0
    total_chars = 0
    with open(input_path, "r", encoding="utf-8", errors="replace", buffering=8 * 1024 * 1024) as src:
        for line in src:
            s = _clean_line(line)
            if len(s) >= 2:
                total_lines += 1
                total_chars += len(s)
    if total_lines == 0:
        raise RuntimeError("Dataset rỗng sau preprocessing.")

    avg_chars = max(1.0, total_chars / total_lines)
    wanted_lines = max(1, min(total_lines, int(target_chars / avg_chars)))
    stride = max(1, total_lines // wanted_lines)
    rng = random.Random(TOKENIZER_SEED)
    offset = rng.randrange(stride)

    written = 0
    valid_line = 0
    with open(input_path, "r", encoding="utf-8", errors="replace", buffering=8 * 1024 * 1024) as src, \
         open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as dst:
        for line in src:
            s = _clean_line(line)
            if len(s) < 2:
                continue
            if valid_line % stride != offset:
                valid_line += 1
                continue
            valid_line += 1
            for start in range(0, len(s), max_piece_chars):
                piece = s[start:start + max_piece_chars]
                if len(piece) >= 2:
                    dst.write(piece + "\n")
                    written += len(piece) + 1
                    if written >= target_chars:
                        return

    if written == 0:
        raise RuntimeError("Không tạo được tokenizer corpus.")


def _validate_special_ids(sp, vocab_size):
    actual = sp.vocab_size()
    if actual != int(vocab_size):
        raise ValueError(
            f"Tokenizer vocab={actual}, cần {vocab_size}. "
            "Hãy huấn luyện lại tokenizer với --rebuild-data."
        )
    expected = {
        "pad": (sp.pad_id(), PAD_ID),
        "unk": (sp.unk_id(), UNK_ID),
        "bos": (sp.bos_id(), BOS_ID),
        "eos": (sp.eos_id(), EOS_ID),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise ValueError(
                f"Tokenizer có {name}_id={got}, cần {want}. "
                "File tokenizer không tương thích với model.py/train.py."
            )
    user_id = sp.piece_to_id("<|user|>")
    assistant_id = sp.piece_to_id("<|assistant|>")
    if user_id == sp.unk_id() or assistant_id == sp.unk_id():
        raise ValueError(
            "Tokenizer thiếu user_defined_symbols <|user|>/<|assistant|>. "
            "Hãy huấn luyện lại tokenizer với --rebuild-data."
        )
    return sp


def train_tokenizer(input_path="cleaned_data.txt", vocab_size=VOCAB_SIZE, force=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy {input_path}.")

    wanted_meta = _wanted_meta(input_path, vocab_size)

    if os.path.exists(MODEL_PATH) and not force:
        try:
            sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
            if sp.vocab_size() == vocab_size:
                meta = _read_meta()
                if meta is None:
                    # Backwards compatibility: an existing tokenizer without
                    # provenance is trusted once, then tracked from now on.
                    _write_meta(wanted_meta)
                    return
                if meta == wanted_meta:
                    return
                print("Corpus đã thay đổi so với tokenizer hiện có; huấn luyện lại tokenizer.")
            else:
                print(
                    f"Tokenizer hiện có vocab={sp.vocab_size()}, cần {vocab_size}; "
                    "huấn luyện lại."
                )
        except Exception as exc:
            print(f"Không đọc được tokenizer hiện có ({exc}); huấn luyện lại.")

    # Keep an existing tokenizer/model/meta until the new one is fully trained
    # and validated. A failed rebuild must not destroy a working tokenizer.
    for p in (CORPUS_PATH, "tokenizer.tmp.model", "tokenizer.tmp.vocab"):
        if os.path.exists(p):
            os.remove(p)

    print("Tạo corpus mẫu trải đều trên toàn bộ dataset...")
    _write_sampled_corpus(input_path, CORPUS_PATH)
    tmp_prefix = "tokenizer.tmp"
    try:
        spm.SentencePieceTrainer.Train(
            input=CORPUS_PATH,
            model_prefix=tmp_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
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
            user_defined_symbols="<|user|>,<|assistant|>",
            num_threads=max(2, os.cpu_count() or 2),
        )

        tmp_model = tmp_prefix + ".model"
        tmp_vocab = tmp_prefix + ".vocab"
        sp = spm.SentencePieceProcessor(model_file=tmp_model)
        _validate_special_ids(sp, vocab_size)

        os.replace(tmp_model, MODEL_PATH)
        os.replace(tmp_vocab, VOCAB_PATH)
        wanted_meta["fingerprint"] = tokenizer_fingerprint(MODEL_PATH)
        _write_meta(wanted_meta)
    finally:
        for p in (CORPUS_PATH, tmp_prefix + ".model", tmp_prefix + ".vocab"):
            if os.path.exists(p):
                os.remove(p)


def load_tokenizer(model_path=MODEL_PATH, expected_vocab=VOCAB_SIZE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy {model_path}. Hãy chạy "
            "`python train.py --mode pretrain --rebuild-data` trước để tạo tokenizer."
        )
    try:
        sp = spm.SentencePieceProcessor(model_file=model_path)
    except Exception as exc:
        raise ValueError(
            f"Không đọc được tokenizer {model_path}: {exc}. "
            "File có thể bị hỏng; hãy tạo lại với --rebuild-data."
        ) from exc
    return _validate_special_ids(sp, expected_vocab)


def encode(text, sp=None, add_bos=False, add_eos=False):
    sp = sp or load_tokenizer()
    ids = sp.encode(str(text).replace("\r", ""), out_type=int)
    if add_bos:
        ids.insert(0, BOS_ID)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode(ids, sp=None, skip_ids=(PAD_ID, UNK_ID, BOS_ID, EOS_ID)):
    sp = sp or load_tokenizer()
    skip = set(skip_ids or ())
    clean = [int(i) for i in ids if int(i) not in skip]
    return sp.decode(clean)


def vocab_size(sp=None):
    return (sp or load_tokenizer()).vocab_size()
