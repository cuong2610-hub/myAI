import argparse
import hashlib
import html
import os
import re
import unicodedata

INPUT = "data.txt"
OUTPUT = "cleaned_data.txt"

CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_RE = re.compile(r"[ \t]+")
EMPTY_HEADING_RE = re.compile(r"^\s*=+\s*=+\s*$")
VIETNAMESE_RE = re.compile(r"[\u0102\u0103\u0110\u0111\u0128\u0129\u0168\u0169\u01a0\u01a1\u01af\u01b0\u1ea0-\u1ef9]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{9,}")


def clean_line(line):
    line = line.replace("\r", "").replace("\x00", "")
    line = unicodedata.normalize("NFKC", line)
    line = html.unescape(line)
    line = CONTROL_RE.sub("", line)
    line = MULTISPACE_RE.sub(" ", line).strip()
    if not line or EMPTY_HEADING_RE.match(line):
        return ""
    line = re.sub(r"^\s*\[\[Category:.*?\]\]\s*$", "", line, flags=re.I)
    line = re.sub(r"^\s*(See also|References|External links)\s*$", "", line, flags=re.I)
    return line.strip()


def _acceptable_document(lines):
    text = " ".join(lines)
    if len(text) < 80 or len(text.split()) < 12:
        return False
    if text.count("\ufffd") / len(text) > 0.01 or REPEATED_CHAR_RE.search(text):
        return False
    return sum(ch.isalnum() for ch in text) / len(text) >= 0.5


def preprocess(src=INPUT, dst=OUTPUT):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    written = skipped = duplicate = documents = 0
    vietnamese_chars = total_letters = 0
    seen = set()

    def emit(lines, fout):
        nonlocal written, skipped, duplicate, documents, vietnamese_chars, total_letters
        if not lines:
            return
        documents += 1
        text = " ".join(lines)
        if not _acceptable_document(lines):
            skipped += 1
            return
        digest = hashlib.sha256(text.casefold().encode("utf-8")).digest()
        if digest in seen:
            duplicate += 1
            return
        seen.add(digest)
        fout.write(text + "\n")
        written += 1
        vietnamese_chars += len(VIETNAMESE_RE.findall(text))
        total_letters += sum(ch.isalpha() for ch in text)

    with open(src, "r", encoding="utf-8", errors="replace", buffering=8 * 1024 * 1024) as fin, \
         open(dst, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as fout:
        document = []
        for line_no, line in enumerate(fin, 1):
            cleaned = clean_line(line)
            if not cleaned:
                emit(document, fout)
                document = []
            elif len(cleaned) >= 2:
                document.append(cleaned)
            if line_no % 1_000_000 == 0:
                print(f"{line_no:,} dòng | giữ {written:,} tài liệu | bỏ {skipped:,} | trùng {duplicate:,}")
        emit(document, fout)

    ratio = 100 * vietnamese_chars / max(1, total_letters)
    print(f"Hoàn tất: giữ {written:,}/{documents:,} tài liệu; bỏ {skipped:,}; trùng {duplicate:,}.")
    print(f"Tỷ lệ ký tự tiếng Việt: {ratio:.2f}%")
    print(f"Output: {dst} | {os.path.getsize(dst) / (1024**3):.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()
    preprocess(args.input, args.output)
