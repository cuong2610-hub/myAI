import argparse
import html
import os
import re
import unicodedata

INPUT = "data.txt"
OUTPUT = "cleaned_data.txt"

CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_RE = re.compile(r"[ \t]+")
EMPTY_HEADING_RE = re.compile(r"^\s*=+\s*=+\s*$")


def clean_line(line):
    line = line.replace("\r", "").replace("\x00", "")
    line = unicodedata.normalize("NFKC", line)
    line = html.unescape(line)
    line = CONTROL_RE.sub("", line)
    line = MULTISPACE_RE.sub(" ", line).strip()
    if not line or EMPTY_HEADING_RE.match(line):
        return ""
    # Strip common Wikipedia markup without trying to aggressively rewrite content.
    line = re.sub(r"^\s*\[\[Category:.*?\]\]\s*$", "", line, flags=re.I)
    line = re.sub(r"^\s*(See also|References|External links)\s*$", "", line, flags=re.I)
    return line.strip()


def preprocess(src=INPUT, dst=OUTPUT):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    written = 0
    skipped = 0
    with open(src, "r", encoding="utf-8", errors="replace", buffering=8 * 1024 * 1024) as fin, \
         open(dst, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as fout:
        for line_no, line in enumerate(fin, 1):
            s = clean_line(line)
            if len(s) < 2:
                skipped += 1
                continue
            fout.write(s + "\n")
            written += 1
            if line_no % 1_000_000 == 0:
                print(f"{line_no:,} dòng | giữ {written:,} | bỏ {skipped:,}")
    print(f"Hoàn tất: {written:,} dòng được giữ, {skipped:,} dòng bỏ.")
    print(f"Output: {dst} | {os.path.getsize(dst) / (1024**3):.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()
    preprocess(args.input, args.output)
