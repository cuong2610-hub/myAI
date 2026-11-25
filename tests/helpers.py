"""Shared helpers for the English test suite (no network, CPU only)."""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from config import TINY_CONFIG, ModelConfig
from model import GPT
from tokenizer import load_tokenizer, train_tokenizer

WORDS = (
    "the quick brown fox jumps over lazy dog machine learning transformer "
    "attention python code computer science model language data training neural "
    "network gradient token embedding sentence english text example value system "
    "user assistant river bank matrix vector probability sampling temperature "
    "validation checkpoint streaming dataset retrieval document paragraph"
).split()


def make_temp_dir(prefix: str = "myai_en_test_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def remove_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def write_corpus(path: Path, lines: int = 3000, seed: int = 0) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(lines):
            n = rng.randint(8, 28)
            f.write(" ".join(rng.choice(WORDS) for _ in range(n)) + ".\n")
    return path


def build_tokenizer(directory: Path, vocab_size: int = 512, seed: int = 0):
    directory.mkdir(parents=True, exist_ok=True)
    corpus = write_corpus(directory / "corpus.txt", lines=2500, seed=seed)
    path = train_tokenizer(
        corpus_path=corpus,
        out_dir=directory / "tok",
        vocab_size=vocab_size,
        force=True,
        num_threads=2,
    )
    return load_tokenizer(path, expected_vocab=vocab_size), path


def build_tiny_model(cfg: ModelConfig | None = None, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg or TINY_CONFIG)
