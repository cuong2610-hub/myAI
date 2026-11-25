"""Guards that keep the repository English-only and self-contained."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from english import config as en_config

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent

NON_ENGLISH_CHARS = re.compile(
    "["
    "ăâđêôơư"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
    "òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
    "]"
)

# Legacy root modules that the English package must never import.
LEGACY_MODULES = {
    "agent",
    "chat",
    "checkpoint",
    "config",
    "data",
    "distill_generate",
    "generate",
    "model",
    "preprocess",
    "rag",
    "sft",
    "tokenizer",
    "train",
    "utils",
    "web_search",
}

# Root files that belonged only to the removed edition.
LEGACY_FILES = (
    "agent.py",
    "bench_kv_cache.py",
    "chat.py",
    "data.txt",
    "data_meta.json",
    "distill_generate.py",
    "distill_seed_prompts.jsonl",
    "instruction_data.jsonl",
    "model.py",
    "preprocess.py",
    "rag.py",
    "README_EN.md",
    "requirements_distill.txt",
    "requirements_en.txt",
    "state.db",
    "test_audit.py",
    "test_prompts.py",
    "tokenizer.model",
    "tokenizer.py",
    "tokenizer.vocab",
    "train.py",
    "train_tokens.bin",
    "val_tokens.bin",
    "web_search.py",
)


class EnglishOnlyTests(unittest.TestCase):
    def test_english_sources_have_no_non_english_diacritics(self):
        offenders = []
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                if NON_ENGLISH_CHARS.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()[:80]}")
        self.assertEqual(offenders, [], "Non-English text found in English sources")

    def test_english_package_does_not_import_legacy_modules(self):
        offenders = []
        for path in sorted(PACKAGE_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative import inside the package
                        continue
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    root = name.split(".", 1)[0]
                    if name == "english" or root not in LEGACY_MODULES:
                        continue
                    offenders.append(f"{path.name}: import {name}")
        self.assertEqual(offenders, [], "English modules import legacy modules")

    def test_artifact_paths_are_english_specific(self):
        dirs = [
            en_config.DEFAULT_TOKENIZER_DIR,
            en_config.DEFAULT_CHECKPOINT_DIR,
            en_config.DEFAULT_DATA_DIR,
            en_config.DEFAULT_RUNS_DIR,
            en_config.DEFAULT_MEMORY_DIR,
        ]
        for directory in dirs:
            self.assertTrue(
                directory.name.endswith("_en"),
                f"{directory} is not clearly English-specific",
            )
            self.assertEqual(directory.parent, REPO_ROOT)
        self.assertEqual(en_config.DEFAULT_DATASET, "HuggingFaceFW/fineweb-edu")

    def test_legacy_edition_is_removed(self):
        for name in LEGACY_FILES:
            self.assertFalse((REPO_ROOT / name).exists(), f"legacy file still present: {name}")
        self.assertFalse((REPO_ROOT / "memory").exists(), "legacy memory store still present")
        self.assertFalse((REPO_ROOT / "_gen_new").exists(), "legacy generated data still present")

        required = (
            "english/train.py",
            "english/sft.py",
            "english/chat.py",
            "english/tokenizer.py",
            "data/sft_sample_en.jsonl",
            "README.md",
            "KAGGLE_EN.md",
            "kaggle_en.ipynb",
            "requirements.txt",
        )
        for name in required:
            self.assertTrue((REPO_ROOT / name).exists(), f"required English file missing: {name}")


if __name__ == "__main__":
    unittest.main()
