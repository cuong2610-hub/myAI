"""Guards that keep the repository flat, English-only and self-contained.

The project uses a flat layout: every English module lives at the repository
root and the tests live in ``tests/``.  These checks fail if a package
directory is reintroduced, if a source file starts importing a removed legacy
module, or if the English artifact directories stop pointing at the repository
root.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import config as en_config

REPO_ROOT = Path(__file__).resolve().parents[1]

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

# Modules that were removed with the legacy edition.  The flat English project
# must never import them (and must not use relative imports either).
REMOVED_MODULES = {
    "bench_kv_cache",
    "distill_generate",
    "english",
    "preprocess",
    "test_audit",
    "test_prompts",
}

# Root files that belonged only to the removed edition.
REMOVED_FILES = (
    "README_EN.md",
    "bench_kv_cache.py",
    "cleaned_data.txt",
    "data.txt",
    "data_meta.json",
    "distill_generate.py",
    "distill_seed_prompts.jsonl",
    "preprocess.py",
    "requirements_distill.txt",
    "requirements_en.txt",
    "state.db",
    "test_audit.py",
    "test_prompts.py",
    "tokenizer.model",
    "tokenizer.vocab",
    "train_tokens.bin",
    "val_tokens.bin",
)

REQUIRED_FILES = (
    "__init__.py",
    "agent.py",
    "chat.py",
    "checkpoint.py",
    "config.py",
    "data.py",
    "generate.py",
    "model.py",
    "rag.py",
    "sft.py",
    "tokenizer.py",
    "train.py",
    "utils.py",
    "web_search.py",
    "tests/helpers.py",
    "data/sft_sample_en.jsonl",
    "README.md",
    "KAGGLE_EN.md",
    "kaggle_en.ipynb",
    "requirements.txt",
)


def _python_sources():
    sources = list(REPO_ROOT.glob("*.py")) + list((REPO_ROOT / "tests").glob("*.py"))
    return [p for p in sorted(sources) if "__pycache__" not in p.parts]


class FlatLayoutTests(unittest.TestCase):
    def test_english_sources_have_no_non_english_diacritics(self):
        offenders = []
        for path in _python_sources():
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                if NON_ENGLISH_CHARS.search(line):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()[:80]}"
                    )
        self.assertEqual(offenders, [], "Non-English text found in English sources")

    def test_sources_use_absolute_imports_only(self):
        offenders = []
        for path in _python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: relative import")
        self.assertEqual(offenders, [], "Relative imports do not work in the flat layout")

    def test_sources_do_not_import_removed_modules(self):
        offenders = []
        for path in _python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    root = name.split(".", 1)[0]
                    if root in REMOVED_MODULES:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}: import {name}"
                        )
        self.assertEqual(offenders, [], "Sources import removed legacy modules")

    def test_flat_layout_has_no_package_directory(self):
        self.assertFalse(
            (REPO_ROOT / "english").exists(),
            "The package directory was intentionally removed; sources live at the repository root",
        )
        for path in _python_sources():
            self.assertIn(
                path.parent,
                (REPO_ROOT, REPO_ROOT / "tests"),
                f"{path} is outside the flat layout",
            )

    def test_config_paths_point_at_the_repository_root(self):
        self.assertEqual(Path(en_config.REPO_ROOT), REPO_ROOT)
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
            self.assertEqual(Path(directory).parent, REPO_ROOT)
        self.assertEqual(en_config.DEFAULT_DATASET, "HuggingFaceFW/fineweb-edu")

    def test_removed_legacy_files_are_absent(self):
        for name in REMOVED_FILES:
            self.assertFalse(
                (REPO_ROOT / name).exists(), f"legacy file still present: {name}"
            )
        self.assertFalse((REPO_ROOT / "memory").exists(), "legacy memory store present")
        self.assertFalse((REPO_ROOT / "_gen_new").exists(), "legacy generated data present")

    def test_required_flat_files_are_present(self):
        for name in REQUIRED_FILES:
            self.assertTrue(
                (REPO_ROOT / name).exists(), f"required English file missing: {name}"
            )


if __name__ == "__main__":
    unittest.main()
