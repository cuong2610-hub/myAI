"""Guards that keep the English edition separate from the Vietnamese original."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from english import config as en_config

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent

VIETNAMESE_CHARS = re.compile(
    "["
    "ăâđêôơư"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
    "òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
    "]"
)

# Root-level Vietnamese modules that the English package must never import.
VIETNAMESE_MODULES = {
    "agent",
    "chat",
    "checkpoint",
    "config",
    "data",
    "distill_generate",
    "generate",
    "model",
    "rag",
    "sft",
    "tokenizer",
    "train",
    "utils",
    "web_search",
}


class EditionIsolationTests(unittest.TestCase):
    def test_english_sources_have_no_vietnamese_diacritics(self):
        offenders = []
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                if VIETNAMESE_CHARS.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()[:80]}")
        self.assertEqual(offenders, [], "Vietnamese text found in English sources")

    def test_english_package_does_not_import_vietnamese_modules(self):
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
                    if name == "english" or root not in VIETNAMESE_MODULES:
                        continue
                    offenders.append(f"{path.name}: import {name}")
        self.assertEqual(offenders, [], "English modules import Vietnamese modules")

    def test_artifact_paths_are_separate(self):
        pairs = [
            (en_config.DEFAULT_TOKENIZER_DIR, "tokenizer"),
            (en_config.DEFAULT_CHECKPOINT_DIR, "checkpoints"),
            (en_config.DEFAULT_DATA_DIR, "data"),
            (en_config.DEFAULT_RUNS_DIR, "runs"),
            (en_config.DEFAULT_MEMORY_DIR, "memory"),
        ]
        for english_dir, vietnamese_name in pairs:
            self.assertNotEqual(english_dir.name, vietnamese_name)
            self.assertTrue(
                english_dir.name.endswith("_en"),
                f"{english_dir} is not clearly English-specific",
            )
            self.assertNotEqual(english_dir, REPO_ROOT / vietnamese_name)
        self.assertEqual(en_config.DEFAULT_DATASET, "HuggingFaceFW/fineweb-edu")

    def test_original_edition_files_still_exist(self):
        for name in ("train.py", "chat.py", "rag.py", "agent.py", "web_search.py", "test_audit.py"):
            self.assertTrue((REPO_ROOT / name).exists(), f"original file missing: {name}")
        self.assertTrue((REPO_ROOT / "memory").is_dir())
        self.assertTrue((REPO_ROOT / "data").is_dir())


if __name__ == "__main__":
    unittest.main()
