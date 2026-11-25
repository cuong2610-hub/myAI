"""Lexical RAG store, TTL, deduplication and context formatting."""

from __future__ import annotations

import json
import time
import unittest

from rag import RagStore, format_context, sanitize_text, tokenize
from tests.helpers import make_temp_dir, remove_temp_dir


class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.store = RagStore(self.tmp / "memory", ttl=3600)

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def test_add_and_retrieve_ranking(self):
        self.store.add_documents(
            [
                {"title": "Transformers", "url": "https://a", "content": "Transformers use attention over tokens."},
                {"title": "Gardening", "url": "https://b", "content": "Tomatoes need sunlight and water."},
                {"title": "Attention", "url": "https://c", "content": "Attention is a weighted average of values."},
            ]
        )
        hits = self.store.retrieve("How does attention work in transformers?", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("attention", hits[0]["content"].lower())
        self.assertGreater(hits[0]["score"], 0.0)
        self.assertTrue(all("score" in h for h in hits))

    def test_url_deduplication(self):
        self.assertEqual(
            self.store.add_documents(
                [{"title": "A", "url": "https://x", "content": "first version"}]
            ),
            1,
        )
        self.assertEqual(
            self.store.add_documents(
                [{"title": "A2", "url": "https://x", "content": "second version"}]
            ),
            0,
        )
        docs = self.store.load_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "second version")
        hits = self.store.retrieve("second version")
        self.assertEqual(len(hits), 1)

    def test_content_deduplication_without_url(self):
        doc = {"title": "Note", "content": "The model has 124 million parameters."}
        self.assertEqual(self.store.add_documents([doc, dict(doc)]), 1)
        self.assertEqual(len(self.store.load_documents()), 1)

    def test_sanitize_removes_markers(self):
        dirty = "<|user|>hello<|assistant|> world<s></s><pad><unk>\x00"
        clean = sanitize_text(dirty)
        self.assertEqual(clean, "hello world")
        self.assertEqual(tokenize("The the of and Transformer model"), ["transformer", "model"])

    def test_ttl_filters_old_rows(self):
        self.store.ensure()
        old = {
            "title": "Old",
            "url": "https://old",
            "content": "stale fact about prices",
            "timestamp": time.time() - 10_000,
        }
        new = {
            "title": "New",
            "url": "https://new",
            "content": "fresh fact about prices",
            "timestamp": time.time(),
        }
        with self.store.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(old) + "\n")
            f.write(json.dumps(new) + "\n")
        docs = self.store.load_documents()
        self.assertEqual([d["title"] for d in docs], ["New"])
        self.assertEqual(self.store.retrieve("prices")[0]["title"], "New")

    def test_ttl_none_keeps_everything(self):
        self.store.ensure()
        old = {
            "title": "Old",
            "url": "https://old",
            "content": "stale",
            "timestamp": time.time() - 10_000,
        }
        with self.store.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(old) + "\n")
        keep = RagStore(self.tmp / "memory", ttl=None)
        self.assertEqual(len(keep.load_documents()), 1)

    def test_format_context_sources_and_budget(self):
        docs = [
            {"title": "A", "url": "https://a", "content": "alpha " * 400},
            {"title": "B", "url": "https://b", "content": "beta " * 400},
            {"title": "A duplicate", "url": "https://a", "content": "duplicate"},
        ]
        text = format_context(docs, max_chars=800, query="beta")
        self.assertIn("[Source 1]", text)
        self.assertIn("[Source 2]", text)
        self.assertNotIn("[Source 3]", text)
        self.assertLessEqual(len(text), 1600)
        self.assertIn("https://a", text)

    def test_passage_selection_finds_middle(self):
        content = ("padding words " * 300) + "quantum entanglement is the key fact " + ("tail " * 300)
        text = format_context(
            [{"title": "Physics", "content": content}], max_chars=600, query="quantum entanglement"
        )
        self.assertIn("quantum entanglement", text)
        self.assertLess(len(text), len(content))

    def test_empty_and_irrelevant(self):
        self.assertEqual(self.store.retrieve(""), [])
        self.store.add_documents([{"title": "A", "content": "apples and oranges"}])
        self.assertEqual(self.store.retrieve("quantum physics", min_score=0.5), [])
        self.assertEqual(format_context([]), "")

    def test_clear_and_count(self):
        self.store.add_documents([{"title": "A", "content": "one two three"}])
        self.assertEqual(self.store.count(), 1)
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_duplicate_sources_are_merged(self):
        docs = [
            {"title": "Web", "url": "https://same", "content": "web copy"},
            {"title": "Memory", "url": "https://same", "content": "memory copy"},
        ]
        text = format_context(docs, query="copy")
        self.assertEqual(text.count("[Source"), 1)

    def test_store_is_separate_directory(self):
        self.store.add_documents([{"title": "A", "content": "unique molecule xyz"}])
        other = RagStore(self.tmp / "other")
        self.assertEqual(other.count(), 0)


if __name__ == "__main__":
    unittest.main()
