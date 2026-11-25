"""Web/memory routing decisions and grounded context assembly."""

from __future__ import annotations

import unittest

from agent import RouteResult, needs_web, retrieve_context
from rag import RagStore
from tests.helpers import make_temp_dir, remove_temp_dir


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.store = RagStore(self.tmp / "memory")
        self.store.add_documents(
            [
                {
                    "title": "Python",
                    "url": "https://python.org",
                    "content": "Python is a programming language with a large standard library.",
                }
            ]
        )

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def test_needs_web(self):
        self.assertTrue(needs_web("What is the latest Python version?"))
        self.assertTrue(needs_web("Show me today's weather in London."))
        self.assertTrue(needs_web("search the web for transformer papers"))
        self.assertTrue(needs_web("/web cats"))
        self.assertTrue(needs_web("current events"))
        self.assertFalse(needs_web("What is Python?"))
        self.assertFalse(needs_web("Explain gradient descent."))
        self.assertFalse(needs_web(""))

    def test_memory_route(self):
        result = retrieve_context("What is Python?", store=self.store)
        self.assertFalse(result.use_web)
        self.assertEqual(result.reason, "memory")
        self.assertTrue(result.memory_docs)
        self.assertIn("[Source 1]", result.context)
        self.assertIn("programming language", result.context)

    def test_web_route_and_remember(self):
        calls = []

        def collector(query, max_results=4):
            calls.append((query, max_results))
            return [
                {
                    "title": "Fresh news",
                    "url": "https://news.example/item",
                    "content": "A newly released version was announced today.",
                }
            ]

        result = retrieve_context(
            "What is the latest Python version?",
            store=self.store,
            collector=collector,
            max_web_results=1,
        )
        self.assertTrue(result.use_web)
        self.assertEqual(result.reason, "web")
        self.assertEqual(calls, [("What is the latest Python version?", 1)])
        self.assertIn("newly released", result.context)
        self.assertEqual(len(self.store.retrieve("newly released version")), 1)

    def test_web_failure_falls_back_to_memory(self):
        def broken(query, max_results=4):
            raise RuntimeError("offline")

        result = retrieve_context(
            "latest Python news", store=self.store, collector=broken
        )
        self.assertFalse(result.use_web)
        self.assertEqual(result.reason, "web_failed")
        self.assertTrue(result.memory_docs)

    def test_web_disabled(self):
        result = retrieve_context(
            "latest Python news", store=self.store, allow_web=False
        )
        self.assertFalse(result.use_web)
        self.assertEqual(result.reason, "memory")

    def test_force_web(self):
        def collector(query, max_results=4):
            return [{"title": "T", "url": "https://t", "content": "forced result"}]

        result = retrieve_context(
            "What is Python?", store=self.store, collector=collector, force_web=True
        )
        self.assertTrue(result.use_web)
        self.assertIn("forced result", result.context)

    def test_context_prefers_web_and_deduplicates(self):
        result = RouteResult(
            True,
            "web",
            memory_docs=[
                {"title": "Memory", "url": "https://same", "content": "memory copy"},
                {"title": "Local", "url": "", "content": "local only fact"},
            ],
            web_docs=[
                {"title": "Web", "url": "https://same", "content": "web copy"},
                {"title": "Web2", "url": "https://two", "content": "second web fact"},
            ],
            query="copy fact",
        )
        text = result.context
        self.assertIn("web copy", text)
        self.assertNotIn("memory copy", text)
        self.assertIn("local only fact", text)
        self.assertIn("second web fact", text)
        self.assertEqual(text.count("[Source"), 3)


if __name__ == "__main__":
    unittest.main()
