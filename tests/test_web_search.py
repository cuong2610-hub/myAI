"""Offline tests for web search parsing (network calls are monkeypatched)."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import web_search


DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example Page</a>
  <a class="result__snippet">A useful <b>snippet</b> about examples.</a>
</div>
<div class="result">
  <a class="result__a" href="https://second.example/x">Second Result</a>
  <div class="result__snippet">Another snippet.</div>
</div>
</body></html>
"""

WIKI_JSON = {
    "query": {
        "search": [
            {"title": "Transformer (machine learning)", "snippet": "A <span>transformer</span> uses attention."},
            {"title": "Attention", "snippet": "Attention weighs values."},
        ]
    }
}

PAGE_HTML = """
<html><head><style>.x{}</style><script>bad()</script></head>
<body><h1>Title</h1><p>First paragraph.</p><p>Second paragraph.</p></body></html>
"""


class WebSearchTests(unittest.TestCase):
    def tearDown(self):
        pass

    def test_duckduckgo_parsing(self):
        with mock.patch.object(web_search, "_request", return_value=DDG_HTML):
            results = web_search.search_duckduckgo("example", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Example Page")
        self.assertEqual(results[0].url, "https://example.com/page")
        self.assertIn("useful snippet", results[0].snippet)
        self.assertEqual(results[1].url, "https://second.example/x")

    def test_wikipedia_parsing(self):
        with mock.patch.object(web_search, "_request", return_value=json.dumps(WIKI_JSON)):
            results = web_search.search_wikipedia("transformer", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertIn("Transformer", results[0].title)
        self.assertTrue(results[0].url.startswith("https://en.wikipedia.org/wiki/"))
        self.assertNotIn("<span>", results[0].snippet)

    def test_search_web_falls_back_to_wikipedia(self):
        def fake_request(url, timeout=12):
            if "duckduckgo" in url:
                return DDG_HTML
            return json.dumps(WIKI_JSON)

        with mock.patch.object(web_search, "_request", side_effect=fake_request):
            results = web_search.search_web("transformers", max_results=4)
        self.assertGreaterEqual(len(results), 3)
        urls = [r.url for r in results]
        self.assertEqual(len(urls), len(set(urls)))

    def test_fetch_page_extracts_text(self):
        with mock.patch.object(web_search, "_request", return_value=PAGE_HTML):
            text = web_search.fetch_page("https://example.com")
        self.assertIn("First paragraph.", text)
        self.assertIn("Second paragraph.", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn(".x{}", text)

    def test_fetch_page_rejects_non_http(self):
        self.assertIsNone(web_search.fetch_page("file:///etc/passwd"))
        self.assertIsNone(web_search.fetch_page(""))

    def test_fetch_page_failure_returns_none(self):
        with mock.patch.object(web_search, "_request", side_effect=OSError("offline")):
            self.assertIsNone(web_search.fetch_page("https://example.com"))

    def test_search_failures_are_empty(self):
        with mock.patch.object(web_search, "_request", side_effect=OSError("offline")):
            self.assertEqual(web_search.search_duckduckgo("x"), [])
            self.assertEqual(web_search.search_wikipedia("x"), [])
            self.assertEqual(web_search.search_web("x"), [])

    def test_duck_url_decoding(self):
        self.assertEqual(
            web_search._duck_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fb"),
            "https://a.example/b",
        )
        self.assertEqual(web_search._duck_url("https://plain.example"), "https://plain.example")

    def test_collect_sources(self):
        result = web_search.SearchResult("T", "https://t", "snippet")
        with mock.patch.object(web_search, "search_web", return_value=[result]):
            with mock.patch.object(web_search, "fetch_page", return_value="page body"):
                docs = web_search.collect_sources("q", max_results=1, delay=0)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "page body")
        self.assertEqual(docs[0]["url"], "https://t")

    def test_cli_json_output(self):
        result = web_search.SearchResult("T", "https://t", "snippet")
        with mock.patch.object(web_search, "search_web", return_value=[result]):
            code = web_search.main(["query", "--json"])
        self.assertEqual(code, 0)
        with mock.patch.object(web_search, "search_web", return_value=[]):
            self.assertEqual(web_search.main(["nothing"]), 1)


if __name__ == "__main__":
    unittest.main()
