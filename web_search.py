"""Internet retrieval for the English edition (no API key required).

Uses DuckDuckGo's HTML endpoint and Wikipedia's public API, both of which work
from a Kaggle notebook with Internet enabled.  All network entry points are
small functions so tests can monkeypatch ``_request`` and run offline.
"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "myAI-English/1.0 (+research project)"
DEFAULT_TIMEOUT = 12
MAX_PAGE_CHARS = 18_000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self):
        return asdict(self)


class _DuckParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: List[SearchResult] = []
        self._title = ""
        self._url = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._url = attrs.get("href", "")
            self._title = ""
        elif tag in ("a", "div") and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet = ""

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._url and self._title:
                # DuckDuckGo emits the snippet *after* the title anchor, so the
                # result is appended now and its snippet is attached later.
                self.results.append(SearchResult(self._title.strip(), self._url.strip(), ""))
        if self._in_snippet and tag in ("a", "div"):
            self._in_snippet = False
            text = _clean_text(self._snippet)
            if text and self.results and not self.results[-1].snippet:
                self.results[-1].snippet = text

    def handle_data(self, data):
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data


class _TextParser(HTMLParser):
    BLOCK = {"p", "div", "article", "section", "li", "h1", "h2", "h3", "h4", "pre", "blockquote"}
    SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK and self.skip == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK and self.skip == 0:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip == 0 and data.strip():
            self.parts.append(data)


def _request(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _duck_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        q = parse_qs(parsed.query).get("uddg")
        if q:
            return q[0]
    return raw_url


def search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    if not (query or "").strip() or max_results < 1:
        return []
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        text = _request(url)
    except Exception:
        return []
    parser = _DuckParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return []
    out: List[SearchResult] = []
    seen = set()
    for result in parser.results:
        result.url = _duck_url(result.url)
        if not result.url or result.url in seen:
            continue
        seen.add(result.url)
        result.title = _clean_text(result.title)
        result.snippet = _clean_text(result.snippet)
        out.append(result)
        if len(out) >= max_results:
            break
    return out


def search_wikipedia(
    query: str,
    max_results: int = 5,
    lang: str = "en",
) -> List[SearchResult]:
    if not (query or "").strip() or max_results < 1:
        return []
    api = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1"
        f"&srlimit={int(max_results)}&srsearch=" + quote_plus(query)
    )
    try:
        data = json.loads(_request(api))
    except Exception:
        return []
    out: List[SearchResult] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = _clean_text(re.sub(r"<[^>]+>", " ", item.get("snippet", "")))
        if not title:
            continue
        page_url = f"https://{lang}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
        out.append(SearchResult(title, page_url, snippet))
    return out


def search_web(query: str, max_results: int = 6) -> List[SearchResult]:
    """General web first, then Wikipedia to fill the remaining slots."""
    results = search_duckduckgo(query, max_results=max_results)
    if len(results) < max_results:
        seen = {r.url for r in results}
        for result in search_wikipedia(query, max_results=max_results):
            if result.url not in seen:
                results.append(result)
                seen.add(result.url)
            if len(results) >= max_results:
                break
    return results[:max_results]


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> Optional[str]:
    """Fetch a page and extract readable text, or ``None`` on failure."""
    if not (url or "").strip().lower().startswith(("http://", "https://")):
        return None
    try:
        raw = _request(url)
    except Exception:
        return None
    parser = _TextParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return None
    text = _clean_text(" ".join(parser.parts))
    if not text:
        text = _clean_text(re.sub(r"<[^>]+>", " ", raw))
    return text[:max_chars] if text else None


def collect_sources(
    query: str,
    max_results: int = 5,
    max_chars_each: int = 7000,
    delay: float = 0.15,
) -> List[dict]:
    results = search_web(query, max_results=max_results)
    docs: List[dict] = []
    for i, result in enumerate(results):
        content = fetch_page(result.url, max_chars=max_chars_each)
        if content:
            docs.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "content": content,
                }
            )
        if delay and i + 1 < len(results):
            time.sleep(delay)
    return docs


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check live web search from the English edition.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--fetch", action="store_true", help="Also fetch page text.")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    query = " ".join(args.query)
    if args.fetch:
        rows = collect_sources(query, max_results=args.max_results, max_chars_each=args.max_chars)
    else:
        rows = [r.to_dict() for r in search_web(query, max_results=args.max_results)]
    if not rows:
        print("No results (check Internet access / search availability).")
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for i, row in enumerate(rows, 1):
            print(f"[{i}] {row.get('title', '')}")
            print(f"    {row.get('url', '')}")
            snippet = row.get("snippet") or row.get("content", "")[:300]
            if snippet:
                print(f"    {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
