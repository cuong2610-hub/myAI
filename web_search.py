"""Internet retrieval for AI7.

No API key is required. Uses DuckDuckGo HTML search and Wikipedia's public API.
Designed for Kaggle notebooks with Internet enabled.
"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "AI7-Web/1.0 (+research project)"
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
                self.results.append(SearchResult(self._title.strip(), self._url.strip(), self._snippet.strip()))
        if self._in_snippet and tag in ("a", "div"):
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data


def _request(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8,vi;q=0.7"})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
        charset = r.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _duck_url(raw_url: str) -> str:
    # DuckDuckGo sometimes returns //duckduckgo.com/l/?uddg=<encoded-url>
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        from urllib.parse import parse_qs
        q = parse_qs(parsed.query).get("uddg")
        if q:
            return q[0]
    return raw_url


def search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    if not query.strip():
        return []
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        text = _request(url)
    except Exception:
        return []
    parser = _DuckParser()
    parser.feed(text)
    out = []
    seen = set()
    for r in parser.results:
        r.url = _duck_url(r.url)
        if r.url in seen:
            continue
        seen.add(r.url)
        r.title = _clean_text(r.title)
        r.snippet = _clean_text(r.snippet)
        out.append(r)
        if len(out) >= max_results:
            break
    return out


def search_wikipedia(query: str, max_results: int = 5, lang: str = "vi") -> List[SearchResult]:
    api = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit={max_results}&srsearch=" + quote_plus(query)
    try:
        data = json.loads(_request(api))
    except Exception:
        return []
    out = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub(r"<[^>]+>", " ", item.get("snippet", ""))
        snippet = _clean_text(snippet)
        page_url = f"https://{lang}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
        out.append(SearchResult(title, page_url, snippet))
    return out


def search_web(query: str, max_results: int = 6) -> List[SearchResult]:
    """Search Wikipedia first for knowledge questions, then general web."""
    results = search_duckduckgo(query, max_results=max_results)
    if len(results) < max_results:
        wiki = search_wikipedia(query, max_results=max_results)
        seen = {r.url for r in results}
        for r in wiki:
            if r.url not in seen:
                results.append(r)
                seen.add(r.url)
            if len(results) >= max_results:
                break
    return results


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


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> Optional[str]:
    """Fetch a page and extract readable text. Returns None on failure."""
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
        # Fallback: strip tags with a simple regex.
        text = _clean_text(re.sub(r"<[^>]+>", " ", raw))
    return text[:max_chars] if text else None


def collect_sources(query: str, max_results: int = 5, max_chars_each: int = 7000, delay: float = 0.15):
    results = search_web(query, max_results=max_results)
    docs = []
    for i, r in enumerate(results):
        content = fetch_page(r.url, max_chars=max_chars_each)
        if content:
            docs.append({
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "content": content,
            })
        if delay and i + 1 < len(results):
            time.sleep(delay)
    return docs
