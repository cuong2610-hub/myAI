"""Routing between the local retrieval store and live web search (English)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from rag import RagStore, format_context
from web_search import collect_sources

CURRENT_HINTS = (
    "today",
    "tonight",
    "current",
    "currently",
    "nowadays",
    "latest",
    "newest",
    "recent",
    "recently",
    "news",
    "weather",
    "forecast",
    "price",
    "prices",
    "exchange rate",
    "stock price",
    "release",
    "released",
    "update",
    "updated",
    "version",
    "2026",
    "2027",
)

WEB_HINTS = (
    "search the web",
    "search online",
    "search the internet",
    "look up online",
    "look it up",
    "google",
    "search google",
    "website",
    "web page",
    "source",
    "sources",
    "citation",
    "cite",
    "link",
    "links",
    "url",
    "internet",
    "browse",
)

ENTITY_PATTERN = re.compile(
    r"(?<!\w)(ai|api|python|pytorch|cuda|kaggle|windows|android|iphone|samsung|nvidia)(?!\w)",
    re.IGNORECASE,
)
ENTITY_UPDATE_PATTERN = re.compile(
    r"(?<!\w)(latest|newest|current|currently|update|updated|release|version|new)(?!\w)",
    re.IGNORECASE,
)


def _build_boundary_regex(patterns: tuple[str, ...]) -> re.Pattern:
    escaped = [re.escape(unicodedata.normalize("NFC", p.strip())) for p in patterns if p.strip()]
    return re.compile(rf"(?<!\w)({'|'.join(escaped)})(?!\w)", re.IGNORECASE)


CURRENT_REGEX = _build_boundary_regex(CURRENT_HINTS)
WEB_REGEX = _build_boundary_regex(WEB_HINTS)


@dataclass
class RouteResult:
    use_web: bool
    reason: str
    memory_docs: List[Dict] = field(default_factory=list)
    web_docs: List[Dict] = field(default_factory=list)
    query: str = ""

    @property
    def context(self) -> str:
        seen = set()
        merged: List[Dict] = []
        for doc in list(self.web_docs) + list(self.memory_docs):
            key = RagStore._dedupe_key(doc)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return format_context(merged, query=self.query)


def needs_web(query: str) -> bool:
    q = unicodedata.normalize("NFC", query or "").lower().strip()
    if not q:
        return False
    if q.startswith("/web ") or q == "/web":
        return True
    if WEB_REGEX.search(q) or CURRENT_REGEX.search(q):
        return True
    if ENTITY_PATTERN.search(q) and ENTITY_UPDATE_PATTERN.search(q):
        return True
    return False


def retrieve_context(
    query: str,
    *,
    force_web: bool = False,
    use_memory: bool = True,
    max_web_results: int = 4,
    remember_web: bool = True,
    allow_web: bool = True,
    store: Optional[RagStore] = None,
    collector: Optional[Callable[..., List[Dict]]] = None,
) -> RouteResult:
    """Retrieve grounded context for ``query``.

    ``store`` and ``collector`` are injectable for tests and offline use.
    """
    mem: List[Dict] = []
    if use_memory:
        try:
            mem = (store or RagStore()).retrieve(query, top_k=3)
        except Exception as exc:
            print(f"[WARN] Could not read the RAG store: {exc}")
            mem = []

    use_web_now = bool(force_web or (allow_web and needs_web(query)))
    if not use_web_now:
        return RouteResult(False, "memory", mem, [], query=query)

    search = collector or collect_sources
    try:
        web_docs = search(query, max_results=max_web_results)
    except Exception as exc:
        print(f"[WARN] Web search failed ({type(exc).__name__}: {exc}); using RAG only.")
        return RouteResult(False, "web_failed", mem, [], query=query)

    if remember_web and web_docs:
        try:
            (store or RagStore()).add_documents(web_docs)
        except Exception as exc:
            print(f"[WARN] Could not persist web documents: {exc}")
    return RouteResult(True, "web", mem, web_docs, query=query)
