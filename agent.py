"""AI7 routing logic: decide when to use local memory or the web."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from rag import add_documents, format_context, retrieve
from web_search import collect_sources

CURRENT_HINTS = (
    "hôm nay", "hiện tại", "mới nhất", "mới đây", "cập nhật", "tin tức", "thời tiết",
    "giá", "tỷ giá", "bản mới", "phiên bản mới", "release", "latest", "today",
    "current", "news", "weather", "price", "version", "update", "2026", "2027",
)
WEB_HINTS = (
    "tìm trên mạng", "tra web", "lên mạng", "google", "website", "nguồn", "link", "url",
    "search web", "search the web", "internet",
)


@dataclass
class RouteResult:
    use_web: bool
    reason: str
    memory_docs: List[Dict] = field(default_factory=list)
    web_docs: List[Dict] = field(default_factory=list)

    @property
    def context(self) -> str:
        docs = self.memory_docs + self.web_docs
        return format_context(docs)


def needs_web(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    if q.startswith("/web ") or q == "/web":
        return True
    if any(x in q for x in WEB_HINTS):
        return True
    if any(x in q for x in CURRENT_HINTS):
        return True
    # Questions about named contemporary entities are safer with retrieval.
    if re.search(r"\b(ai|api|python|pytorch|cuda|kaggle|windows|android|iphone|samsung)\b", q) and (
        "phiên bản" in q or "hiện" in q or "mới" in q or "hỗ trợ" in q
    ):
        return True
    return False


def retrieve_context(query: str, *, force_web: bool = False, use_memory: bool = True,
                     max_web_results: int = 4, remember_web: bool = True,
                     allow_web: bool = True) -> RouteResult:
    mem = retrieve(query, top_k=3) if use_memory else []
    use_web = force_web or (allow_web and needs_web(query))
    reason = "memory"

    if use_web:
        reason = "web"
        web_docs = collect_sources(query, max_results=max_web_results)
        if remember_web and web_docs:
            add_documents(web_docs)
        return RouteResult(True, reason, mem, web_docs)

    # For normal factual questions, a good memory hit is enough to ground the prompt.
    return RouteResult(False, reason, mem, [])
