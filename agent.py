"""AI9 routing logic: decide when to use local memory or the web."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from rag import add_documents, format_context, retrieve
from web_search import collect_sources

CURRENT_HINTS = (
    "hôm nay", "hiện tại", "hiện nay", "mới nhất", "mới đây", "vừa mới", "cập nhật", "tin tức", "thời tiết",
    "tỷ giá", "giá cả", "giá vàng", "giá xăng", "giá bao nhiêu", "bảng giá", "giá vé", "giá bán", "báo giá",
    "bản mới", "phiên bản mới", "release", "latest", "today", "current", "news", "weather", "price",
    "version", "update", "2026", "2027",
)
WEB_HINTS = (
    "tìm trên mạng", "tra web", "tra cứu web", "lên mạng", "google", "tra google", "search google",
    "website", "trang web", "nguồn tin", "nguồn tham khảo", "dẫn nguồn", "cho nguồn", "xin nguồn", "tìm nguồn",
    "link", "đường link", "gửi link", "cho link", "xin link", "url", "địa chỉ web",
    "search web", "search the web", "internet",
)

def _build_boundary_regex(patterns: tuple[str, ...]) -> re.Pattern:
    escaped = [
        re.escape(unicodedata.normalize("NFC", p.strip()))
        for p in patterns
        if p.strip()
    ]
    return re.compile(rf"(?<!\w)({'|'.join(escaped)})(?!\w)", re.IGNORECASE)

CURRENT_REGEX = _build_boundary_regex(CURRENT_HINTS)
WEB_REGEX = _build_boundary_regex(WEB_HINTS)

ENTITY_PATTERN = re.compile(
    r"(?<!\w)(ai|api|python|pytorch|cuda|kaggle|windows|android|iphone|samsung)(?!\w)",
    re.IGNORECASE,
)
ENTITY_UPDATE_PATTERN = re.compile(
    r"(?<!\w)(phiên bản mới|bản mới|mới nhất|hiện tại|hiện nay|cập nhật|update)(?!\w)",
    re.IGNORECASE,
)


@dataclass
class RouteResult:
    use_web: bool
    reason: str
    memory_docs: List[Dict] = field(default_factory=list)
    web_docs: List[Dict] = field(default_factory=list)

    @property
    def context(self) -> str:
        # Khử trùng lặp theo URL: ưu tiên web_docs (mới hơn), loại bỏ tài liệu trùng từ memory_docs
        seen_urls = set()
        merged = []
        for d in self.web_docs:
            u = d.get("url", "").strip().lower()
            if u and u not in seen_urls:
                seen_urls.add(u)
                merged.append(d)
        for d in self.memory_docs:
            u = d.get("url", "").strip().lower()
            if not u or u not in seen_urls:
                if u:
                    seen_urls.add(u)
                merged.append(d)
        return format_context(merged)


def needs_web(query: str) -> bool:
    q = unicodedata.normalize("NFC", query).lower().strip()
    if not q:
        return False
    if q.startswith("/web ") or q == "/web":
        return True
    if WEB_REGEX.search(q):
        return True
    if CURRENT_REGEX.search(q):
        return True
    # Questions about named contemporary entities are safer with retrieval.
    if ENTITY_PATTERN.search(q) and ENTITY_UPDATE_PATTERN.search(q):
        return True
    return False


def retrieve_context(query: str, *, force_web: bool = False, use_memory: bool = True,
                     max_web_results: int = 4, remember_web: bool = True,
                     allow_web: bool = True) -> RouteResult:
    mem = []
    if use_memory:
        try:
            mem = retrieve(query, top_k=3)
        except Exception as exc:
            print(f"[WARN] Lỗi đọc bộ nhớ RAG: {exc}")
            mem = []

    use_web = force_web or (allow_web and needs_web(query))
    reason = "memory"

    if use_web:
        try:
            web_docs = collect_sources(query, max_results=max_web_results)
            if remember_web and web_docs:
                try:
                    add_documents(web_docs)
                except Exception as exc:
                    print(f"[WARN] Không thể lưu tài liệu web vào RAG: {exc}")
            return RouteResult(True, "web", mem, web_docs)
        except Exception as exc:
            print(f"[WARN] Tìm kiếm web thất bại ({exc}); fallback về bộ nhớ RAG.")
            return RouteResult(False, "web_failed", mem, [])

    # For normal factual questions, a good memory hit is enough to ground the prompt.
    return RouteResult(False, reason, mem, [])
