"""Small dependency-free lexical RAG store for AI9.

This is intentionally simple: 46M parameter AI9 does not need another large
embedding model just to remember a few retrieved documents.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

MEMORY_DIR = Path("memory")
MEMORY_FILE = MEMORY_DIR / "documents.jsonl"
DEFAULT_TTL = 7 * 86400  # 7 ngày (604,800 giây)

SPECIAL_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
)


def sanitize_text(text: str) -> str:
    """Lọc bỏ các marker đặc biệt để tránh prompt injection và nhiễu token."""
    if not text:
        return ""
    for marker in SPECIAL_MARKERS:
        text = text.replace(marker, "")
    text = text.replace("\x00", "").replace("\r", "")
    return text.strip()


_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_STOPWORDS = {
    "là", "và", "của", "cho", "trong", "một", "các", "những", "được", "với", "từ",
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "for", "on", "with",
}


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1 and w.lower() not in _STOPWORDS]


def ensure_store():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.touch()


def load_documents(ttl: Optional[float] = DEFAULT_TTL) -> List[Dict]:
    """Tải danh sách tài liệu từ RAG store, lọc bỏ tài liệu hết hạn TTL và khử trùng URL."""
    ensure_store()
    now = time.time()
    try:
        fallback_time = MEMORY_FILE.stat().st_mtime
    except Exception:
        fallback_time = now

    docs = []
    seen_urls = set()
    with MEMORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_url = d.get("url", "").strip()
            norm_url = raw_url.lower()
            if not norm_url or norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)

            ts = d.get("timestamp")
            if ts is None:
                d["timestamp"] = fallback_time
                ts = fallback_time

            # Bỏ tài liệu đã quá hạn TTL (dữ liệu cũ như giá, thời tiết...)
            if ttl is not None and (now - float(ts)) > ttl:
                continue

            d["title"] = sanitize_text(d.get("title", ""))
            d["content"] = sanitize_text(d.get("content", ""))
            docs.append(d)
    return docs


def add_documents(documents: List[Dict], ttl: Optional[float] = DEFAULT_TTL) -> int:
    """Lưu tài liệu vào RAG: lưu kèm timestamp, khử trùng lặp theo URL và dọn dẹp tài liệu hết hạn."""
    if not documents:
        return 0
    ensure_store()
    now = time.time()

    # Tải các tài liệu hợp lệ hiện có
    existing_docs = {}
    for d in load_documents(ttl=ttl):
        u = d.get("url", "").strip().lower()
        if u:
            existing_docs[u] = d

    added = 0
    for d in documents:
        raw_url = d.get("url", "").strip()
        u = raw_url.lower()
        if not u:
            continue
        text = sanitize_text(d.get("content") or d.get("snippet") or "")
        if not text:
            continue
        title = sanitize_text(d.get("title", ""))
        row = {
            "title": title,
            "url": raw_url,
            "content": text,
            "timestamp": float(d.get("timestamp", now)),
        }
        if u not in existing_docs:
            added += 1
        existing_docs[u] = row

    # Lưu lại atomic vào file
    tmp_file = MEMORY_FILE.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        for row in existing_docs.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_file.replace(MEMORY_FILE)

    global _INDEX_KEY, _INDEX
    _INDEX_KEY = None
    _INDEX = None
    return added


def clear_memory():
    global _INDEX_KEY, _INDEX
    ensure_store()
    MEMORY_FILE.write_text("", encoding="utf-8")
    _INDEX_KEY = None
    _INDEX = None


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q = {}
    d = {}
    for t in query_tokens:
        q[t] = q.get(t, 0) + 1
    for t in doc_tokens:
        d[t] = d.get(t, 0) + 1
    dot = sum(q[k] * d.get(k, 0) for k in q)
    qn = math.sqrt(sum(v * v for v in q.values()))
    dn = math.sqrt(sum(v * v for v in d.values()))
    return dot / (qn * dn + 1e-9)


def retrieve(query: str, top_k: int = 3, min_score: float = 0.04, ttl: Optional[float] = DEFAULT_TTL) -> List[Dict]:
    q = _tokens(query)
    ensure_store()
    stat = MEMORY_FILE.stat()
    docs = _load_index(stat.st_mtime_ns, stat.st_size, ttl=ttl)
    scored = []
    now = time.time()
    for d, tokens in docs:
        ts = d.get("timestamp")
        if ts is not None and ttl is not None and (now - float(ts)) > ttl:
            continue
        s = _score(q, tokens)
        if s >= min_score:
            item = dict(d)
            item["score"] = s
            scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Khử trùng lặp URL trong kết quả trả về
    seen_urls = set()
    deduped = []
    for item in scored:
        u = item.get("url", "").strip().lower()
        if u and u in seen_urls:
            continue
        if u:
            seen_urls.add(u)
        deduped.append(item)
    return deduped[:top_k]


_INDEX_KEY = None
_INDEX = None


def _load_index(mtime_ns: int, size: int, ttl: Optional[float] = DEFAULT_TTL):
    global _INDEX_KEY, _INDEX
    key = (mtime_ns, size)
    if _INDEX_KEY != key:
        _INDEX = [
            (d, _tokens((d.get("title", "") + " " + d.get("content", ""))[:30_000]))
            for d in load_documents(ttl=ttl)
        ]
        _INDEX_KEY = key
    return _INDEX


DEFAULT_CONTEXT_CHARS = 1_800


def _cut_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip()


def _select_passage(content: str, budget: int, query_tokens: List[str]) -> str:
    """Chọn đoạn liên quan nhất tới truy vấn trong ngân sách, thay vì luôn lấy đầu trang."""
    if budget <= 0:
        return ""
    if len(content) <= budget:
        return content
    if not query_tokens:
        return _cut_at_word(content, budget)

    stride = max(1, budget // 2)
    best_start, best_score = 0, -1.0
    for start in range(0, len(content) - budget + 1, stride):
        s = _score(query_tokens, _tokens(content[start:start + budget]))
        if s > best_score:
            best_start, best_score = start, s
    return _cut_at_word(content[best_start:best_start + budget], budget)


def format_context(docs: List[Dict], max_chars: int = DEFAULT_CONTEXT_CHARS, query: str = "") -> str:
    """Gộp tài liệu thành ngữ cảnh, chia đều ngân sách ký tự cho mỗi tài liệu.

    Ngân sách mặc định ~1.800 ký tự để không vượt context 768 token của model;
    chat.py còn cắt chính xác theo token trong build_prompt.
    """
    unique = []
    seen_urls = set()
    for d in docs:
        raw_url = d.get("url", "").strip()
        u = raw_url.lower()
        if u and u in seen_urls:
            continue
        content = sanitize_text(d.get("content", ""))
        if not content:
            continue
        if u:
            seen_urls.add(u)
        unique.append((sanitize_text(d.get("title", "")), raw_url, content))
    if not unique:
        return ""

    query_tokens = _tokens(query)
    per_doc = max_chars // len(unique)
    chunks = []
    for i, (title, raw_url, content) in enumerate(unique, 1):
        header = f"[Nguồn {i}] {title}\nURL: {raw_url}\n"
        body = _select_passage(content, per_doc - len(header), query_tokens)
        if body:
            chunks.append(header + body)
    return "\n\n".join(chunks)
