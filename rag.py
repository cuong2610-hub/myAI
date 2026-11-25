"""Dependency-free lexical retrieval store for the English edition.

A ~124M parameter model does not need a second embedding model to ground answers
in a handful of retrieved pages.  Documents are stored as JSONL under
``memory_en/`` with a TTL, URL/content deduplication and a small TF-style cosine
scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import DEFAULT_MEMORY_DIR

DEFAULT_TTL = 7 * 86400  # 7 days
DEFAULT_CONTEXT_CHARS = 3_000
MAX_DOC_CHARS = 30_000
MAX_DOCS = 20_000

SPECIAL_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
)

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


def sanitize_text(text: str) -> str:
    """Remove chat/special markers and control characters from retrieved text."""
    if not text:
        return ""
    for marker in SPECIAL_MARKERS:
        text = text.replace(marker, "")
    text = text.replace("\x00", "").replace("\r", "")
    return text.strip()


def tokenize(text: str) -> List[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) > 1 and w.lower() not in _STOPWORDS
    ]


def _score(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q: Dict[str, int] = {}
    d: Dict[str, int] = {}
    for t in query_tokens:
        q[t] = q.get(t, 0) + 1
    for t in doc_tokens:
        d[t] = d.get(t, 0) + 1
    dot = sum(count * d.get(term, 0) for term, count in q.items())
    qn = math.sqrt(sum(v * v for v in q.values()))
    dn = math.sqrt(sum(v * v for v in d.values()))
    return dot / (qn * dn + 1e-9)


def _cut_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip()


def _select_passage(content: str, budget: int, query_tokens: Sequence[str]) -> str:
    """Pick the most query-relevant window instead of always taking the lead."""
    if budget <= 0:
        return ""
    if len(content) <= budget:
        return content
    if not query_tokens:
        return _cut_at_word(content, budget)
    stride = max(1, budget // 2)
    best_start, best_score = 0, -1.0
    for start in range(0, len(content) - budget + 1, stride):
        score = _score(query_tokens, tokenize(content[start:start + budget]))
        if score > best_score:
            best_start, best_score = start, score
    return _cut_at_word(content[best_start:best_start + budget], budget)


class RagStore:
    """A small persistent lexical document store."""

    def __init__(
        self,
        directory: str | Path | None = None,
        filename: str = "documents.jsonl",
        ttl: Optional[float] = DEFAULT_TTL,
        max_docs: int = MAX_DOCS,
    ):
        self.directory = Path(directory) if directory is not None else Path(DEFAULT_MEMORY_DIR)
        self.path = self.directory / filename
        self.ttl = ttl
        self.max_docs = int(max_docs)
        self._index_key = None
        self._index: Optional[List] = None

    # -- storage ---------------------------------------------------------
    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def clear(self) -> None:
        self.ensure()
        self.path.write_text("", encoding="utf-8")
        self._index_key = None
        self._index = None

    def count(self) -> int:
        return len(self.load_documents())

    def load_documents(self) -> List[Dict]:
        self.ensure()
        now = time.time()
        try:
            fallback_time = self.path.stat().st_mtime
        except OSError:
            fallback_time = now
        docs: List[Dict] = []
        seen = set()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = self._dedupe_key(row)
                if not key or key in seen:
                    continue
                seen.add(key)
                ts = row.get("timestamp")
                if ts is None:
                    ts = fallback_time
                row["timestamp"] = ts
                if self.ttl is not None and (now - float(ts)) > self.ttl:
                    continue
                row["title"] = sanitize_text(row.get("title", ""))
                row["content"] = sanitize_text(row.get("content", ""))
                if not row["content"]:
                    continue
                docs.append(row)
        return docs

    @staticmethod
    def _dedupe_key(row: Dict) -> str:
        url = str(row.get("url", "")).strip().lower()
        if url:
            return url
        content = sanitize_text(row.get("content") or row.get("snippet") or "")
        if not content:
            return ""
        return "local:" + hashlib.sha1(content[:2048].encode("utf-8", errors="replace")).hexdigest()

    def add_documents(self, documents: Sequence[Dict]) -> int:
        if not documents:
            return 0
        self.ensure()
        now = time.time()
        existing: Dict[str, Dict] = {}
        for row in self.load_documents():
            key = self._dedupe_key(row)
            if key:
                existing[key] = row

        added = 0
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            content = sanitize_text(doc.get("content") or doc.get("snippet") or "")[:MAX_DOC_CHARS]
            if not content:
                continue
            row = {
                "title": sanitize_text(doc.get("title", "")),
                "url": str(doc.get("url", "")).strip(),
                "content": content,
                "timestamp": float(doc.get("timestamp", now)),
            }
            key = self._dedupe_key(row)
            if not key:
                continue
            if key not in existing:
                added += 1
            existing[key] = row

        rows = list(existing.values())
        if len(rows) > self.max_docs:
            rows.sort(key=lambda r: float(r.get("timestamp", 0.0)))
            rows = rows[-self.max_docs:]
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        self._index_key = None
        self._index = None
        return added

    # -- retrieval -------------------------------------------------------
    def _load_index(self):
        self.ensure()
        stat = self.path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if self._index_key != key or self._index is None:
            self._index = [
                (row, tokenize((row.get("title", "") + " " + row.get("content", ""))[:MAX_DOC_CHARS]))
                for row in self.load_documents()
            ]
            self._index_key = key
        return self._index

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.04,
    ) -> List[Dict]:
        if not (query or "").strip() or top_k < 1:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        now = time.time()
        scored: List[Dict] = []
        for row, tokens in self._load_index():
            ts = row.get("timestamp")
            if ts is not None and self.ttl is not None and (now - float(ts)) > self.ttl:
                continue
            score = _score(q_tokens, tokens)
            if score >= min_score:
                item = dict(row)
                item["score"] = score
                scored.append(item)
        scored.sort(key=lambda r: r["score"], reverse=True)
        seen = set()
        out: List[Dict] = []
        for item in scored:
            key = self._dedupe_key(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= top_k:
                break
        return out


def format_context(
    docs: Sequence[Dict],
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    query: str = "",
) -> str:
    """Merge documents into a compact context with per-source budgets."""
    unique: List[Dict] = []
    seen = set()
    for doc in docs:
        key = RagStore._dedupe_key(doc)
        if not key or key in seen:
            continue
        content = sanitize_text(doc.get("content", ""))
        if not content:
            continue
        seen.add(key)
        unique.append(doc)
    if not unique:
        return ""
    query_tokens = tokenize(query)
    per_doc = max(200, max_chars // len(unique))
    chunks: List[str] = []
    for i, doc in enumerate(unique, 1):
        title = sanitize_text(doc.get("title", "")) or "Untitled"
        url = str(doc.get("url", "")).strip()
        header = f"[Source {i}] {title}\n" + (f"URL: {url}\n" if url else "")
        body = _select_passage(sanitize_text(doc["content"]), max(50, per_doc - len(header)), query_tokens)
        if body:
            chunks.append(header + body)
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Default-store convenience helpers (used by chat/agent)
# ---------------------------------------------------------------------------

_default_store: Optional[RagStore] = None


def default_store() -> RagStore:
    global _default_store
    if _default_store is None:
        _default_store = RagStore()
    return _default_store


def add_documents(documents: Sequence[Dict]) -> int:
    return default_store().add_documents(documents)


def load_documents() -> List[Dict]:
    return default_store().load_documents()


def retrieve(query: str, top_k: int = 3, min_score: float = 0.04) -> List[Dict]:
    return default_store().retrieve(query, top_k=top_k, min_score=min_score)


def clear_memory() -> None:
    default_store().clear()
