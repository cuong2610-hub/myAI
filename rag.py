"""Small dependency-free lexical RAG store for AI9.

This is intentionally simple: 46M parameter AI9 does not need another large
embedding model just to remember a few retrieved documents.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Dict

MEMORY_DIR = Path("memory")
MEMORY_FILE = MEMORY_DIR / "documents.jsonl"

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


def load_documents() -> List[Dict]:
    ensure_store()
    docs = []
    with MEMORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def add_documents(documents: List[Dict]) -> int:
    if not documents:
        return 0
    ensure_store()
    existing = {d.get("url") for d in load_documents()}
    added = 0
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        for d in documents:
            url = d.get("url", "")
            if not url or url in existing:
                continue
            text = (d.get("content") or d.get("snippet") or "").strip()
            if not text:
                continue
            row = {
                "title": d.get("title", ""),
                "url": url,
                "content": text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(url)
            added += 1
    return added


def clear_memory():
    ensure_store()
    MEMORY_FILE.write_text("", encoding="utf-8")


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


def retrieve(query: str, top_k: int = 3, min_score: float = 0.04) -> List[Dict]:
    q = _tokens(query)
    ensure_store()
    stat = MEMORY_FILE.stat()
    docs = _load_index(stat.st_mtime_ns, stat.st_size)
    scored = []
    for d, tokens in docs:
        s = _score(q, tokens)
        if s >= min_score:
            item = dict(d)
            item["score"] = s
            scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


_INDEX_KEY = None
_INDEX = None


def _load_index(mtime_ns: int, size: int):
    global _INDEX_KEY, _INDEX
    key = (mtime_ns, size)
    if _INDEX_KEY != key:
        _INDEX = [
            (d, _tokens((d.get("title", "") + " " + d.get("content", ""))[:30_000]))
            for d in load_documents()
        ]
        _INDEX_KEY = key
    return _INDEX


def format_context(docs: List[Dict], max_chars: int = 10_000) -> str:
    chunks = []
    used = 0
    for i, d in enumerate(docs, 1):
        chunk = f"[Nguồn {i}] {d.get('title', '')}\nURL: {d.get('url', '')}\n{d.get('content', '')}"
        remain = max_chars - used
        if remain <= 0:
            break
        chunk = chunk[:remain]
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)
