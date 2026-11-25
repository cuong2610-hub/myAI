"""Resumable FineWeb-Edu streaming and token packing.

Key properties
--------------
* Uses Hugging Face ``streaming=True``; the full corpus is never loaded into RAM.
* Deterministic train/validation split by hashing document ids.
* Per-rank sharding for DDP.
* A *custom* shuffle buffer whose contents are stored in the checkpoint.  HF's
  own shuffle buffer is not restorable (verified), so relying on it would make
  resume approximate.
* Exact resume is possible when the world size and stream configuration are
  unchanged.  A world-size change requires ``--allow-data-restart`` and is
  reported loudly by the training script.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from config import (
    BOS_ID,
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_SPLIT,
    EOS_ID,
)


class DataStreamError(RuntimeError):
    pass


@dataclass
class RetryConfig:
    max_retries: int = 6
    base_delay: float = 2.0
    max_delay: float = 60.0


def _backoff(attempt: int, cfg: RetryConfig) -> float:
    return min(cfg.max_delay, cfg.base_delay * (2 ** attempt))


def document_bucket(doc_id: Any) -> int:
    """Deterministic 64-bit bucket for a FineWeb document id."""
    raw = str(doc_id).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def is_val_document(doc_id: Any, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    threshold = int(float(val_fraction) * (2 ** 64))
    return document_bucket(doc_id) < threshold


def _load_streaming(
    source: str | Path,
    config: Optional[str],
    split: str,
    retries: int,
    remote_batch_size: int = 128,
):
    """Load a local JSON/JSONL file or a Hugging Face streaming dataset.

    ``remote_batch_size`` bounds the number of rows in each parquet table so
    that exact resume only has to replay a small tail (see ``DocumentStream``).
    Builders that do not accept the option are loaded again without it.
    """
    from datasets import DownloadConfig, load_dataset

    src = str(source)
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    try:
        kwargs["download_config"] = DownloadConfig(max_retries=max(1, int(retries)))
    except TypeError:
        pass

    if os.path.exists(src):
        # Local development / tests: one JSON object per line.
        return load_dataset("json", data_files=src, **kwargs)
    if config:
        kwargs["name"] = config
    try:
        return load_dataset(src, batch_size=int(remote_batch_size), **kwargs)
    except TypeError:
        return load_dataset(src, **kwargs)


class DocumentStream:
    """Stream of accepted documents with an exact-resume shuffle buffer."""

    def __init__(
        self,
        source: str | Path = DEFAULT_DATASET,
        config: Optional[str] = DEFAULT_DATASET_CONFIG,
        split: str = DEFAULT_SPLIT,
        *,
        dataset_obj: Any = None,
        world_size: int = 1,
        rank: int = 0,
        val_fraction: float = 0.01,
        mode: str = "train",
        shuffle_buffer: int = 1000,
        seed: int = 1337,
        min_chars: int = 32,
        max_chars: int = 200_000,
        retry: Optional[RetryConfig] = None,
        include_url: bool = False,
    ):
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        if not 0 <= rank < max(1, world_size):
            raise ValueError("rank must be in [0, world_size)")
        self.source = str(source)
        self.config = config
        self.split = split
        self.mode = mode
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.val_fraction = float(val_fraction)
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.retry = retry or RetryConfig()
        self.include_url = include_url
        self._rng = random.Random(int(seed) + 1009 * int(rank))
        self._epoch = 0
        self._buffer: List[Dict[str, Any]] = []
        self.docs_seen = 0
        self.skipped = 0
        self._last_state: Any = None
        self._dataset = (
            dataset_obj
            if dataset_obj is not None
            else _load_streaming(source, config, split, self.retry.max_retries)
        )
        if mode == "train" and self.world_size > 1:
            try:
                self._dataset = self._dataset.shard(self.world_size, self.rank)
            except Exception as exc:
                raise DataStreamError(
                    f"Cannot shard {self.source!r} into {self.world_size} ranks: {exc}. "
                    "Use a dataset with at least that many shards."
                ) from exc
        self._it = iter(self._dataset)
        self._last_state = self._safe_state()
        # HF's iterable ``state_dict`` is only updated at internal table
        # boundaries, so restoring it can jump over the remaining rows of the
        # table that is currently being read.  We therefore keep the state of
        # the *start* of the current table plus the number of raw rows already
        # consumed from it: restore then rewinds to that boundary and skips
        # exactly ``_replay_docs`` rows, which is deterministic and cheap
        # (bounded by one table, e.g. ``batch_size`` rows for parquet).
        self._state_boundary: Any = copy.deepcopy(self._last_state)
        self._replay_docs = 0

    # -- state -----------------------------------------------------------
    def _safe_state(self) -> Any:
        try:
            return copy.deepcopy(self._dataset.state_dict())
        except Exception:
            return None

    def _rewind_to_boundary(self) -> None:
        """Restore the last table boundary and replay consumed raw rows."""
        if self._state_boundary is not None:
            try:
                self._dataset.load_state_dict(copy.deepcopy(self._state_boundary))
            except Exception as exc:
                raise DataStreamError(
                    "The dataset iterator cannot be restored at its boundary "
                    f"({type(exc).__name__}: {exc})."
                ) from exc
        self._it = iter(self._dataset)
        for i in range(self._replay_docs):
            try:
                next(self._it)
            except StopIteration as exc:
                raise DataStreamError(
                    f"Dataset ended while replaying {i + 1}/{self._replay_docs} "
                    "rows from the resume boundary; the source changed."
                ) from exc
        self._last_state = self._safe_state()

    def _advance_state(self, state_after: Any) -> None:
        """Track the current table boundary and the rows consumed since it."""
        if state_after != self._last_state:
            self._state_boundary = copy.deepcopy(self._last_state)
            self._replay_docs = 1
        else:
            self._replay_docs += 1
        self._last_state = state_after

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": 2,
            "source": self.source,
            "config": self.config,
            "split": self.split,
            "mode": self.mode,
            "world_size": self.world_size,
            "rank": self.rank,
            "val_fraction": self.val_fraction,
            "shuffle_buffer": self.shuffle_buffer,
            "hf_state": copy.deepcopy(self._state_boundary),
            "hf_replay_docs": int(self._replay_docs),
            "buffer": copy.deepcopy(self._buffer),
            "rng": self._rng.getstate(),
            "epoch": self._epoch,
            "docs_seen": self.docs_seen,
            "skipped": self.skipped,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if state.get("source") not in (None, self.source):
            raise DataStreamError("Data state belongs to a different dataset source")
        if int(state.get("world_size", self.world_size)) != self.world_size:
            raise DataStreamError(
                f"Data state world_size={state.get('world_size')} != current {self.world_size}"
            )
        if int(state.get("rank", self.rank)) != self.rank:
            raise DataStreamError(
                f"Data state rank={state.get('rank')} != current rank {self.rank}"
            )
        hf_state = state.get("hf_state")
        # Format 1 states only stored the coarse HF cursor.  They are still
        # accepted, but resume is only exact if that cursor landed on a table
        # boundary.
        replay = int(state.get("hf_replay_docs", 0) or 0)
        if hf_state is not None:
            self._state_boundary = copy.deepcopy(hf_state)
            self._replay_docs = replay
            self._rewind_to_boundary()
        self._buffer = copy.deepcopy(state.get("buffer") or [])
        rng_state = state.get("rng")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._epoch = int(state.get("epoch", 0))
        self.docs_seen = int(state.get("docs_seen", 0))
        self.skipped = int(state.get("skipped", 0))

    # -- iteration -------------------------------------------------------
    def _payload(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        out = {"text": doc["text"], "id": doc["id"]}
        if self.include_url:
            out["url"] = doc.get("url", "")
        return out

    def _next_accepted(self) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                doc = next(self._it)
            except StopIteration:
                raise
            except Exception as exc:
                if attempt >= self.retry.max_retries:
                    raise DataStreamError(
                        f"Dataset stream failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = _backoff(attempt, self.retry)
                print(
                    f"[WARN] Data stream error ({type(exc).__name__}: {exc}); "
                    f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry.max_retries})",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                attempt += 1
                self._rewind_to_boundary()
                continue

            self._advance_state(self._safe_state())
            text = doc.get("text", "")
            if not isinstance(text, str):
                self.skipped += 1
                continue
            text = text.strip()
            if len(text) < self.min_chars or len(text) > self.max_chars:
                self.skipped += 1
                continue
            doc_id = doc.get("id")
            if doc_id is None or str(doc_id).strip() == "":
                doc_id = doc.get("url") or hashlib.sha1(
                    text[:2048].encode("utf-8", errors="replace")
                ).hexdigest()
            val = is_val_document(doc_id, self.val_fraction)
            if self.mode == "train" and val:
                self.skipped += 1
                continue
            if self.mode == "val" and not val:
                self.skipped += 1
                continue
            self.docs_seen += 1
            return self._payload({"text": text, "id": str(doc_id), "url": doc.get("url", "")})

    def next_doc(self) -> Dict[str, Any]:
        if self.shuffle_buffer <= 0:
            return self._next_accepted()
        while len(self._buffer) < self.shuffle_buffer:
            self._buffer.append(self._next_accepted())
        idx = self._rng.randrange(len(self._buffer))
        doc = self._buffer[idx]
        self._buffer[idx] = self._next_accepted()
        return doc

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            yield self.next_doc()


class SyntheticDocumentStream:
    """In-memory stream used by smoke tests and unit tests (no network)."""

    def __init__(
        self,
        texts: Sequence[str],
        *,
        seed: int = 1337,
        shuffle_buffer: int = 0,
        repeat: bool = True,
        val_fraction: float = 0.0,
        mode: str = "train",
    ):
        if not texts:
            raise ValueError("texts must not be empty")
        self.texts = [str(t) for t in texts]
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.repeat = bool(repeat)
        self.val_fraction = float(val_fraction)
        self.mode = mode
        self._rng = random.Random(seed)
        self._index = 0
        self._buffer: List[Dict[str, Any]] = []
        self._epoch = 0
        self.docs_seen = 0
        self.skipped = 0

    def _raw(self) -> Dict[str, Any]:
        if self._index >= len(self.texts):
            if not self.repeat:
                raise StopIteration
            self._index = 0
            self._epoch += 1
            self._rng.shuffle(self.texts)
        text = self.texts[self._index]
        self._index += 1
        doc_id = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        return {"text": text, "id": doc_id}

    def next_doc(self) -> Dict[str, Any]:
        if self.shuffle_buffer <= 0:
            doc = self._raw()
        else:
            while len(self._buffer) < self.shuffle_buffer:
                self._buffer.append(self._raw())
            idx = self._rng.randrange(len(self._buffer))
            doc = self._buffer[idx]
            self._buffer[idx] = self._raw()
        self.docs_seen += 1
        return doc

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": 2,
            "source": "synthetic",
            "hf_state": None,
            "buffer": copy.deepcopy(self._buffer),
            "rng": self._rng.getstate(),
            "epoch": self._epoch,
            "index": self._index,
            "texts": list(self.texts),
            "world_size": 1,
            "rank": 0,
            "docs_seen": self.docs_seen,
            "skipped": self.skipped,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.texts = list(state.get("texts", self.texts))
        self._buffer = copy.deepcopy(state.get("buffer") or [])
        rng_state = state.get("rng")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._epoch = int(state.get("epoch", 0))
        self._index = int(state.get("index", 0))
        self.docs_seen = int(state.get("docs_seen", 0))
        self.skipped = int(state.get("skipped", 0))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            yield self.next_doc()


class PackedTokenStream:
    """Pack documents into fixed-length token windows (no padding)."""

    def __init__(
        self,
        doc_stream: Any,
        sp,
        block_size: int,
        *,
        max_doc_tokens: int = 8192,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        self.doc_stream = doc_stream
        self.sp = sp
        self.block_size = int(block_size)
        self.max_doc_tokens = int(max_doc_tokens)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.buffer: List[int] = []
        self.tokens_seen = 0
        self.docs_encoded = 0
        self.doc_tokens_skipped = 0

    def _doc_tokens(self, text: str) -> List[int]:
        ids = [int(i) for i in self.sp.encode(text, out_type=int)]
        if not ids:
            self.doc_tokens_skipped += 1
            return []
        pieces = [ids[i:i + self.max_doc_tokens] for i in range(0, len(ids), self.max_doc_tokens)]
        out: List[int] = []
        for piece in pieces:
            if self.add_bos:
                out.append(BOS_ID)
            out.extend(piece)
            if self.add_eos:
                out.append(EOS_ID)
        return out

    def _fill(self, minimum: int) -> None:
        while len(self.buffer) < minimum:
            doc = self.doc_stream.next_doc()
            tokens = self._doc_tokens(doc["text"])
            if not tokens:
                continue
            self.buffer.extend(tokens)

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for _ in range(int(batch_size)):
            self._fill(self.block_size + 1)
            x = np.asarray(self.buffer[: self.block_size], dtype=np.int64)
            y = np.asarray(self.buffer[1: self.block_size + 1], dtype=np.int64)
            del self.buffer[: self.block_size]
            xs.append(x)
            ys.append(y)
            self.tokens_seen += self.block_size
        return np.stack(xs), np.stack(ys)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": 1,
            "block_size": self.block_size,
            "max_doc_tokens": self.max_doc_tokens,
            "doc_state": self.doc_stream.state_dict(),
            "buffer": list(self.buffer),
            "tokens_seen": self.tokens_seen,
            "docs_encoded": self.docs_encoded,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if int(state.get("block_size", self.block_size)) != self.block_size:
            raise DataStreamError(
                f"Packed data block_size={state.get('block_size')} != {self.block_size}"
            )
        self.doc_stream.load_state_dict(state.get("doc_state") or {})
        self.buffer = [int(i) for i in state.get("buffer", [])]
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.docs_encoded = int(state.get("docs_encoded", 0))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self.next_batch(1)


# ---------------------------------------------------------------------------
# Validation corpus
# ---------------------------------------------------------------------------


def build_val_tokens(
    path: str | Path,
    meta_path: str | Path,
    sp,
    *,
    block_size: int,
    max_tokens: int = 2_000_000,
    source: str = DEFAULT_DATASET,
    config: Optional[str] = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
    val_fraction: float = 0.01,
    seed: int = 1337,
    tokenizer_sha: Optional[str] = None,
    dataset_obj: Any = None,
    max_scan_docs: int = 2_000_000,
    retry: Optional[RetryConfig] = None,
) -> Dict[str, Any]:
    """Build a deterministic validation token bin from hash-split documents."""
    p = Path(path)
    mp = Path(meta_path)
    expected = {
        "format": 1,
        "source": str(source),
        "config": config,
        "split": split,
        "val_fraction": float(val_fraction),
        "block_size": int(block_size),
        "max_tokens": int(max_tokens),
        "tokenizer_sha": tokenizer_sha,
    }
    if p.exists() and mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
            if old.get("meta") == expected:
                return old
        except Exception:
            pass
    if int(sp.vocab_size()) > 65535:
        raise ValueError("uint16 validation bins require vocab_size <= 65535")

    stream = DocumentStream(
        source=source,
        config=config,
        split=split,
        dataset_obj=dataset_obj,
        world_size=1,
        rank=0,
        val_fraction=val_fraction,
        mode="val",
        shuffle_buffer=0,
        seed=seed,
        retry=retry,
    )
    tokens: List[int] = []
    docs = 0
    scanned = 0
    try:
        while len(tokens) < max_tokens and scanned < max_scan_docs:
            doc = stream.next_doc()
            scanned += 1
            ids = [int(i) for i in sp.encode(doc["text"], out_type=int)]
            if not ids:
                continue
            tokens.append(BOS_ID)
            tokens.extend(ids[: max(0, max_tokens - len(tokens) - 1)])
            tokens.append(EOS_ID)
            docs += 1
    except StopIteration:
        pass

    if len(tokens) < block_size + 2:
        raise DataStreamError(
            f"Validation stream produced only {len(tokens)} tokens; "
            "increase val_fraction or lower block_size."
        )
    tokens = tokens[:max_tokens]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    np.asarray(tokens, dtype=np.uint16).tofile(tmp)
    os.replace(tmp, p)
    meta = {
        "meta": expected,
        "tokens": len(tokens),
        "documents": docs,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = mp.with_name(mp.name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_meta, mp)
    return meta


def load_val_tokens(
    path: str | Path,
    meta_path: str | Path,
    *,
    tokenizer_sha: Optional[str] = None,
    block_size: Optional[int] = None,
) -> np.memmap:
    p = Path(path)
    mp = Path(meta_path)
    if not p.exists() or not mp.exists():
        raise FileNotFoundError("Validation token bin is missing; build it first")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    expected = meta.get("meta", {})
    if tokenizer_sha is not None and expected.get("tokenizer_sha") != tokenizer_sha:
        raise DataStreamError("Validation bin was built with a different tokenizer")
    if block_size is not None and expected.get("block_size") != int(block_size):
        raise DataStreamError("Validation bin was built with a different block_size")
    return np.memmap(p, dtype=np.uint16, mode="r")


def batch_from_bin(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
    device=None,
):
    import torch

    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise DataStreamError("Validation data is shorter than block_size + 1")
    starts = rng.integers(0, max_start, size=int(batch_size))
    x = np.stack([np.asarray(data[i: i + block_size], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i + 1: i + block_size + 1], dtype=np.int64) for i in starts])
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    if device is not None:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
    return x_t, y_t
