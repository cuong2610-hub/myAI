"""Tests for streaming FineWeb-style data, packing and resumable state."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from english.config import BOS_ID, EOS_ID
from english.data import (
    DocumentStream,
    PackedTokenStream,
    RetryConfig,
    SyntheticDocumentStream,
    batch_from_bin,
    build_val_tokens,
    document_bucket,
    is_val_document,
    load_val_tokens,
)
from english.tests.helpers import build_tokenizer, make_temp_dir, remove_temp_dir

RETRY = RetryConfig(max_retries=1, base_delay=0.0)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()
        cls.sp, cls.tok_path = build_tokenizer(cls.tmp, vocab_size=512)
        cls.jsonl = cls.tmp / "docs.jsonl"
        import random

        rng = random.Random(7)
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
        with cls.jsonl.open("w", encoding="utf-8") as f:
            for i in range(240):
                text = " ".join(rng.choice(words) for _ in range(rng.randint(30, 90)))
                f.write(json.dumps({"id": f"doc-{i}", "text": text}) + "\n")

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def make_stream(self, **kwargs):
        defaults = dict(
            source=self.jsonl,
            config=None,
            split="train",
            val_fraction=0.2,
            world_size=1,
            rank=0,
            mode="train",
            shuffle_buffer=8,
            seed=11,
            min_chars=10,
            max_chars=50_000,
            retry=RETRY,
        )
        defaults.update(kwargs)
        return DocumentStream(**defaults)

    def test_hash_split_is_deterministic(self):
        first = [is_val_document(f"id-{i}", 0.1) for i in range(2000)]
        second = [is_val_document(f"id-{i}", 0.1) for i in range(2000)]
        self.assertEqual(first, second)
        fraction = sum(first) / len(first)
        self.assertGreater(fraction, 0.05)
        self.assertLess(fraction, 0.15)
        self.assertEqual(document_bucket("abc"), document_bucket("abc"))
        self.assertNotEqual(document_bucket("abc"), document_bucket("abd"))
        self.assertFalse(is_val_document("x", 0.0))
        self.assertTrue(is_val_document("x", 1.0))

    def test_stream_resume_is_exact(self):
        a = self.make_stream()
        for _ in range(5):
            a.next_doc()
        state = a.state_dict()
        rest_a = [a.next_doc()["id"] for _ in range(12)]
        b = self.make_stream()
        b.load_state_dict(state)
        rest_b = [b.next_doc()["id"] for _ in range(12)]
        self.assertEqual(rest_a, rest_b)
        self.assertEqual(a.docs_seen, b.docs_seen)
        self.assertEqual(len(state["buffer"]), 8)

    def test_stream_state_world_size_mismatch(self):
        a = self.make_stream()
        state = a.state_dict()
        state["world_size"] = 2
        b = self.make_stream()
        from english.data import DataStreamError

        with self.assertRaises(DataStreamError):
            b.load_state_dict(state)

    def test_train_and_val_disjoint(self):
        train = self.make_stream(mode="train")
        val = self.make_stream(mode="val")
        train_ids = {train.next_doc()["id"] for _ in range(40)}
        val_ids = {val.next_doc()["id"] for _ in range(20)}
        self.assertFalse(train_ids & val_ids)
        self.assertTrue(all(is_val_document(i, 0.2) for i in val_ids))
        self.assertTrue(all(not is_val_document(i, 0.2) for i in train_ids))

    def test_packing_shapes_and_shift(self):
        stream = PackedTokenStream(self.make_stream(), self.sp, block_size=32,
                                   max_doc_tokens=64)
        x, y = stream.next_batch(3)
        self.assertEqual(x.shape, (3, 32))
        self.assertEqual(y.shape, (3, 32))
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
        self.assertEqual(stream.tokens_seen, 3 * 32)
        self.assertTrue(np.issubdtype(x.dtype, np.integer))

    def test_packing_resume_is_exact(self):
        stream = PackedTokenStream(self.make_stream(), self.sp, block_size=32,
                                   max_doc_tokens=64)
        for _ in range(4):
            stream.next_batch(2)
        state = stream.state_dict()
        expected_x, expected_y = stream.next_batch(2)
        restored = PackedTokenStream(self.make_stream(), self.sp, block_size=32,
                                     max_doc_tokens=64)
        restored.load_state_dict(state)
        got_x, got_y = restored.next_batch(2)
        np.testing.assert_array_equal(expected_x, got_x)
        np.testing.assert_array_equal(expected_y, got_y)
        self.assertEqual(restored.tokens_seen, stream.tokens_seen)

    def test_packing_rejects_wrong_block_size(self):
        stream = PackedTokenStream(self.make_stream(), self.sp, block_size=32)
        state = stream.state_dict()
        other = PackedTokenStream(self.make_stream(), self.sp, block_size=64)
        from english.data import DataStreamError

        with self.assertRaises(DataStreamError):
            other.load_state_dict(state)

    def test_document_boundaries_have_bos_eos(self):
        stream = PackedTokenStream(self.make_stream(), self.sp, block_size=8,
                                   max_doc_tokens=8, add_bos=True, add_eos=True)
        x, _ = stream.next_batch(4)
        flat = x.reshape(-1)
        self.assertTrue(np.any(flat == BOS_ID))
        self.assertTrue(np.any(flat == EOS_ID))

    def test_validation_bin_build_and_load(self):
        bin_path = self.tmp / "val.bin"
        meta_path = self.tmp / "val.json"
        meta = build_val_tokens(
            bin_path,
            meta_path,
            self.sp,
            block_size=32,
            max_tokens=4000,
            source=self.jsonl,
            config=None,
            val_fraction=0.2,
            retry=RETRY,
        )
        self.assertGreaterEqual(meta["tokens"], 32 + 2)
        self.assertLessEqual(meta["tokens"], 4000)
        data = load_val_tokens(bin_path, meta_path, block_size=32)
        self.assertEqual(len(data), meta["tokens"])
        rng = np.random.default_rng(0)
        x, y = batch_from_bin(data, 32, 3, rng)
        self.assertEqual(tuple(x.shape), (3, 32))
        self.assertEqual(tuple(y.shape), (3, 32))
        self.assertTrue(bool((y[:, :-1] == x[:, 1:]).all()))
        with self.assertRaises(Exception):
            load_val_tokens(bin_path, meta_path, block_size=64)

    def test_validation_bin_meta_is_reused(self):
        bin_path = self.tmp / "val2.bin"
        meta_path = self.tmp / "val2.json"
        first = build_val_tokens(
            bin_path, meta_path, self.sp, block_size=32, max_tokens=2000,
            source=self.jsonl, config=None, val_fraction=0.2, retry=RETRY,
        )
        mtime = bin_path.stat().st_mtime_ns
        second = build_val_tokens(
            bin_path, meta_path, self.sp, block_size=32, max_tokens=2000,
            source=self.jsonl, config=None, val_fraction=0.2, retry=RETRY,
        )
        self.assertEqual(first["meta"], second["meta"])
        self.assertEqual(mtime, bin_path.stat().st_mtime_ns)

    def test_synthetic_stream_state(self):
        stream = SyntheticDocumentStream(
            ["alpha beta gamma " * 5, "delta epsilon " * 6, "zeta eta " * 7],
            seed=3,
            shuffle_buffer=2,
            repeat=True,
        )
        for _ in range(3):
            stream.next_doc()
        state = stream.state_dict()
        expected = [stream.next_doc()["text"] for _ in range(6)]
        clone = SyntheticDocumentStream(
            ["alpha beta gamma " * 5, "delta epsilon " * 6, "zeta eta " * 7],
            seed=3,
            shuffle_buffer=2,
            repeat=True,
        )
        clone.load_state_dict(state)
        got = [clone.next_doc()["text"] for _ in range(6)]
        self.assertEqual(expected, got)


if __name__ == "__main__":
    unittest.main()
