"""SFT data handling, prompt masking, resumability and CLI smoke test."""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path

import torch

from checkpoint import peek_checkpoint, save_checkpoint
from config import TINY_CONFIG
from model import GPT
from sft import (
    SftExample,
    SftStream,
    collate,
    load_examples,
    main as sft_main,
    split_examples,
    truncate_example,
)
from tests.helpers import build_tokenizer, make_temp_dir, remove_temp_dir, write_corpus
from tokenizer import load_tokenizer, tokenizer_fingerprint, train_tokenizer


def make_rows(count: int = 16):
    rows = []
    for i in range(count):
        rows.append(
            {
                "user": f"Question number {i} about topic {i % 4}?",
                "assistant": f"This is a short answer for question {i} with useful detail.",
            }
        )
    # Duplicate one prompt with a second answer to exercise prompt grouping.
    rows.append(
        {
            "user": "Question number 0 about topic 0?",
            "assistant": "An alternative answer for the same prompt.",
        }
    )
    return rows


class SftDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()
        cls.sp, cls.tok_path = build_tokenizer(cls.tmp, vocab_size=512)
        cls.data_path = cls.tmp / "sft.jsonl"
        with cls.data_path.open("w", encoding="utf-8") as f:
            for row in make_rows(16):
                f.write(json.dumps(row) + "\n")
        cls.examples = load_examples([cls.data_path], cls.sp, TINY_CONFIG.block_size)

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def test_examples_have_masked_prompts(self):
        self.assertEqual(len(self.examples), 17)
        for ex in self.examples:
            self.assertGreater(ex.trainable, 0)
            self.assertIn(-100, ex.labels)
            self.assertLessEqual(len(ex.ids), TINY_CONFIG.block_size)
            self.assertEqual(len(ex.ids), len(ex.labels))

    def test_truncation_keeps_answer(self):
        ids = [10] * 150 + [20] * 40
        labels = [-100] * 150 + [20] * 40
        out = truncate_example(ids, labels, block_size=64)
        self.assertIsNotNone(out)
        out_ids, out_labels = out
        self.assertLessEqual(len(out_ids), 64)
        self.assertGreater(sum(1 for x in out_labels if x != -100), 0)
        self.assertEqual(out_labels[-1], 20)

    def test_truncation_rejects_all_masked(self):
        self.assertIsNone(truncate_example([1, 2, 3], [-100, -100, -100], 8))

    def test_split_is_prompt_disjoint_and_stable(self):
        train, val, keys = split_examples(self.examples, val_ratio=0.25, seed=5)
        train_keys = {ex.prompt_key for ex in train}
        val_keys = {ex.prompt_key for ex in val}
        self.assertFalse(train_keys & val_keys)
        self.assertEqual(sorted(val_keys), keys)
        self.assertTrue(train and val)
        train2, val2, keys2 = split_examples(self.examples, val_ratio=0.25, seed=5)
        self.assertEqual([ex.ids for ex in val], [ex.ids for ex in val2])
        self.assertEqual(keys, keys2)

    def test_collate_pads_and_masks(self):
        batch = self.examples[:3]
        x, y, mask = collate(batch)
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, mask.shape)
        self.assertTrue(bool((y[x == 0] == -100).all()))
        for i, ex in enumerate(batch):
            self.assertTrue(bool(mask[i, :len(ex.ids)].all()))
            self.assertFalse(bool(mask[i, len(ex.ids):].any()))

    def test_stream_resume_exact(self):
        train, _, _ = split_examples(self.examples, val_ratio=0.2, seed=1)
        a = SftStream(train, batch_size=3, seed=9)
        for _ in range(2):
            a.next_batch()
        state = a.state_dict()
        expected = []
        for _ in range(3):
            expected.extend(ex.ids for ex in a.next_batch())
        b = SftStream(train, batch_size=3, seed=9)
        b.load_state_dict(state)
        got = []
        for _ in range(3):
            got.extend(ex.ids for ex in b.next_batch())
        self.assertEqual(expected, got)

    def test_stream_rejects_different_dataset(self):
        train, _, _ = split_examples(self.examples, val_ratio=0.2, seed=1)
        a = SftStream(train, batch_size=2)
        state = a.state_dict()
        with self.assertRaises(ValueError):
            SftStream(train[:-1], batch_size=2).load_state_dict(state)


class SftCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)
        cls.tmp = make_temp_dir()
        corpus = write_corpus(cls.tmp / "corpus.txt", lines=2500)
        cls.tok_path = train_tokenizer(
            corpus_path=corpus,
            out_dir=cls.tmp / "tok" / "smoke",
            vocab_size=512,
            force=True,
            num_threads=2,
        )
        cls.sp = load_tokenizer(cls.tok_path, expected_vocab=512)
        cls.data_path = cls.tmp / "sft.jsonl"
        with cls.data_path.open("w", encoding="utf-8") as f:
            for row in make_rows(12):
                f.write(json.dumps(row) + "\n")

        cls.ckpt_dir = cls.tmp / "ckpt"
        smoke_dir = cls.ckpt_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        model = GPT(TINY_CONFIG)
        save_checkpoint(
            smoke_dir / "pretrain_best.pth",
            model=model,
            config=TINY_CONFIG,
            mode="pretrain",
            step=10,
            tokens_seen=1000,
            best_val=1.0,
            tokenizer_fp=tokenizer_fingerprint(cls.tok_path),
            include_optimizer=False,
        )

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def run_sft(self, *extra):
        argv = [
            "--smoke-test",
            "--checkpoint-dir", str(self.ckpt_dir),
            "--tokenizer", str(self.tok_path),
            "--data", str(self.data_path),
            "--max-steps", "2",
            *extra,
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = sft_main(argv)
        return code, out.getvalue()

    def test_cli_smoke_and_resume(self):
        code, output = self.run_sft()
        self.assertEqual(code, 0, output)
        last = self.ckpt_dir / "smoke" / "sft_last.pth"
        self.assertTrue(last.exists())
        info = peek_checkpoint(last)
        self.assertEqual(info.mode, "sft")
        self.assertEqual(info.step, 2)
        self.assertTrue(info.has_data_state)
        self.assertTrue(info.has_optimizer)

        code, output = self.run_sft("--max-steps", "4")
        self.assertEqual(code, 0, output)
        info = peek_checkpoint(last)
        self.assertEqual(info.step, 4)
        self.assertIn("sft_last", str(last))
        self.assertTrue((self.ckpt_dir / "smoke" / "sft_best.pth").exists())


if __name__ == "__main__":
    unittest.main()
