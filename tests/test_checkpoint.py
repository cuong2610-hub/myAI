"""Checkpoint save/load, compatibility checks and atomicity."""

from __future__ import annotations

import json
import unittest

import torch

from checkpoint import (
    CheckpointError,
    atomic_save,
    checkpoint_summary,
    find_latest,
    load_checkpoint,
    load_model_weights,
    peek_checkpoint,
    save_checkpoint,
)
from config import TINY_CONFIG, ModelConfig
from model import GPT
from tests.helpers import make_temp_dir, remove_temp_dir
from utils import build_optimizer, capture_rng_state

FP = {"sha256": "a" * 64, "size": 1234, "path": "tokenizer_en/tokenizer.model"}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.model = GPT(TINY_CONFIG)
        self.optimizer = build_optimizer(self.model, lr=1e-3)
        loss = self.model(
            torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)
        )[1]
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.path = self.tmp / "pretrain_last.pth"

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def save(self, **kwargs):
        params = dict(
            model=self.model,
            config=TINY_CONFIG,
            mode="pretrain",
            step=7,
            tokens_seen=12345,
            best_val=1.25,
            optimizer=self.optimizer,
            tokenizer_fp=FP,
            rng_state=capture_rng_state(),
            data_state={"format": 1, "buffer": [1, 2, 3]},
            train_args={"plan_tokens": 100},
            history=[{"step": 7, "val_loss": 1.25}],
        )
        params.update(kwargs)
        return save_checkpoint(self.path, **params)

    def test_roundtrip_metadata_and_weights(self):
        self.save()
        self.assertTrue(self.path.exists())
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.tmp.iterdir()))
        info = peek_checkpoint(self.path)
        self.assertEqual(info.edition, "en")
        self.assertEqual(info.format, 1)
        self.assertEqual(info.mode, "pretrain")
        self.assertEqual(info.step, 7)
        self.assertEqual(info.tokens_seen, 12345)
        self.assertAlmostEqual(info.best_val, 1.25)
        self.assertEqual(info.tokenizer_sha256, "a" * 64)
        self.assertTrue(info.has_optimizer)
        self.assertTrue(info.has_rng)
        self.assertTrue(info.has_data_state)

        fresh = GPT(TINY_CONFIG)
        fresh_opt = build_optimizer(fresh, lr=1e-3)
        loaded = load_checkpoint(
            self.path,
            fresh,
            config=TINY_CONFIG,
            optimizer=fresh_opt,
            tokenizer_fp=FP,
            expected_mode="pretrain",
        )
        self.assertTrue(loaded.optimizer_loaded)
        self.assertEqual(loaded.data_state["buffer"], [1, 2, 3])
        for (name, a), b in zip(self.model.state_dict().items(), fresh.state_dict().values()):
            self.assertTrue(torch.equal(a, b), name)

    def test_model_only_checkpoint(self):
        self.save(include_optimizer=False)
        info = peek_checkpoint(self.path)
        self.assertFalse(info.has_optimizer)
        self.assertFalse(info.has_data_state)
        fresh = GPT(TINY_CONFIG)
        load_model_weights(self.path, fresh, config=TINY_CONFIG, tokenizer_fp=FP)
        for a, b in zip(self.model.state_dict().values(), fresh.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_edition_marker_is_enforced(self):
        self.save()
        raw = torch.load(self.path, map_location="cpu", weights_only=False)
        raw["edition"] = "vi"
        atomic_save(raw, self.path)
        with self.assertRaises(CheckpointError):
            load_checkpoint(self.path, GPT(TINY_CONFIG), config=TINY_CONFIG)

    def test_architecture_mismatch_rejected(self):
        self.save()
        other = ModelConfig(
            vocab_size=512, block_size=128, n_layer=3, n_head=4, n_embd=128, n_inner=256
        )
        with self.assertRaises(CheckpointError):
            load_checkpoint(self.path, GPT(other), config=other, tokenizer_fp=FP)

    def test_mode_mismatch_rejected(self):
        self.save()
        with self.assertRaises(CheckpointError):
            load_checkpoint(
                self.path, GPT(TINY_CONFIG), config=TINY_CONFIG,
                expected_mode="sft", tokenizer_fp=FP,
            )

    def test_tokenizer_mismatch_rejected(self):
        self.save()
        other = {"sha256": "b" * 64, "size": 1234}
        with self.assertRaises(CheckpointError):
            load_checkpoint(
                self.path, GPT(TINY_CONFIG), config=TINY_CONFIG, tokenizer_fp=other
            )

    def test_rng_restored(self):
        torch.manual_seed(123)
        state = capture_rng_state()
        expected = torch.rand(5)
        torch.manual_seed(999)
        self.save(rng_state=state)
        load_checkpoint(
            self.path, GPT(TINY_CONFIG), config=TINY_CONFIG,
            tokenizer_fp=FP, load_optimizer=False, load_data=False, load_rng=True,
        )
        got = torch.rand(5)
        self.assertTrue(torch.equal(expected, got))

    def test_best_val_zero_is_preserved(self):
        self.save(best_val=0.0)
        self.assertEqual(peek_checkpoint(self.path).best_val, 0.0)

    def test_corrupt_checkpoint_raises(self):
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaises(CheckpointError):
            peek_checkpoint(self.path)
        with self.assertRaises(CheckpointError):
            load_checkpoint(self.path, GPT(TINY_CONFIG))

        missing = self.tmp / "missing.pth"
        with self.assertRaises(CheckpointError):
            peek_checkpoint(missing)

    def test_find_latest_and_summary(self):
        self.save()
        latest = find_latest(self.tmp)
        self.assertEqual(latest, self.path)
        payload = json.loads(checkpoint_summary(self.path))
        self.assertEqual(payload["edition"], "en")
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["vocab_size"], 512)

    def test_atomic_save_overwrites(self):
        atomic_save({"a": 1}, self.path)
        atomic_save({"a": 2}, self.path)
        self.assertEqual(
            torch.load(self.path, map_location="cpu", weights_only=False)["a"], 2
        )


if __name__ == "__main__":
    unittest.main()
