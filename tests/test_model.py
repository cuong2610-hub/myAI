"""Tests for the English model architecture and exact parameter count."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from config import BASE_CONFIG, TINY_CONFIG, ModelConfig, get_preset
from model import GPT, build_rope_cache


class ModelTests(unittest.TestCase):
    def test_base_parameter_count_is_exact(self):
        self.assertEqual(BASE_CONFIG.param_count(), 124_278_528)
        model = GPT(BASE_CONFIG)
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(actual, 124_278_528)
        self.assertEqual(model.num_parameters(), 124_278_528)
        self.assertEqual(model.num_parameters(non_embedding=True), 99_112_704)
        del model

    def test_tiny_parameter_count_matches_formula(self):
        model = GPT(TINY_CONFIG)
        self.assertEqual(TINY_CONFIG.param_count(), 393_856)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 393_856)

    def test_wide_parameter_count_matches_formula(self):
        cfg = get_preset("wide")
        model = GPT(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), cfg.param_count())

    def test_tied_embeddings_share_storage(self):
        model = GPT(TINY_CONFIG)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertEqual(len({id(p) for p in model.parameters()}),
                         sum(1 for _ in model.parameters()))

    def test_forward_shapes_and_loss(self):
        torch.manual_seed(0)
        model = GPT(TINY_CONFIG)
        x = torch.randint(4, TINY_CONFIG.vocab_size, (3, 16))
        y = torch.randint(4, TINY_CONFIG.vocab_size, (3, 16))
        logits, loss, cache = model(x, y)
        self.assertEqual(tuple(logits.shape), (3, 16, TINY_CONFIG.vocab_size))
        self.assertIsNone(cache)
        expected = F.cross_entropy(
            logits.reshape(-1, TINY_CONFIG.vocab_size), y.reshape(-1)
        )
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        self.assertTrue(torch.isfinite(loss))

    def test_ignore_index_masking(self):
        torch.manual_seed(1)
        model = GPT(TINY_CONFIG)
        x = torch.randint(4, TINY_CONFIG.vocab_size, (2, 12))
        y = torch.randint(4, TINY_CONFIG.vocab_size, (2, 12))
        y[:, :6] = -100
        logits, loss, _ = model(x, y)
        expected = F.cross_entropy(
            logits.reshape(-1, TINY_CONFIG.vocab_size), y.reshape(-1), ignore_index=-100
        )
        self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_backward_creates_gradients(self):
        model = GPT(TINY_CONFIG)
        x = torch.randint(4, TINY_CONFIG.vocab_size, (2, 8))
        y = torch.randint(4, TINY_CONFIG.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        grads = [p for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g.grad).all() for g in grads))

    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(3)
        model = GPT(TINY_CONFIG).eval()
        seq = torch.randint(4, TINY_CONFIG.vocab_size, (1, 12))
        with torch.no_grad():
            full, _, _ = model(seq)
            part1, _, cache = model(seq[:, :5], use_cache=True)
            part2, _, cache = model(seq[:, 5:9], past_key_values=cache, use_cache=True)
            part3, _, _ = model(seq[:, 9:], past_key_values=cache, use_cache=True)
        self.assertLess(float((full[:, :5] - part1).abs().max()), 1e-5)
        self.assertLess(float((full[:, 5:9] - part2).abs().max()), 1e-5)
        self.assertLess(float((full[:, 9:] - part3).abs().max()), 1e-5)

    def test_attention_mask_pads(self):
        torch.manual_seed(4)
        model = GPT(TINY_CONFIG).eval()
        x = torch.randint(4, TINY_CONFIG.vocab_size, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[1, 7:] = False
        with torch.no_grad():
            full, _, _ = model(x)
            masked, _, _ = model(x, attention_mask=mask)
        # Real tokens of batch 0 and prefix of batch 1 are unaffected by later pads.
        self.assertLess(float((full[0] - masked[0]).abs().max()), 1e-5)
        self.assertLess(float((full[1, :7] - masked[1, :7]).abs().max()), 1e-5)

    def test_sequence_too_long_raises(self):
        model = GPT(TINY_CONFIG).eval()
        x = torch.zeros(1, TINY_CONFIG.block_size + 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x)

    def test_mismatched_targets_raise(self):
        model = GPT(TINY_CONFIG)
        x = torch.zeros(1, 8, dtype=torch.long)
        y = torch.zeros(1, 7, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x, y)

    def test_rope_cache(self):
        cos, sin = build_rope_cache(8, 16, 10000.0)
        self.assertEqual(tuple(cos.shape), (16, 8))
        self.assertEqual(tuple(sin.shape), (16, 8))
        self.assertLess(float((cos ** 2 + sin ** 2 - 1.0).abs().max()), 1e-5)
        with self.assertRaises(ValueError):
            build_rope_cache(7, 16)

    def test_gqa_variant(self):
        cfg = ModelConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            n_inner=128,
            n_kv_head=2,
        )
        model = GPT(cfg).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), cfg.param_count())
        x = torch.randint(4, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            logits, _, _ = model(x)
            first, _, cache = model(x[:, :4], use_cache=True)
            second, _, _ = model(x[:, 4:], past_key_values=cache, use_cache=True)
        self.assertEqual(tuple(logits.shape), (2, 8, cfg.vocab_size))
        self.assertLess(float((logits[:, :4] - first).abs().max()), 1e-5)
        self.assertLess(float((logits[:, 4:] - second).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
