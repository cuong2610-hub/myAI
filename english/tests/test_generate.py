"""Sampling, streaming generation and KV-cache behaviour."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from english.config import EOS_ID, PAD_ID, TINY_CONFIG
from english.generate import (
    GenerationConfig,
    TextStreamer,
    apply_repetition_penalty,
    filter_top_k,
    filter_top_p,
    generate_ids,
    generate_text,
    sample_token,
)
from english.model import GPT
from english.tests.helpers import build_tokenizer, make_temp_dir, remove_temp_dir


class FakeEos(nn.Module):
    block_size = 16

    def forward(self, idx, past_key_values=None, use_cache=False):
        logits = torch.zeros(1, idx.size(1), TINY_CONFIG.vocab_size)
        logits[0, -1, EOS_ID] = 20.0
        return logits, None, [None]


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()
        cls.sp, _ = build_tokenizer(cls.tmp, vocab_size=512)
        torch.manual_seed(0)
        cls.model = GPT(TINY_CONFIG).eval()

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=0)
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=0)
        with self.assertRaises(ValueError):
            GenerationConfig(repetition_penalty=0)
        self.assertTrue(GenerationConfig(temperature=0).greedy)

    def test_repetition_penalty(self):
        logits = torch.tensor([1.0, -2.0, 3.0])
        out = apply_repetition_penalty(logits, [2, 1], 2.0)
        self.assertAlmostEqual(float(out[2]), 1.5)
        self.assertAlmostEqual(float(out[1]), -4.0)
        self.assertAlmostEqual(float(out[0]), 1.0)
        same = apply_repetition_penalty(logits, [], 2.0)
        self.assertTrue(torch.equal(same, logits))

    def test_top_k_and_top_p(self):
        logits = torch.tensor([1.0, -2.0, 3.0, 0.5])
        kept = filter_top_k(logits, 2)
        self.assertTrue(torch.isneginf(kept[1]))
        self.assertTrue(torch.isneginf(kept[3]))
        self.assertTrue(torch.isfinite(kept[2]))
        peaked = filter_top_p(torch.tensor([10.0, 0.0, 0.0, 0.0]), 0.9)
        self.assertEqual(int(torch.argmax(peaked)), 0)
        self.assertEqual(int(torch.isfinite(peaked).sum()), 1)
        greedy = sample_token(torch.tensor([0.0, 5.0, 1.0]), GenerationConfig(temperature=0))
        self.assertEqual(greedy, 1)

    def test_seeded_sampling_is_reproducible(self):
        prompt = [2, 10, 11, 12]
        cfg = GenerationConfig(max_new_tokens=8, temperature=0.9, top_k=30, top_p=0.9, seed=42)
        a = generate_ids(self.model, prompt, cfg)
        b = generate_ids(self.model, prompt, cfg)
        self.assertEqual(a, b)
        c = generate_ids(self.model, prompt, GenerationConfig(
            max_new_tokens=8, temperature=0.9, top_k=30, top_p=0.9, seed=43))
        self.assertNotEqual(a, c)

    def test_greedy_is_deterministic(self):
        prompt = [2, 30, 31]
        cfg = GenerationConfig(max_new_tokens=10, temperature=0)
        self.assertEqual(generate_ids(self.model, prompt, cfg),
                         generate_ids(self.model, prompt, cfg))

    def test_eos_stops_generation(self):
        cfg = GenerationConfig(max_new_tokens=5, temperature=0)
        self.assertEqual(generate_ids(FakeEos(), [2, 5], cfg), [])

    def test_min_new_tokens_can_delay_eos(self):
        cfg = GenerationConfig(max_new_tokens=5, temperature=0, min_new_tokens=2)
        ids = generate_ids(FakeEos(), [2, 5], cfg)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(t not in {EOS_ID, PAD_ID} for t in ids))

    def test_prompt_overflow_raises(self):
        cfg = GenerationConfig(max_new_tokens=8)
        prompt = [2] * (TINY_CONFIG.block_size - 4)
        with self.assertRaises(ValueError):
            generate_ids(self.model, prompt, cfg)

    def test_empty_prompt_defaults_to_bos(self):
        ids = generate_ids(
            self.model,
            [],
            GenerationConfig(max_new_tokens=2, temperature=0, min_new_tokens=2),
        )
        self.assertEqual(len(ids), 2)

    def test_text_streamer_reconstructs_text(self):
        text = "hello world this is a streaming test"
        ids = [int(i) for i in self.sp.encode(text, out_type=int)]
        streamer = TextStreamer(self.sp)
        pieces = [streamer.push(t) for t in ids]
        pieces.append(streamer.flush())
        self.assertEqual("".join(pieces), text)

    def test_generate_text(self):
        out = generate_text(
            self.model, [2, 20], GenerationConfig(max_new_tokens=3, temperature=0), self.sp
        )
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
