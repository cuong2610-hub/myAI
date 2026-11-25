"""DDP environment handling, AMP selection, schedules and utilities."""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import TINY_CONFIG
from model import GPT
from utils import (
    JsonlLogger,
    autocast_context,
    build_optimizer,
    capture_rng_state,
    distributed_mean,
    gather_object,
    get_dist_env,
    human_bytes,
    init_nccl,
    is_bf16_supported,
    lr_at_step,
    lr_at_tokens,
    make_grad_scaler,
    resolve_amp,
    restore_rng_state,
    seed_everything,
    unwrap_model,
)


class UtilsTests(unittest.TestCase):
    def test_dist_env_single_process(self):
        saved = {k: os.environ.pop(k, None) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        try:
            self.assertEqual(get_dist_env(), (False, 0, 0, 1))
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_dist_env_two_ranks(self):
        saved = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        os.environ.update({"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"})
        try:
            self.assertEqual(get_dist_env(), (True, 1, 1, 2))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_nccl_requires_cuda(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA available; CPU-only failure path not applicable")
        with self.assertRaises(RuntimeError):
            init_nccl(0, 0, 1)

    def test_amp_selection(self):
        cpu = torch.device("cpu")
        self.assertEqual(resolve_amp("off", cpu), (False, torch.float32, False))
        self.assertEqual(resolve_amp("fp16", cpu), (False, torch.float32, False))
        self.assertEqual(resolve_amp("auto", cpu), (False, torch.float32, False))
        with self.assertRaises(ValueError):
            resolve_amp("int8", cpu)
        self.assertFalse(is_bf16_supported())
        scaler = make_grad_scaler(False)
        self.assertFalse(scaler.is_enabled())
        with autocast_context(cpu, torch.float32, False):
            x = torch.ones(2) * 2
        self.assertTrue(torch.equal(x, torch.tensor([2.0, 2.0])))

    def test_schedules(self):
        peak, low = 1e-3, 1e-4
        self.assertAlmostEqual(lr_at_step(0, 100, 10, peak, low), peak / 10)
        self.assertAlmostEqual(lr_at_step(9, 100, 10, peak, low), peak)
        self.assertAlmostEqual(lr_at_step(100, 100, 10, peak, low), low)
        self.assertLess(lr_at_step(60, 100, 10, peak, low), peak)
        self.assertGreater(lr_at_step(60, 100, 10, peak, low), low)
        self.assertAlmostEqual(lr_at_tokens(0, 1000, 100, peak, low), peak / 100)
        self.assertAlmostEqual(lr_at_tokens(1000, 1000, 100, peak, low), low)

    def test_rng_roundtrip(self):
        seed_everything(123)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        random.random()
        np.random.rand()
        torch.rand(1)
        restore_rng_state(state)
        got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        for a, b in zip(expected, got):
            self.assertAlmostEqual(a, b, places=6)

    def test_no_distributed_helpers(self):
        self.assertEqual(distributed_mean(2.5, torch.device("cpu")), 2.5)
        payload = {"a": [1, 2, 3]}
        self.assertEqual(gather_object(payload), [payload])
        model = GPT(TINY_CONFIG)
        self.assertIs(unwrap_model(model), model)

    def test_optimizer_parameter_groups(self):
        model = GPT(TINY_CONFIG)
        opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
        self.assertEqual(len(opt.param_groups), 2)
        decay = opt.param_groups[0]
        no_decay = opt.param_groups[1]
        self.assertEqual(decay["weight_decay"], 0.1)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        no_decay_ids = {id(p) for p in no_decay["params"]}
        self.assertIn(id(model.token_embedding.weight), no_decay_ids)
        self.assertIn(id(model.ln_f.weight), no_decay_ids)
        self.assertNotIn(id(model.blocks[0].mlp.down.weight), no_decay_ids)

    def test_jsonl_logger_and_formatting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            logger = JsonlLogger(path)
            logger.log({"event": "a", "value": 1})
            logger.log({"event": "b", "value": 2})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["event"] for r in rows], ["a", "b"])
            self.assertTrue(all("time" in r for r in rows))
        self.assertEqual(human_bytes(512), "512.0 B")
        self.assertEqual(human_bytes(2048), "2.0 KiB")

    def test_train_parser_defaults_and_overrides(self):
        from train import build_parser, resolve_config

        args = build_parser().parse_args([])
        self.assertEqual(args.preset, "base")
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.grad_accum, 4)
        self.assertEqual(args.plan_tokens, 5_000_000_000)
        cfg = resolve_config(args)
        self.assertEqual(cfg.param_count(), 124_278_528)
        args2 = build_parser().parse_args(["--n-layer", "3", "--n-embd", "192", "--n-head", "6"])
        cfg2 = resolve_config(args2)
        self.assertEqual(cfg2.n_layer, 3)
        self.assertEqual(cfg2.n_embd, 192)


if __name__ == "__main__":
    unittest.main()
