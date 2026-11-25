"""Chat prompt budgeting, streaming plumbing and checkpoint resolution."""

from __future__ import annotations

import unittest

import torch

from english.chat import ChatConfig, ChatSession, load_chat, resolve_checkpoint
from english.checkpoint import save_checkpoint
from english.config import TINY_CONFIG
from english.model import GPT
from english.rag import RagStore
from english.tests.helpers import build_tokenizer, make_temp_dir, remove_temp_dir
from english.tokenizer import decode, encode, tokenizer_fingerprint


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()
        cls.sp, cls.tok_path = build_tokenizer(cls.tmp, vocab_size=512)
        torch.manual_seed(0)
        cls.model = GPT(TINY_CONFIG)
        cls.store = RagStore(cls.tmp / "memory")
        cls.config = ChatConfig(
            max_new_tokens=6,
            temperature=0,
            use_rag=False,
            use_web=False,
            seed=1,
        )

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def make_session(self, **overrides):
        cfg = ChatConfig(
            max_new_tokens=6,
            temperature=0,
            use_rag=False,
            use_web=False,
            seed=1,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return ChatSession(self.model, self.sp, TINY_CONFIG, "cpu", cfg, store=self.store)

    def test_reply_streams_and_records_history(self):
        session = self.make_session()
        out = session.reply("Hello there")
        self.assertIsInstance(out, str)
        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[0], {"role": "user", "content": "Hello there"})
        self.assertEqual(session.history[1]["role"], "assistant")
        self.assertEqual(session.history[1]["content"], out.strip())
        session.reset()
        self.assertEqual(session.history, [])

    def test_prompt_fits_generation_budget(self):
        session = self.make_session(max_new_tokens=16)
        ids = session.build_prompt("Explain attention briefly.")
        self.assertLessEqual(len(ids), TINY_CONFIG.block_size - 16)
        self.assertEqual(ids[0], 2)

    def test_long_prompt_is_truncated(self):
        session = self.make_session(max_new_tokens=8)
        ids = session.build_prompt("word " * 500)
        self.assertLessEqual(len(ids), TINY_CONFIG.block_size - 8)

    def test_context_is_injected_into_system_message(self):
        session = self.make_session()
        context = "[Source 1] Transformers\nThey use attention."
        ids = session.build_prompt("What do transformers use?", context)
        text = decode(ids, sp=self.sp)
        self.assertIn("Retrieved context", text)
        self.assertIn("They use attention", text)
        header = encode("<|assistant|>\n", sp=self.sp)
        self.assertEqual(ids[-len(header):], header)

    def test_offline_route_is_not_used(self):
        session = self.make_session()
        session.reply("What is Python?", allow_web=False)
        self.assertIsNotNone(session.last_route)
        self.assertFalse(session.last_route.use_web)

    def test_injected_route_context(self):
        session = self.make_session(use_rag=True)
        doc = {"title": "Note", "url": "", "content": "The project uses two T4 GPUs."}
        ids = session.build_prompt("Which GPUs?", "[Source 1] " + doc["content"])
        text = decode(ids, sp=self.sp)
        self.assertIn("two T4 GPUs", text)

    def test_rag_note_roundtrip(self):
        session = self.make_session()
        session.store.clear()
        added = session.add_note("The English checkpoint lives in checkpoints_en.")
        self.assertEqual(added, 1)
        hits = session.store.retrieve("Where is the English checkpoint?", top_k=3)
        self.assertTrue(hits)
        self.assertIn("checkpoints_en", hits[0]["content"])

    def test_resolve_checkpoint_prefers_sft_best(self):
        for name in ("pretrain_last.pth", "pretrain_best.pth", "sft_last.pth", "sft_best.pth"):
            (self.tmp / name).write_bytes(b"placeholder")
        chosen = resolve_checkpoint("", self.tmp)
        self.assertEqual(chosen.name, "sft_best.pth")
        explicit = resolve_checkpoint(str(self.tmp / "pretrain_last.pth"), self.tmp)
        self.assertEqual(explicit.name, "pretrain_last.pth")
        with self.assertRaises(SystemExit):
            resolve_checkpoint(str(self.tmp / "missing.pth"), self.tmp)

    def test_load_chat_end_to_end(self):
        ckpt_dir = self.tmp / "ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        save_checkpoint(
            ckpt_dir / "pretrain_best.pth",
            model=self.model,
            config=TINY_CONFIG,
            mode="pretrain",
            step=3,
            tokens_seen=100,
            best_val=1.0,
            tokenizer_fp=tokenizer_fingerprint(self.tok_path),
            include_optimizer=False,
        )
        session, info = load_chat(
            checkpoint=str(ckpt_dir / "pretrain_best.pth"),
            tokenizer=str(self.tok_path),
            device="cpu",
            config=self.config,
            checkpoint_dir=ckpt_dir,
            store_dir=str(self.tmp / "store"),
        )
        self.assertEqual(info.mode, "pretrain")
        text = session.reply("Say something short.")
        self.assertIsInstance(text, str)
        self.assertEqual(len(session.history), 2)


if __name__ == "__main__":
    unittest.main()
