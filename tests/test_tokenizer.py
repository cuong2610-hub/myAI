"""Tests for the English tokenizer, special ids and chat formatting."""

from __future__ import annotations

import unittest
from pathlib import Path

from config import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SYSTEM_ID,
    UNK_ID,
    USER_ID,
)
from tests.helpers import build_tokenizer, make_temp_dir, remove_temp_dir
from tokenizer import (
    SPECIAL_IDS,
    TokenizerError,
    build_chat_ids,
    decode,
    encode,
    format_chat_prompt,
    load_tokenizer,
    normalize_messages,
    tokenizer_fingerprint,
)

MESSAGES = [
    {"role": "system", "content": "You are a concise tutor."},
    {"role": "user", "content": "What is a transformer?"},
    {"role": "assistant", "content": "A neural network that uses attention."},
]


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()
        cls.sp, cls.path = build_tokenizer(cls.tmp, vocab_size=512)

    @classmethod
    def tearDownClass(cls):
        remove_temp_dir(cls.tmp)

    def test_special_ids(self):
        self.assertEqual(
            SPECIAL_IDS,
            {
                "pad": PAD_ID,
                "unk": UNK_ID,
                "bos": BOS_ID,
                "eos": EOS_ID,
                "user": USER_ID,
                "assistant": ASSISTANT_ID,
                "system": SYSTEM_ID,
            },
        )
        self.assertEqual(int(self.sp.pad_id()), 0)
        self.assertEqual(int(self.sp.unk_id()), 1)
        self.assertEqual(int(self.sp.bos_id()), 2)
        self.assertEqual(int(self.sp.eos_id()), 3)
        self.assertEqual(int(self.sp.piece_to_id("<|user|>")), 4)
        self.assertEqual(int(self.sp.piece_to_id("<|assistant|>")), 5)
        self.assertEqual(int(self.sp.piece_to_id("<|system|>")), 6)
        self.assertEqual(int(self.sp.vocab_size()), 512)

    def test_roundtrip(self):
        text = "Hello world, this is an English tokenizer test 123."
        ids = encode(text, sp=self.sp, add_bos=True, add_eos=True)
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(decode(ids, sp=self.sp), text)

    def test_unseen_unicode_roundtrip(self):
        text = "Zebra \u03c0 \u00e9\u00e8 \U0001f642 data"
        ids = encode(text, sp=self.sp)
        self.assertEqual(decode(ids, sp=self.sp), text)

    def test_wrong_vocab_rejected(self):
        with self.assertRaises(TokenizerError):
            load_tokenizer(self.path, expected_vocab=32768)

    def test_fingerprint_stable(self):
        a = tokenizer_fingerprint(self.path)
        b = tokenizer_fingerprint(self.path)
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["size"], b["size"])
        self.assertEqual(len(a["sha256"]), 64)

    def test_chat_labels_mask_prompt(self):
        ids, labels = build_chat_ids(MESSAGES, self.sp)
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(labels[0], -100)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(labels[-1], EOS_ID)
        assistant_ids = encode(MESSAGES[-1]["content"], sp=self.sp)
        trained = [x for x in labels if x != -100]
        self.assertEqual(trained, assistant_ids + [EOS_ID])
        self.assertEqual(sum(1 for x in labels if x != -100), len(assistant_ids) + 1)

    def test_chat_prompt_ends_with_assistant_header(self):
        text = format_chat_prompt(MESSAGES[:2])
        self.assertTrue(text.endswith("<|assistant|>\n"))
        ids = encode(text, sp=self.sp)
        header = encode("<|assistant|>\n", sp=self.sp)
        self.assertEqual(ids[-len(header):], header)

    def test_chat_no_eos_option(self):
        ids, labels = build_chat_ids(MESSAGES, self.sp, add_bos=False, add_eos=False)
        self.assertNotEqual(ids[-1], EOS_ID)
        self.assertEqual(len(ids), len(labels))

    def test_normalize_messages(self):
        rows = normalize_messages({"user": "Hi", "assistant": "Hello"})
        self.assertEqual([r["role"] for r in rows], ["user", "assistant"])
        rows = normalize_messages(
            {"system": "Be brief.", "messages": [{"role": "user", "content": "Hi"},
                                                  {"role": "assistant", "content": "Hello"}]}
        )
        self.assertEqual(rows[0]["role"], "system")
        self.assertEqual(len(rows), 3)

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            format_chat_prompt([{"role": "robot", "content": "hi"}])
        with self.assertRaises(ValueError):
            normalize_messages({"messages": [{"role": "robot", "content": "hi"}]})


if __name__ == "__main__":
    unittest.main()
