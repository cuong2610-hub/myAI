"""End-to-end CPU smoke test of the pretraining entry point and resume."""

from __future__ import annotations

import contextlib
import io
import os
import unittest

from checkpoint import peek_checkpoint
from train import main as train_main


class TrainSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)
        from tests.helpers import make_temp_dir

        cls.tmp = make_temp_dir()

    @classmethod
    def tearDownClass(cls):
        from tests.helpers import remove_temp_dir

        remove_temp_dir(cls.tmp)

    def run_train(self, *extra):
        argv = [
            "--smoke-test",
            "--tokenizer-dir", str(self.tmp / "tok"),
            "--checkpoint-dir", str(self.tmp / "ckpt"),
            "--data-dir", str(self.tmp / "data"),
            "--runs-dir", str(self.tmp / "runs"),
            *extra,
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = train_main(argv)
        return code, out.getvalue()

    def test_dry_run_reports_base_parameter_count(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = train_main(
                ["--dry-run", "--checkpoint-dir", str(self.tmp / "dry"), "--runs-dir", str(self.tmp / "dry_runs")]
            )
        self.assertEqual(code, 0)
        self.assertIn("124,278,528", out.getvalue())

    def test_smoke_training_and_resume(self):
        code, output = self.run_train("--max-steps", "2")
        self.assertEqual(code, 0, output)
        last = self.tmp / "ckpt" / "smoke" / "pretrain_last.pth"
        best = self.tmp / "ckpt" / "smoke" / "pretrain_best.pth"
        self.assertTrue(last.exists())
        self.assertTrue(best.exists())
        info = peek_checkpoint(last)
        self.assertEqual(info.edition, "en")
        self.assertEqual(info.mode, "pretrain")
        self.assertEqual(info.step, 2)
        self.assertEqual(info.config["n_layer"], 2)
        self.assertEqual(info.config["vocab_size"], 512)
        self.assertTrue(info.has_optimizer)
        self.assertTrue(info.has_data_state)
        self.assertIn("393,856", output)

        code, output = self.run_train("--max-steps", "3")
        self.assertEqual(code, 0, output)
        info = peek_checkpoint(last)
        self.assertEqual(info.step, 3)
        self.assertIn("data=exact", output)
        self.assertGreater(info.tokens_seen, 0)


if __name__ == "__main__":
    unittest.main()
