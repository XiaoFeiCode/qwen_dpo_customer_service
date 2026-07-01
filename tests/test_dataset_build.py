from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_dpo_cs.build_dataset import build_datasets
from qwen_dpo_cs.io_utils import read_jsonl


class DatasetBuildTest(unittest.TestCase):
    def test_builds_sft_dpo_eval_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw_file = root / "data" / "raw" / "customer_service_seed.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_datasets(raw_file, tmp)
            sft = read_jsonl(paths["sft"])
            dpo = read_jsonl(paths["dpo"])
            eval_records = read_jsonl(paths["eval"])

        self.assertEqual(len(sft), 8)
        self.assertEqual(len(dpo), 8)
        self.assertEqual(len(eval_records), 8)
        self.assertIn("assistant", sft[0]["text"])
        self.assertIn("prompt_text", dpo[0])
        self.assertIn("chosen_text", dpo[0])
        self.assertIn("rejected_text", dpo[0])


if __name__ == "__main__":
    unittest.main()
