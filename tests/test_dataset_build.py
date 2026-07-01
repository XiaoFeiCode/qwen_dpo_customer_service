from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_dpo_cs.build_dataset import build_datasets
from qwen_dpo_cs.io_utils import read_jsonl
from qwen_dpo_cs.jddc import convert_jddc


class DatasetBuildTest(unittest.TestCase):
    def test_builds_sft_dpo_eval_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw_file = root / "tests" / "fixtures" / "jddc_sample.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            preference_file = Path(tmp) / "preference_pairs.jsonl"
            total = convert_jddc(raw_file, preference_file)
            paths = build_datasets(preference_file, tmp)
            sft = read_jsonl(paths["sft"])
            dpo = read_jsonl(paths["dpo"])
            eval_records = read_jsonl(paths["eval"])

        self.assertEqual(total, 3)
        self.assertEqual(len(sft), 3)
        self.assertEqual(len(dpo), 3)
        self.assertEqual(len(eval_records), 3)
        self.assertIn("assistant", sft[0]["text"])
        self.assertIn("prompt_text", dpo[0])
        self.assertIn("chosen_text", dpo[0])
        self.assertIn("rejected_text", dpo[0])


if __name__ == "__main__":
    unittest.main()
