from __future__ import annotations

import unittest
from pathlib import Path

from qwen_dpo_cs.build_dataset import build_datasets
from qwen_dpo_cs.evaluation import evaluate_records
from qwen_dpo_cs.io_utils import read_jsonl


class EvaluationTest(unittest.TestCase):
    def test_rule_based_evaluation_metrics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        out_dir = root / "data" / "processed"
        paths = build_datasets(root / "data" / "raw" / "customer_service_seed.jsonl", out_dir)
        records = read_jsonl(paths["eval"])
        predictions, metrics = evaluate_records(records)

        self.assertEqual(len(predictions), 8)
        self.assertEqual(metrics["total"], 8)
        self.assertGreaterEqual(metrics["preference_pair_accuracy"], 0.8)
        self.assertLessEqual(metrics["invalid_response_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
