from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from qwen_dpo_cs.inference import RuleBasedCustomerServiceResponder
from qwen_dpo_cs.io_utils import read_jsonl, write_json, write_jsonl
from qwen_dpo_cs.scoring import keyword_recall, score_reply


def evaluate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responder = RuleBasedCustomerServiceResponder()
    predictions: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    keyword_scores: list[float] = []
    invalid_flags: list[bool] = []
    refusal_hits: list[bool] = []
    pair_hits: list[bool] = []

    for record in records:
        user_messages = [m["content"] for m in record["prompt"] if m["role"] == "user"]
        response = responder.generate(user_messages, category=record.get("category"))
        score = score_reply(response, refusal_expected=record.get("refusal_expected", False))
        chosen_score = score_reply(
            record["chosen"], refusal_expected=record.get("refusal_expected", False)
        )
        rejected_score = score_reply(
            record["rejected"], refusal_expected=record.get("refusal_expected", False)
        )
        kw_recall = keyword_recall(response, record.get("expected_keywords", []))
        refusal_expected = bool(record.get("refusal_expected", False))
        refusal_hit = score.has_boundary if refusal_expected else not score.has_boundary

        predictions.append(
            {
                "id": record["id"],
                "category": record.get("category", "unknown"),
                "response": response,
                "quality_score": score.total,
                "keyword_recall": round(kw_recall, 3),
                "invalid": score.invalid,
                "refusal_expected": refusal_expected,
                "refusal_hit": refusal_hit,
                "chosen_score": chosen_score.total,
                "rejected_score": rejected_score.total,
                "preference_pair_hit": chosen_score.total > rejected_score.total,
            }
        )
        quality_scores.append(score.total)
        keyword_scores.append(kw_recall)
        invalid_flags.append(score.invalid)
        refusal_hits.append(refusal_hit)
        pair_hits.append(chosen_score.total > rejected_score.total)

    metrics = {
        "total": len(records),
        "avg_quality_score": round(mean(quality_scores), 3) if quality_scores else 0.0,
        "avg_keyword_recall": round(mean(keyword_scores), 3) if keyword_scores else 0.0,
        "invalid_response_rate": round(sum(invalid_flags) / len(invalid_flags), 3)
        if invalid_flags
        else 0.0,
        "refusal_accuracy": round(sum(refusal_hits) / len(refusal_hits), 3)
        if refusal_hits
        else 0.0,
        "preference_pair_accuracy": round(sum(pair_hits) / len(pair_hits), 3)
        if pair_hits
        else 0.0,
    }
    return predictions, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline customer service evaluation.")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--prediction-out", default="output/eval/rule_predictions.jsonl")
    parser.add_argument("--metrics-out", default="output/eval/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.eval_file)
    predictions, metrics = evaluate_records(records)
    write_jsonl(args.prediction_out, predictions)
    write_json(args.metrics_out, metrics)
    print(f"predictions: {Path(args.prediction_out)}")
    print(f"metrics: {Path(args.metrics_out)}")
    print(metrics)


if __name__ == "__main__":
    main()
