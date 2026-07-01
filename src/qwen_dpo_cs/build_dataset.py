from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_dpo_cs.io_utils import read_jsonl, write_jsonl
from qwen_dpo_cs.prompting import to_dpo_record, to_eval_record, to_sft_record

REQUIRED_FIELDS = {"id", "user_messages", "chosen", "rejected"}


def validate_raw_records(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"Record #{index} missing fields: {sorted(missing)}")
        record_id = str(record["id"])
        if record_id in seen:
            raise ValueError(f"Duplicate id: {record_id}")
        seen.add(record_id)
        if not str(record["chosen"]).strip() or not str(record["rejected"]).strip():
            raise ValueError(f"Record {record_id} has empty chosen/rejected response")


def build_datasets(input_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    records = read_jsonl(input_path)
    validate_raw_records(records)

    sft_records = [to_sft_record(record) for record in records]
    dpo_records = [to_dpo_record(record) for record in records]
    eval_records = [to_eval_record(record) for record in records]

    sft_path = out_dir / "sft_train.jsonl"
    dpo_path = out_dir / "dpo_train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    report_path = out_dir / "dataset_report.md"

    write_jsonl(sft_path, sft_records)
    write_jsonl(dpo_path, dpo_records)
    write_jsonl(eval_path, eval_records)
    write_report(report_path, records)

    return {
        "sft": sft_path,
        "dpo": dpo_path,
        "eval": eval_path,
        "report": report_path,
    }


def write_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(record.get("category", "unknown") for record in records)
    refusal_count = sum(1 for record in records if record.get("refusal_expected"))
    lines = [
        "# Dataset Report",
        "",
        f"- total_records: {len(records)}",
        f"- refusal_expected_records: {refusal_count}",
        "",
        "## Category Distribution",
        "",
    ]
    for category, count in sorted(categories.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `sft_train.jsonl`: ChatML text for supervised fine-tuning.",
            "- `dpo_train.jsonl`: prompt/chosen/rejected preference pairs.",
            "- `eval.jsonl`: offline evaluation prompts and expected signals.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT/DPO/eval datasets from raw JSONL.")
    parser.add_argument("--input", default="data/raw/customer_service_seed.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_datasets(args.input, args.out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
