# Data Pipeline

## Input

The input is a JDDC/JDDC-style customer service dialogue collection. The converter supports JSON and JSONL files with common dialogue keys:

- `messages`
- `dialogue`
- `dialog`
- `turns`
- `utterances`
- `conversation`
- `session`

Each turn can use `role/content`, `speaker/utterance`, `text`, `message`, `query`, or `response` keys. Unknown roles are normalized by turn order.

## Normalization

The normalizer converts heterogeneous records into a unified multi-turn representation:

```json
{
  "id": "dialogue_id_000",
  "category": "refund",
  "user_messages": ["..."],
  "chosen": "...",
  "rejected": "...",
  "expected_keywords": ["订单", "退货", "售后"],
  "refusal_expected": false,
  "source": "jddc"
}
```

`category` is inferred with transparent keyword rules. It is used for preference construction and evaluation grouping, not as a gold label.

## Output

`build_dataset.py` creates:

- `sft_train.jsonl`: ChatML-style supervised fine-tuning samples
- `dpo_train.jsonl`: `prompt/chosen/rejected` pairs for TRL DPO
- `eval.jsonl`: prompts and expected evaluation signals
- `dataset_report.md`: record count and category distribution
