from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from qwen_dpo_cs.io_utils import write_jsonl


USER_ROLES = {"user", "customer", "buyer", "客户", "用户", "买家", "usr"}
ASSISTANT_ROLES = {"assistant", "agent", "seller", "客服", "商家", "sys", "system"}


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    content: str


def iter_raw_records(input_path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".txt"}
        )
    else:
        files = [path]

    for file in files:
        if file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                for key in ("data", "dialogues", "sessions", "conversations"):
                    value = data.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                yield item
                        break
                else:
                    yield data


def normalize_role(role: Any, index: int) -> str:
    raw = str(role or "").strip().lower()
    if raw in USER_ROLES:
        return "user"
    if raw in ASSISTANT_ROLES:
        return "assistant"
    return "user" if index % 2 == 0 else "assistant"


def text_from_turn(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if not isinstance(turn, dict):
        return ""
    for key in ("content", "text", "utterance", "message", "query", "response", "value"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_turns(record: dict[str, Any]) -> list[DialogueTurn]:
    raw_turns = None
    for key in ("messages", "dialogue", "dialog", "turns", "utterances", "conversation", "session"):
        value = record.get(key)
        if isinstance(value, list):
            raw_turns = value
            break

    if raw_turns is None:
        query = record.get("query") or record.get("question") or record.get("user")
        response = record.get("response") or record.get("answer") or record.get("assistant")
        if isinstance(query, str) and isinstance(response, str):
            raw_turns = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]

    if not raw_turns:
        return []

    turns: list[DialogueTurn] = []
    for index, raw_turn in enumerate(raw_turns):
        content = text_from_turn(raw_turn)
        if not content:
            continue
        role_value = raw_turn.get("role") if isinstance(raw_turn, dict) else None
        role_value = role_value or (raw_turn.get("speaker") if isinstance(raw_turn, dict) else None)
        role = normalize_role(role_value, len(turns))
        if turns and turns[-1].role == role:
            turns[-1] = DialogueTurn(role=role, content=f"{turns[-1].content}\n{content}")
        else:
            turns.append(DialogueTurn(role=role, content=content))
    return turns


def infer_category(text: str) -> str:
    if any(term in text for term in ("退货", "退款", "七天", "售后")):
        return "refund"
    if any(term in text for term in ("物流", "快递", "发货", "派送")):
        return "logistics"
    if any(term in text for term in ("尺码", "尺寸", "大小", "颜色")):
        return "consulting"
    if any(term in text for term in ("投诉", "差评", "赔偿")):
        return "complaint"
    if any(term in text for term in ("手机号", "身份证", "验证码", "隐私")):
        return "safety"
    return "general"


def expected_keywords_for(text: str, category: str) -> list[str]:
    keywords = {
        "refund": ["订单", "退货", "售后"],
        "logistics": ["订单", "物流", "快递"],
        "consulting": ["商品", "尺码"],
        "complaint": ["抱歉", "订单", "处理"],
        "safety": ["隐私", "不能", "官方"],
        "general": ["订单", "问题"],
    }
    return [kw for kw in keywords[category] if kw in text or category != "general"][:3]


def refusal_expected(text: str) -> bool:
    return any(term in text for term in ("手机号", "身份证", "验证码", "删差评", "撤销差评", "诱导"))


def corrupt_response(response: str, category: str, rng: random.Random) -> str:
    strategies = [
        "terse",
        "vague",
        "overpromise",
        "privacy_leak",
    ]
    if category == "safety":
        strategy = "privacy_leak"
    else:
        strategy = rng.choice(strategies[:3])

    if strategy == "terse":
        return "可以的，你直接在订单里操作一下就行。"
    if strategy == "vague":
        return "这个情况不太确定，建议你再看看页面说明，不行就联系一下客服。"
    if strategy == "overpromise":
        return "没问题，我可以直接保证给你退款或赔偿，后续不用再提供其他信息。"
    return "可以，我帮你查询用户手机号、身份证或验证码后再处理。"


def build_examples_from_dialogue(
    record_id: str,
    turns: list[DialogueTurn],
    rng: random.Random,
    min_response_chars: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    history: list[str] = []
    pair_index = 0
    for turn in turns:
        if turn.role == "user":
            history.append(turn.content)
            history = history[-4:]
            continue
        if turn.role != "assistant" or not history:
            continue
        response = turn.content.strip()
        if len(response) < min_response_chars:
            continue
        user_text = "\n".join(history)
        category = infer_category(user_text + "\n" + response)
        example_id = f"{record_id}_{pair_index:03d}"
        pair_index += 1
        examples.append(
            {
                "id": example_id,
                "category": category,
                "user_messages": list(history),
                "chosen": response,
                "rejected": corrupt_response(response, category, rng),
                "expected_keywords": expected_keywords_for(user_text + "\n" + response, category),
                "refusal_expected": refusal_expected(user_text),
                "source": "jddc",
            }
        )
    return examples


def convert_jddc(
    input_path: str | Path,
    out_file: str | Path,
    max_dialogues: int | None = None,
    min_response_chars: int = 12,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    count = 0
    for index, record in enumerate(iter_raw_records(input_path)):
        if max_dialogues is not None and count >= max_dialogues:
            break
        turns = extract_turns(record)
        if not turns:
            continue
        record_id = str(record.get("id") or record.get("dialogue_id") or record.get("session_id") or index)
        examples.extend(build_examples_from_dialogue(record_id, turns, rng, min_response_chars))
        count += 1
    write_jsonl(out_file, examples)
    return len(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JDDC-style dialogues to preference pairs.")
    parser.add_argument("--input", required=True, help="JDDC JSON/JSONL file or directory.")
    parser.add_argument("--out-file", default="data/processed/preference_pairs.jsonl")
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--min-response-chars", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = convert_jddc(
        input_path=args.input,
        out_file=args.out_file,
        max_dialogues=args.max_dialogues,
        min_response_chars=args.min_response_chars,
        seed=args.seed,
    )
    print(f"preference_pairs: {args.out_file}")
    print(f"total_examples: {total}")


if __name__ == "__main__":
    main()
