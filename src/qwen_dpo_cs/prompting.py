from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "你是电商平台客服助手。回答必须礼貌、准确、可执行；"
    "遇到订单、隐私、退款、售后等问题时先说明可处理范围，"
    "需要用户补充信息时只询问必要字段；"
    "对超出权限或违反规则的请求要明确拒绝并给出合规替代方案。"
)


def normalize_user_messages(raw_messages: list[str] | str) -> list[str]:
    if isinstance(raw_messages, str):
        return [raw_messages]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("user_messages must be a non-empty string or list[str]")
    return [str(message).strip() for message in raw_messages if str(message).strip()]


def build_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_message in normalize_user_messages(record["user_messages"]):
        messages.append({"role": "user", "content": user_message})
    return messages


def build_sft_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return build_prompt_messages(record) + [{"role": "assistant", "content": record["chosen"]}]


def format_chatml(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Small ChatML fallback used for dataset files and smoke tests.

    Training scripts prefer tokenizer.apply_chat_template when the selected model provides
    a native chat template. This function keeps local data inspectable and dependency-free.
    """

    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def to_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = build_sft_messages(record)
    return {
        "id": record["id"],
        "category": record.get("category", "unknown"),
        "messages": messages,
        "text": format_chatml(messages),
    }


def to_dpo_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = build_prompt_messages(record)
    chosen_messages = [{"role": "assistant", "content": record["chosen"]}]
    rejected_messages = [{"role": "assistant", "content": record["rejected"]}]
    return {
        "id": record["id"],
        "category": record.get("category", "unknown"),
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "prompt_text": format_chatml(prompt_messages, add_generation_prompt=True),
        "chosen_text": record["chosen"],
        "rejected_text": record["rejected"],
    }


def to_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "category": record.get("category", "unknown"),
        "prompt": build_prompt_messages(record),
        "expected_keywords": record.get("expected_keywords", []),
        "refusal_expected": bool(record.get("refusal_expected", False)),
        "chosen": record["chosen"],
        "rejected": record["rejected"],
    }
