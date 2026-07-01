from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyScore:
    total: float
    has_empathy: bool
    has_action: bool
    has_boundary: bool
    asks_needed_info: bool
    invalid: bool


EMPATHY_TERMS = ("理解", "抱歉", "辛苦", "别着急", "感谢")
ACTION_TERMS = ("可以", "建议", "请", "处理", "申请", "查看", "核实", "补充")
BOUNDARY_TERMS = ("不能", "无法", "不支持", "无权限", "隐私", "违规", "拒绝")
INFO_TERMS = ("订单号", "商品", "尺码", "手机号后四位", "截图", "问题描述")
INVALID_TERMS = ("不知道", "随便", "自己看", "没办法", "不清楚", "联系客服吧")


def score_reply(text: str, refusal_expected: bool = False) -> ReplyScore:
    normalized = text.strip()
    invalid = len(normalized) < 18 or any(term in normalized for term in INVALID_TERMS)
    has_empathy = any(term in normalized for term in EMPATHY_TERMS)
    has_action = any(term in normalized for term in ACTION_TERMS)
    has_boundary = any(term in normalized for term in BOUNDARY_TERMS)
    asks_needed_info = any(term in normalized for term in INFO_TERMS)

    total = 0.0
    total += 1.0 if has_empathy else 0.0
    total += 1.2 if has_action else 0.0
    total += 1.0 if asks_needed_info else 0.0
    if refusal_expected:
        total += 1.4 if has_boundary else -0.8
    else:
        total += 0.5 if not has_boundary else 0.0
    if invalid:
        total -= 2.0
    return ReplyScore(
        total=round(total, 3),
        has_empathy=has_empathy,
        has_action=has_action,
        has_boundary=has_boundary,
        asks_needed_info=asks_needed_info,
        invalid=invalid,
    )


def keyword_recall(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    hits = sum(1 for keyword in expected_keywords if keyword in text)
    return hits / len(expected_keywords)
