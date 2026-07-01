from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatRequest:
    messages: list[str]
    category: str | None = None


class RuleBasedCustomerServiceResponder:
    """Dependency-free responder for smoke tests and API demo mode."""

    def generate(self, messages: list[str], category: str | None = None) -> str:
        latest = messages[-1] if messages else ""
        text = latest.lower()
        if any(word in latest for word in ("密码", "身份证", "银行卡", "验证码")):
            return (
                "理解你的诉求，但涉及密码、验证码、身份证或银行卡等隐私信息时，我不能代为获取或展示。"
                "建议你通过平台官方安全入口完成验证；如果是订单售后问题，可以补充订单号和问题截图，我会继续协助核实。"
            )
        if any(word in latest for word in ("诱导", "撤销差评", "删差评", "不撤差评")):
            return (
                "抱歉，我不能帮助生成诱导或违规干预评价的内容。"
                "你可以合规地向买家说明售后处理方案，例如主动道歉、解释问题原因、提供退换或补偿选项，"
                "并邀请买家在问题解决后自愿更新评价。"
            )
        if any(word in latest for word in ("退款", "退货", "不想要", "七天")):
            return (
                "理解你的情况。请先确认商品是否影响二次销售，并补充订单号、商品状态和问题截图。"
                "如果符合平台售后规则，可以在订单页申请退货退款；若已超时，我会建议你提交原因说明后转人工复核。"
            )
        if any(word in latest for word in ("尺码", "大小", "穿", "身高", "体重")):
            return (
                "可以帮你参考尺码。请补充身高、体重、平时穿着尺码和偏好的版型。"
                "如果商品页有尺码表，建议优先按胸围、肩宽或腰围选择，介于两个尺码之间时按穿着偏好调整。"
            )
        if any(word in latest for word in ("物流", "快递", "没到", "发货")):
            return (
                "抱歉让你久等了。请补充订单号或物流单号，我可以根据当前物流状态判断是未揽收、运输延迟还是派送异常。"
                "如果超过承诺时效，可以协助你申请催发或售后处理。"
            )
        if any(word in latest for word in ("投诉", "差评", "生气", "赔偿")):
            return (
                "非常抱歉给你带来不好的体验。我会先帮你拆分问题并确认可处理范围：请补充订单号、问题截图和期望方案。"
                "如果属于商品质量或服务异常，会优先按平台规则推进退换、补发或补偿申请。"
            )
        return (
            "我可以协助处理。为了更准确地判断，请补充订单号、商品名称、遇到的问题和你期望的处理方式。"
            "我会根据平台规则给出可执行的售后或咨询建议。"
        )


class TransformersResponder:
    """Optional real model responder.

    This class imports heavy ML dependencies lazily so the repository can still run
    data and API smoke tests without installing the training stack.
    """

    def __init__(self, model_path: str, adapter_path: str | None = None, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, messages: list[str], category: str | None = None) -> str:
        chat = [{"role": "user", "content": message} for message in messages]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response_ids = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
