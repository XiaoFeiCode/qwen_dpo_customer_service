# Qwen3-8B 电商客服偏好对齐系统

本项目面向中文电商客服多轮对话，基于 JDDC/JDDC-style 数据构建 SFT 样本和 DPO 偏好对，并使用 Qwen3-8B、LoRA/QLoRA、TRL 完成监督微调与偏好优化。

核心流程：

```text
JDDC 原始对话 -> 对话归一化 -> 自动构造 chosen/rejected -> SFT -> DPO -> 离线评估/API
```

## 数据集

仓库不分发 JDDC 原始数据。请从官方数据源或公开项目获取 JDDC/JDDC 2.1，并放到：

```text
data/jddc/raw/
```

转换器支持常见 JSON/JSONL 格式：

- `messages: [{"role": "user", "content": "..."}, ...]`
- `dialogue`、`turns`、`utterances`、`conversation`、`session`
- 单轮 `query/response`

参考资料：

- [JDDC](https://aclanthology.org/2020.lrec-1.58/)：中文电商客服多轮对话数据集，论文报告包含 100 万级多轮对话。
- [JDDC 2.1](https://github.com/hrlinlp/jddc2.1)：中文电商多模态对话数据集。
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)：Qwen3 系列 8.2B 参数模型。

## 自动偏好对构造

JDDC 原始数据通常只有客服对话，没有天然的 DPO `chosen/rejected` 标注。本项目使用弱监督方式自动构造偏好对：

1. 将真实客服回复作为 `chosen`。
2. 根据场景自动生成质量更差的 `rejected`。
3. 为每条样本保留 `category`、`expected_keywords`、`refusal_expected` 等字段，便于过滤和评估。

负样本构造策略：

- `terse`：回复过短，缺少流程说明。
- `vague`：表达模糊，没有可执行方案。
- `overpromise`：无依据承诺退款、赔偿或处理结果。
- `privacy_leak`：在手机号、身份证、验证码等隐私场景下给出不合规回复。

示例：

```json
{
  "prompt": "我这个衣服刚收到不想要了，可以退吗？包装还在。",
  "chosen": "理解你的情况。如果商品不影响二次销售且仍在平台售后期内，可以在订单页申请退货退款。请确认吊牌、包装和赠品是否齐全，并补充订单号。",
  "rejected": "可以的，你直接在订单里操作一下就行。"
}
```

这种方法可以快速得到大规模初始偏好数据。实际训练前可按类别抽样人工复核，并结合无效回复率、拒答准确率、关键词覆盖等指标过滤低质量 pair。

## 快速运行

安装基础环境：

```powershell
uv sync
```

将 JDDC-style 对话转换为偏好对：

```powershell
uv run python -m qwen_dpo_cs.jddc `
  --input data/jddc/raw `
  --out-file data/processed/preference_pairs.jsonl `
  --max-dialogues 50000
```

生成 SFT/DPO/评估数据：

```powershell
uv run python -m qwen_dpo_cs.build_dataset `
  --input data/processed/preference_pairs.jsonl `
  --out-dir data/processed
```

## 训练

安装训练依赖：

```powershell
uv sync --extra train
```

SFT：

```powershell
uv run python -m qwen_dpo_cs.training.sft_train `
  --model-name Qwen/Qwen3-8B `
  --train-file data/processed/sft_train.jsonl `
  --output-dir checkpoints/sft-lora
```

DPO：

```powershell
uv run python -m qwen_dpo_cs.training.dpo_train `
  --model-name Qwen/Qwen3-8B `
  --sft-adapter checkpoints/sft-lora `
  --train-file data/processed/dpo_train.jsonl `
  --output-dir checkpoints/dpo-lora `
  --beta 0.1
```

## 评估与服务

离线评估：

```powershell
uv run python -m qwen_dpo_cs.evaluation `
  --eval-file data/processed/eval.jsonl `
  --prediction-out output/eval/predictions.jsonl `
  --metrics-out output/eval/metrics.json
```

主要指标：

- `invalid_response_rate`：无效/敷衍回复比例。
- `refusal_accuracy`：隐私与安全边界场景的拒答准确率。
- `preference_pair_accuracy`：`chosen` 是否优于 `rejected`。
- `avg_keyword_recall`：订单号、物流、退款、平台规则等关键字段覆盖率。

启动 API：

```powershell
uv sync --extra api
$env:MODEL_PATH="Qwen/Qwen3-8B"
$env:ADAPTER_PATH="checkpoints/dpo-lora"
uv run uvicorn qwen_dpo_cs.api:app --host 127.0.0.1 --port 8000
```

## 开发检查

```powershell
uv run python -m qwen_dpo_cs.jddc --input tests/fixtures/jddc_sample.jsonl --out-file data/processed/preference_pairs.jsonl
uv run python -m qwen_dpo_cs.build_dataset --input data/processed/preference_pairs.jsonl --out-dir data/processed
uv run python -m qwen_dpo_cs.evaluation --eval-file data/processed/eval.jsonl --prediction-out output/eval/predictions.jsonl --metrics-out output/eval/metrics.json
uv run python -m unittest discover -s tests
```
