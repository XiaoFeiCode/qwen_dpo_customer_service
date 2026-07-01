# GitHub Upload Guide

## 方式一：命令行上传

```powershell
cd D:\桌面\面试\qwen-dpo-customer-service
git init -b main
git add .
git commit -m "init qwen dpo customer service alignment project"
git remote add origin https://github.com/XiaoFeiCode/qwen-dpo-customer-service.git
git push -u origin main
```

## 上传前检查

```powershell
uv sync
uv run python -m qwen_dpo_cs.build_dataset --input data/raw/customer_service_seed.jsonl --out-dir data/processed
uv run python -m qwen_dpo_cs.evaluation --eval-file data/processed/eval.jsonl --prediction-out output/eval/rule_predictions.jsonl --metrics-out output/eval/metrics.json
uv run python -m unittest discover -s tests
```

## 建议仓库描述

E-commerce customer service SFT + DPO preference alignment pipeline based on Qwen, LoRA/QLoRA, TRL and FastAPI.

## README 展示重点

- SFT + DPO 两阶段对齐链路
- chosen/rejected 偏好数据构建
- 拒答准确率、无效回复率、偏好对胜率等评估指标
- FastAPI 推理接口
