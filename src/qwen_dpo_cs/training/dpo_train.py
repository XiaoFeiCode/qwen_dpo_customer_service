from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO preference optimization.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--sft-adapter", default="", help="Optional SFT LoRA adapter path.")
    parser.add_argument("--train-file", default="data/processed/dpo_train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/dpo-lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    raw_dataset = load_dataset("json", data_files=args.train_file, split="train")

    def to_trl_columns(example):
        return {
            "prompt": example["prompt_text"],
            "chosen": example["chosen_text"],
            "rejected": example["rejected_text"],
        }

    keep_columns = ["prompt", "chosen", "rejected"]
    dataset = raw_dataset.map(to_trl_columns, remove_columns=raw_dataset.column_names)
    dataset = dataset.select_columns(keep_columns)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=5,
        save_steps=50,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
