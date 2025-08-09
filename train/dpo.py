"""Skeleton for DPO/ORPO style preference training.
Expects jsonl with {"prompt": [...], "chosen": str, "rejected": str}.
"""
from __future__ import annotations
import os
from datasets import load_dataset
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

MODEL_ID = os.getenv("ONI_DPO_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH = os.getenv("ONI_DPO_DATA", "data/prefs.jsonl")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ds = load_dataset("json", data_files=DATA_PATH)["train"]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    args = TrainingArguments(
        output_dir="checkpoints/dpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=500,
        num_train_epochs=1,
        fp16=True,
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=args,
        beta=0.1,
        train_dataset=ds,
        tokenizer=tok,
        max_length=2048,
        max_target_length=1024,
        prompt_text_field="prompt",
        chosen_response_field="chosen",
        rejected_response_field="rejected",
    )
    trainer.train()

if __name__ == "__main__":
    main()
