"""Minimal SFT loop scaffold. Replace dataset loaders with your traces.
Assumes jsonl samples with {"messages": [...], "response": "..."}.
"""
from __future__ import annotations
import json, os
from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

MODEL_ID = os.getenv("ONI_SFT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH = os.getenv("ONI_SFT_DATA", "data/traces.jsonl")


def format_sample(tok, sample: Dict[str, Any]):
    prompt = tok.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
    return {"input_ids": tok(prompt).input_ids, "labels": tok(sample["response"]).input_ids}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ds = load_dataset("json", data_files=DATA_PATH)["train"]
    ds = ds.map(lambda ex: format_sample(tok, ex), remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    args = TrainingArguments(
        output_dir="checkpoints/sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        num_train_epochs=1,
        fp16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

if __name__ == "__main__":
    main()
