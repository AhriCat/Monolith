"""Sequence-level distillation scaffold: teacher → student.
Provide TEACHER_ID via env and a jsonl dataset of prompts.
"""
from __future__ import annotations
import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

STUDENT_ID = os.getenv("ONI_STUDENT_ID", "Qwen/Qwen2.5-7B-Instruct")
TEACHER_ID = os.getenv("ONI_TEACHER_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
DATA_PATH = os.getenv("ONI_DISTILL_DATA", "data/prompts.jsonl")


def generate_teacher_outputs():
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    ds = load_dataset("json", data_files=DATA_PATH)["train"]
    outs = []
    for ex in ds:
        prompt = teacher_tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        toks = teacher_tok(prompt, return_tensors="pt").to(teacher.device)
        gen = teacher.generate(**toks, max_new_tokens=1024, temperature=0.2, top_p=0.9)
        text = teacher_tok.decode(gen[0], skip_special_tokens=True)
        outs.append({"messages": ex["messages"], "response": text})
    with open("data/teacher_traces.jsonl", "w", encoding="utf-8") as f:
        for o in outs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def finetune_student():
    student_tok = AutoTokenizer.from_pretrained(STUDENT_ID, trust_remote_code=True)
    ds = load_dataset("json", data_files="data/teacher_traces.jsonl")["train"]

    def fmt(ex):
        prompt = student_tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        return {
            "input_ids": student_tok(prompt).input_ids,
            "labels": student_tok(ex["response"]).input_ids,
        }

    ds = ds.map(fmt, remove_columns=ds.column_names)
    student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    args = TrainingArguments(output_dir="checkpoints/distill", per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=3e-5, num_train_epochs=1, fp16=True)
    Trainer(model=student, args=args, train_dataset=ds).train()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_teacher_outputs()
    finetune_student()
