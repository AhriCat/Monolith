from __future__ import annotations
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .base_adapter import BaseAdapter

class QwenTextAdapter(BaseAdapter):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", load_4bit: bool = False):
        self.model_id = model_id
        self.load_4bit = load_4bit
        self.tok = None
        self.lm = None

    def load(self):
        if self.lm is not None:
            return
        kwargs = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True)
        if self.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.lm  = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def infer(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024, temperature: float = 0.2, top_p: float = 0.95) -> str:
        self.load()
        prompt = self.apply_chat_template(self.tok, messages)
        inputs = self.tok(prompt, return_tensors="pt").to(self.lm.device)
        out = self.lm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        return self.tok.decode(out[0], skip_special_tokens=True)
