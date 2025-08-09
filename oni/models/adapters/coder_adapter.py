from __future__ import annotations
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class CoderAdapter:
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-Coder-V2-Instruct", load_4bit: bool = False):
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

    def code(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.load()
        tpl = f"### Instruction:\n{prompt}\n\n### Response:\n"
        toks = self.tok(tpl, return_tensors="pt").to(self.lm.device)
        out = self.lm.generate(**toks, max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.95)
        return self.tok.decode(out[0], skip_special_tokens=True)
