from __future__ import annotations
import json
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .base_adapter import BaseAdapter

class FunctionaryAdapter(BaseAdapter):
    def __init__(self, model_id: str = "meetkai/functionary-small-v3.2", load_4bit: bool = False):
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

    def plan(self, messages: List[Dict[str, str]], tool_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.load()
        sys = {
            "role": "system",
            "content": "Emit a single JSON: {type:'tool_call', name:'<tool>', arguments:{...}} or {type:'final', content:'...'}\n" +
                       "Tools: " + json.dumps(tool_specs, ensure_ascii=False)
        }
        prompt = self.apply_chat_template(self.tok, [sys] + messages)
        toks = self.tok(prompt, return_tensors="pt").to(self.lm.device)
        out = self.lm.generate(**toks, max_new_tokens=512, temperature=0.1, top_p=0.9)
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        try:
            obj_start = txt.rfind("{")
            return json.loads(txt[obj_start:])
        except Exception:
            return {"type": "final", "content": txt}
