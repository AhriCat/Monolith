from __future__ import annotations
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration

class VLMAdapter:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", alt_id: str = "lmms-lab/llava-onevision-qwen2-7b-ov", use_alt: bool = False):
        self.model_id = model_id
        self.alt_id = alt_id
        self.use_alt = use_alt
        self.tok = None
        self.proc = None
        self.vlm = None

    def load(self):
        if self.vlm is not None:
            return
        if self.use_alt:
            self.proc = AutoProcessor.from_pretrained(self.alt_id, trust_remote_code=True)
            self.vlm  = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.alt_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True
            )
        else:
            self.tok  = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.vlm  = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True
            )

    def qa(self, images: List[Union[str, "PIL.Image.Image"]], prompt: str) -> str:
        self.load()
        inputs = self.proc(text=[prompt], images=images, return_tensors="pt").to(self.vlm.device)
        out = self.vlm.generate(**inputs, max_new_tokens=512)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]
