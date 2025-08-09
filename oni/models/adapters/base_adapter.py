from __future__ import annotations
from typing import Any, Dict, List

class BaseAdapter:
    def load(self):
        raise NotImplementedError

    def infer(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    # Optional helper for chat-format models
    @staticmethod
    def apply_chat_template(tok, messages: List[Dict[str, str]]):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            lines = []
            for m in messages:
                lines.append(f"[{m['role'].upper()}]: {m['content']}")
            lines.append("[ASSISTANT]:")
            return "\n".join(lines)
