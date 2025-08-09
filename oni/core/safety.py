from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SafetyConfig:
    allow_tools: bool = True
    max_steps: int = 8
    refusal_threshold: float = 0.85  # if risk >= threshold → refuse

class SafetyPolicy:
    def __init__(self, cfg: SafetyConfig = SafetyConfig()):
        self.cfg = cfg

    def assess(self, prompt: str) -> Tuple[bool, str]:
        """Return (allowed, reason). Toy heuristic; replace with classifier later."""
        lower = prompt.lower()
        banned = ["make a bomb", "credit card", "csam", "dox"]
        for b in banned:
            if b in lower:
                return False, f"Blocked by policy: contains '{b}'"
        return True, "ok"
