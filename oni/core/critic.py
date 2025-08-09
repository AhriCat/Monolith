from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CriticConfig:
    cost_weight: float = 0.1
    safety_weight: float = 0.4
    spec_weight: float = 0.5

class Critic:
    """Lightweight scorer for candidate plans/answers.
    Score = spec + safety + (1 - cost) with weights.
    """
    def __init__(self, cfg: CriticConfig = CriticConfig()):
        self.cfg = cfg

    def score_plan(self, plan: str, task: str, cost_estimate: float = 0.2) -> float:
        spec = self._spec(plan, task)
        safe = self._safety(plan)
        cost = max(0.0, min(1.0, cost_estimate))
        return (
            self.cfg.spec_weight * spec +
            self.cfg.safety_weight * safe +
            self.cfg.cost_weight * (1.0 - cost)
        )

    def _spec(self, plan: str, task: str) -> float:
        must = ["steps", "tools", "goal"] if task in {"plan", "vision", "asr"} else ["answer"]
        score = sum(1 for m in must if m in plan.lower()) / len(must)
        return float(score)

    def _safety(self, plan: str) -> float:
        bad = ["exploit", "jailbreak", "bypass", "bomb", "malware"]
        return 0.0 if any(b in plan.lower() for b in bad) else 1.0
