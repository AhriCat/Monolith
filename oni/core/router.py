from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .schema import ChatMessage, ToolSpec
from .critic import Critic

@dataclass
class RouterConfig:
    topk: int = 1  # simple single-choice for now

class Router:
    def __init__(self, critic: Critic = None, cfg: RouterConfig = RouterConfig()):
        self.critic = critic or Critic()
        self.cfg = cfg

    def choose(self, task: str, context: List[ChatMessage], tool_specs: List[ToolSpec]) -> Dict[str, Any]:
        """Return routing decision; extend with priors/history later."""
        # trivial heuristic routing
        if task == "code":
            return {"target": "coder"}
        if task == "vision":
            return {"target": "vlm"}
        if task == "asr":
            return {"target": "asr"}
        if task == "plan" and tool_specs:
            return {"target": "functionary"}
        return {"target": "text"}
