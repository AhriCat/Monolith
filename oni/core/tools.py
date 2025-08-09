from __future__ import annotations
from typing import Dict, Callable, Any
from .schema import ToolSpec

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, fn: Callable[[Dict[str, Any]], Any]):
        self._tools[spec.name] = fn
        self._specs[spec.name] = spec

    def call(self, name: str, arguments: Dict[str, Any]):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](arguments)

    def specs(self):
        return list(self._specs.values())
