from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]

@dataclass
class ToolCallResult:
    name: str
    arguments: Dict[str, Any]
    result: Any

@dataclass
class ONIRequest:
    task: str  # "chat" | "plan" | "code" | "vision" | "asr" | "embed"
    messages: List[ChatMessage] = field(default_factory=list)
    tools: List[ToolSpec] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    audio: Optional[str] = None
    embed_model: str = "bge"
    params: Dict[str, Any] = field(default_factory=dict)
