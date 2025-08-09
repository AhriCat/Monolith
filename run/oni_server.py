from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Use the monolith for now; you can later swap to adapters/Router.
from main import ONIMonolith, MonolithConfig

app = FastAPI()
engine = ONIMonolith(MonolithConfig())

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat")
async def chat(payload: ChatRequest):
    out = engine.chat([m.dict() for m in payload.messages])
    return {"output": out}
