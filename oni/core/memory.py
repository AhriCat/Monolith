from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import math

@dataclass
class MemoryConfig:
    ttl_seconds: int = 60 * 60 * 24  # 1 day default
    max_items: int = 2048
    namespaces: Tuple[str, ...] = ("global",)

class Memory:
    """Simple in-memory store with TTL + namespacing + cosine retrieval hook.
    Plug your own vectorizer via set_embedder(callable: List[str] -> List[List[float]]).
    """
    def __init__(self, cfg: MemoryConfig = MemoryConfig()):
        self.cfg = cfg
        self._store: Dict[str, List[Dict[str, Any]]] = {ns: [] for ns in cfg.namespaces}
        self._embedder = None

    def set_embedder(self, fn):
        self._embedder = fn

    def upsert(self, text: str, namespace: str = "global", meta: Optional[Dict[str, Any]] = None):
        now = time.time()
        item = {"text": text, "ts": now, "meta": meta or {}, "vec": None}
        self._store.setdefault(namespace, []).append(item)
        # prune
        self._store[namespace] = self._prune(self._store[namespace])

    def search(self, query: str, namespace: str = "global", k: int = 5) -> List[Dict[str, Any]]:
        self._expire(namespace)
        items = self._store.get(namespace, [])
        if not items:
            return []
        if self._embedder is None:
            # lexical fallback: simple substring ranking
            scored = [(i, 1.0 if query.lower() in i["text"].lower() else 0.0) for i in items]
            return [i for i, s in sorted(scored, key=lambda x: x[1], reverse=True)[:k] if s > 0]
        texts = [i["text"] for i in items]
        qv = self._embedder([query])[0]
        vs = self._embedder(texts)
        sims = [self._cos(qv, v) for v in vs]
        ranked = sorted(zip(items, sims), key=lambda x: x[1], reverse=True)[:k]
        return [i for i, _ in ranked]

    # ----- helpers -----
    def _prune(self, arr: List[Dict[str, Any]]):
        if len(arr) <= self.cfg.max_items:
            return arr
        return sorted(arr, key=lambda x: x["ts"], reverse=True)[: self.cfg.max_items]

    def _expire(self, namespace: str):
        now = time.time()
        ttl = self.cfg.ttl_seconds
        self._store[namespace] = [i for i in self._store.get(namespace, []) if now - i["ts"] <= ttl]

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        num = sum(x*y for x, y in zip(a, b))
        da = math.sqrt(sum(x*x for x in a))
        db = math.sqrt(sum(y*y for y in b))
        return 0.0 if da == 0 or db == 0 else num / (da * db)
