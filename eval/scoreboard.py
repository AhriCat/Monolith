from __future__ import annotations
import json
from statistics import mean

class Scoreboard:
    def __init__(self):
        self.rows = []

    def add(self, name: str, quality: float, cost: float, latency_ms: float, refusal: float):
        self.rows.append({"name": name, "quality": quality, "cost": cost, "latency_ms": latency_ms, "refusal": refusal})

    def summary(self):
        return {
            "n": len(self.rows),
            "quality": mean(r["quality"] for r in self.rows) if self.rows else 0,
            "cost": mean(r["cost"] for r in self.rows) if self.rows else 0,
            "latency_ms": mean(r["latency_ms"] for r in self.rows) if self.rows else 0,
            "refusal": mean(r["refusal"] for r in self.rows) if self.rows else 0,
        }

if __name__ == "__main__":
    sb = Scoreboard()
    sb.add("baseline", 0.62, 0.12, 1800, 0.03)
    print(json.dumps(sb.summary(), indent=2))
