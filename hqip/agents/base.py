"""
Base Agent - All agents inherit from this.
Standardized output format for every agent.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class AgentOutput:
    agent_name: str
    direction: str = "NEUTRAL"  # BUY / SELL / NEUTRAL
    confidence: float = 0.0     # 0-100
    score: float = 0.0          # -1.0 to +1.0
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    data: dict = field(default_factory=dict)
    weight: float = 1.0
    error: str = ""

    def to_dict(self):
        return {
            "agent": self.agent_name,
            "direction": self.direction,
            "confidence": round(self.confidence, 1),
            "score": round(self.score, 3),
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "weight": self.weight,
        }

class BaseAgent:
    name = "BaseAgent"
    weight = 1.0

    def analyze(self, df, symbol="", timeframe="", **kwargs) -> AgentOutput:
        raise NotImplementedError

    def _out(self, **kw):
        return AgentOutput(agent_name=self.name, weight=self.weight, **kw)
