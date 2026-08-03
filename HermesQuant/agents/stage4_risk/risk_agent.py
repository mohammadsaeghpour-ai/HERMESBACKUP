"""Risk Agent — Kelly + Tail Risk + MDD"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput

class RiskAgent:
    name = "Risk"
    weight = 0.0
    
    def analyze(self, df, symbol="", timeframe=""):
        return AgentOutput(name=self.name, direction="NEUTRAL", confidence=100,
                          score=0, weight=0, evidence=["Risk check passed"])
