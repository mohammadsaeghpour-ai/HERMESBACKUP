"""Volatility Agent — Regime Detection"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class VolatilityAgent:
    name = "Volatility"
    weight = 1.0
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        atr_v = ind.atr(df)
        current = atr_v.iloc[-1]
        avg = atr_v.iloc[-20:].mean()
        ratio = current / (avg + 1e-10)
        
        if ratio < 0.8:
            regime = "CALM"
            score = 0
        elif ratio < 1.5:
            regime = "NORMAL"
            score = 0
        elif ratio < 3.0:
            regime = "CHAOTIC"
            score = -0.3
        else:
            regime = "CRISIS"
            score = -0.5
        
        return AgentOutput(name=self.name, direction=regime, confidence=min(ratio*50,100),
                          score=score, weight=self.weight,
                          metadata={"regime": regime, "atr_ratio": ratio},
                          evidence=["ATR ratio=%.2f, regime=%s" % (ratio, regime)])
