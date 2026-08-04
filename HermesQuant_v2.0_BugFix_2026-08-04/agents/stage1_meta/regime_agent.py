"""Regime Agent — Market State Detection"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class RegimeAgent:
    name = "Regime"
    weight = 1.0
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=100, direction="NEUTRAL")
        
        atr_v = ind.atr(df)
        ratio = atr_v.iloc[-1] / (atr_v.iloc[-20:].mean() + 1e-10)
        vr = ind.volume_ratio(df).iloc[-1]
        _, st = ind.supertrend(df)
        trend_strength = abs(df["close"].iloc[-1] - st.iloc[-1]) / df["close"].iloc[-1] * 100
        
        if ratio > 3.0:
            regime = "CRISIS"
            score = -0.5
            d = "NEUTRAL"
        elif ratio > 1.5 or vr > 2.0:
            regime = "CHAOTIC"
            score = -0.2
            d = "NEUTRAL"
        elif ratio < 0.8 and vr < 0.8:
            regime = "CALM"
            score = 0
            d = "NEUTRAL"
        else:
            regime = "NORMAL"
            score = 0
            d = "NEUTRAL"
        
        return AgentOutput(name=self.name, direction=d, confidence=100,
                          score=score, weight=self.weight,
                          metadata={"regime": regime, "atr_ratio": ratio},
                          evidence=["Regime=%s ATR_ratio=%.2f" % (regime, ratio)])
