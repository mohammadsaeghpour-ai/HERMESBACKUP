"""Whale Agent — Institutional Activity"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class WhaleAgent:
    name = "Whale"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        vr = ind.volume_ratio(df).iloc[-1]
        price_chg = abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
        
        o, h, l, c = df.iloc[-1]["open"], df.iloc[-1]["high"], df.iloc[-1]["low"], df.iloc[-1]["close"]
        rng = h - l if h != l else 1e-10
        wick = (h - max(o,c) + min(o,c) - l) / rng
        
        if vr > 2.0 and price_chg < 0.1:
            d = "NEUTRAL"
            score = 0
            conf = 80
            ev = ["Whale ABSORPTION detected (vol=%.1fx, chg=%.3f%%)" % (vr, price_chg)]
        elif vr > 1.5 and wick > 0.6:
            d = "NEUTRAL"
            score = -0.2
            conf = 60
            ev = ["Whale MANIPULATION detected (wick=%.1f%%)" % (wick*100)]
        else:
            d = "NEUTRAL"
            score = 0
            conf = 10
            ev = ["No whale activity"]
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev)
