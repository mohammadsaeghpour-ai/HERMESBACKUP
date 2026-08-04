"""Pattern Agent — Candlestick Patterns"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput

class PatternAgent:
    name = "Pattern"
    weight = 1.1
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 5:
            return AgentOutput(name=self.name, confidence=0)
        
        o, h, l, c = (df.iloc[-1]["open"], df.iloc[-1]["high"],
                       df.iloc[-1]["low"], df.iloc[-1]["close"])
        body = abs(c - o)
        rng = h - l if h != l else 1e-10
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        score = 0
        pattern = "none"
        
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            pattern = "hammer"
            score = 0.3
            d = "BUY"
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            pattern = "shooting_star"
            score = -0.3
            d = "SELL"
        elif body < rng * 0.1:
            pattern = "doji"
            score = 0
            d = "NEUTRAL"
        else:
            d = "NEUTRAL"
        
        conf = abs(score) * 100
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight,
                          evidence=["Pattern=%s" % pattern])
