"""Momentum Agent — RSI + MACD"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class MomentumAgent:
    name = "Momentum"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        rsi_v = ind.rsi(df).iloc[-1]
        _, _, hist = ind.macd(df)
        mh = hist.iloc[-1]
        mh_prev = hist.iloc[-2]
        
        score = 0
        evidence = []
        
        if rsi_v > 50 and mh > 0:
            d = "BUY"
            score = (rsi_v - 50) / 50
            if mh > mh_prev: score += 0.2
        elif rsi_v < 50 and mh < 0:
            d = "SELL"
            score = -(50 - rsi_v) / 50
            if mh < mh_prev: score -= 0.2
        else:
            d = "NEUTRAL"
        
        conf = abs(score) * 100
        evidence = ["RSI=%.1f" % rsi_v, "MACD_H=%.2f" % mh]
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=evidence)
