"""Math Brain Agent — Fibonacci + Pivots + ATR"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind
import math

class MathBrainAgent:
    name = "MathBrain"
    weight = 1.4
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        price = df.iloc[-1]["close"]
        h20 = df["high"].iloc[-20:].max()
        l20 = df["low"].iloc[-20:].min()
        rng = h20 - l20
        atr_v = ind.atr(df).iloc[-1]
        
        # Fibonacci levels
        fib_618 = l20 + rng * 0.618
        fib_500 = l20 + rng * 0.500
        fib_382 = l20 + rng * 0.382
        
        # Pivot points
        prev_h = df.iloc[-2]["high"]
        prev_l = df.iloc[-2]["low"]
        prev_c = df.iloc[-2]["close"]
        pp = (prev_h + prev_l + prev_c) / 3
        r1 = 2 * pp - prev_l
        s1 = 2 * pp - prev_h
        
        score = 0
        ev = []
        
        # Price near support
        if abs(price - s1) / price < 0.002:
            score += 0.2
            ev.append("Near S1 pivot $%.1f" % s1)
        if abs(price - fib_618) / price < 0.003:
            score += 0.15
            ev.append("Near Fib 61.8%% $%.1f" % fib_618)
        if abs(price - fib_382) / price < 0.003:
            score -= 0.15
            ev.append("Near Fib 38.2%% $%.1f" % fib_382)
        if abs(price - r1) / price < 0.002:
            score -= 0.2
            ev.append("Near R1 pivot $%.1f" % r1)
        
        d = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "NEUTRAL")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev or ["No math setup"])
