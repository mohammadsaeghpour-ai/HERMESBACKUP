"""Smart Action Agent — Kill Zone + Final Entry"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind
from datetime import datetime, timezone, timedelta

tz = timezone(timedelta(hours=3, minutes=30))

class SmartActionAgent:
    name = "SmartAction"
    weight = 1.7
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        now = datetime.now(tz)
        hour = now.hour
        
        # Kill zones
        is_europe = 8 <= hour <= 14
        is_us = 14 <= hour <= 22
        in_killzone = is_europe or is_us
        
        price = df.iloc[-1]["close"]
        h20 = df["high"].iloc[-20:].max()
        l20 = df["low"].iloc[-20:].min()
        mid = (h20 + l20) / 2
        
        in_premium = price > mid
        in_discount = price < mid
        
        vr = ind.volume_ratio(df).iloc[-1]
        
        score = 0
        ev = []
        
        if not in_killzone:
            ev.append("NOT in kill zone (hour=%d)" % hour)
            return AgentOutput(name=self.name, direction="NO_TRADE", confidence=80,
                              score=0, weight=self.weight, evidence=ev)
        
        ev.append("Kill zone active (hour=%d)" % hour)
        
        if in_discount:
            score += 0.1
            ev.append("Discount zone")
        elif in_premium:
            score -= 0.1
            ev.append("Premium zone")
        
        if vr > 1.2:
            score *= 1.5
            ev.append("Volume confirmed (%.1fx)" % vr)
        
        d = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "NO_TRADE")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*300,
                          score=score, weight=self.weight, evidence=ev)
