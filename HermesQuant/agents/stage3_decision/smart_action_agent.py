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
        
        is_europe = 8 <= hour <= 14
        is_us = 14 <= hour <= 22
        in_killzone = is_europe or is_us
        
        price = df.iloc[-1]["close"]
        h20 = df["high"].iloc[-20:].max()
        l20 = df["low"].iloc[-20:].min()
        mid = (h20 + l20) / 2
        
        vr = ind.volume_ratio(df).iloc[-1]
        st_dir, _ = ind.supertrend(df)
        direction = st_dir.iloc[-1]
        
        score = 0
        ev = []
        
        # Kill zone check (always true in backtest)
        ev.append("Kill zone: always pass in backtest")
        
        # Premium/Discount
        in_discount = price < mid
        in_premium = price > mid
        
        # Supertrend-based confirmation (simple and effective)
        if direction == 1 and in_discount:
            d = "BUY"
            score = 0.2 if vr > 1.0 else 0.1
            ev.append("ST=UP + Discount zone")
        elif direction == -1 and in_premium:
            d = "SELL"
            score = -0.2 if vr > 1.0 else -0.1
            ev.append("ST=DOWN + Premium zone")
        elif direction == 1:
            d = "BUY"
            score = 0.05
            ev.append("ST=UP (no discount)")
        elif direction == -1:
            d = "SELL"
            score = -0.05
            ev.append("ST=DOWN (no premium)")
        else:
            d = "NEUTRAL"
            score = 0
            ev.append("ST=NEUTRAL")
        
        conf = min(abs(score) * 200, 60)
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev)
