"""SMC Agent — Simplified Order Blocks"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class SMCAgent:
    name = "SMC"
    weight = 1.0
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        price = df.iloc[-1]["close"]
        
        # Simple: find recent swing high/low as OB zones
        highs, lows = ind.find_swings(df, lookback=3)
        
        score = 0
        ev = []
        
        # Bullish OB: last significant low before a move up
        if lows:
            nearest_low = lows[-1][1]
            dist_low = (price - nearest_low) / price * 100
            if dist_low < 0.3 and dist_low > 0:
                score += 0.2
                ev.append("Near swing low $%.1f (%.1f%%)" % (nearest_low, dist_low))
        
        # Bearish OB: last significant high before a move down
        if highs:
            nearest_high = highs[-1][1]
            dist_high = (nearest_high - price) / price * 100
            if dist_high < 0.3 and dist_high > 0:
                score -= 0.2
                ev.append("Near swing high $%.1f (%.1f%%)" % (nearest_high, dist_high))
        
        # Volume confirmation
        vr = ind.volume_ratio(df).iloc[-1]
        if vr > 1.3:
            score *= 1.5
            ev.append("Volume confirmed")
        
        d = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "NEUTRAL")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev or ["No SMC setup"])
