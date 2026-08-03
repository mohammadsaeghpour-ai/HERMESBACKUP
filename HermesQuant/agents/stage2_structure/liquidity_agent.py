"""Liquidity Agent — BSL/SSL Detection"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class LiquidityAgent:
    name = "Liquidity"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        highs, lows = ind.find_swings(df, lookback=3)
        price = df.iloc[-1]["close"]
        
        bsl = [h[1] for h in highs if h[1] > price]
        ssl = [l[1] for l in lows if l[1] < price]
        
        bsl_nearest = min(bsl) if bsl else None
        ssl_nearest = max(ssl) if ssl else None
        
        score = 0
        ev = []
        
        if bsl_nearest:
            dist = (bsl_nearest - price) / price * 100
            ev.append("BSL at $%.1f (%.1f%% above)" % (bsl_nearest, dist))
            if dist < 0.3:
                score -= 0.2
                ev.append("Price near BSL — possible sweep")
        
        if ssl_nearest:
            dist = (price - ssl_nearest) / price * 100
            ev.append("SSL at $%.1f (%.1f%% below)" % (ssl_nearest, dist))
            if dist < 0.3:
                score += 0.2
                ev.append("Price near SSL — possible bounce")
        
        d = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "NEUTRAL")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev or ["No liquidity levels"])
