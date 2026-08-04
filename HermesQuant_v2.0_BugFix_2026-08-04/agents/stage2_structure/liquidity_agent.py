"""Liquidity Agent — Fixed Version"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class LiquidityAgent:
    """
    Detects Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL).
    Key insight: Price moves TO liquidity pools then reverses.
    """
    name = "Liquidity"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        highs, lows = ind.find_swings(df, lookback=3)
        price = df.iloc[-1]["close"]
        vr = ind.volume_ratio(df).iloc[-1]
        
        score = 0
        ev = []
        
        # Find nearest BSL (above) and SSL (below)
        bsl = [h[1] for h in highs if h[1] > price * 1.001]
        ssl = [l[1] for l in lows if l[1] < price * 0.999]
        
        bsl_nearest = min(bsl) if bsl else None
        ssl_nearest = max(ssl) if ssl else None
        
        # Key insight: Price moves TOWARD liquidity, then reverses
        # If price is approaching BSL → likely to sweep then reverse DOWN
        # If price is approaching SSL → likely to sweep then reverse UP
        
        if bsl_nearest:
            dist_bsl = (bsl_nearest - price) / price * 100
            if dist_bsl < 0.5:  # Close to BSL
                # Price might sweep BSL then reverse
                score = -0.2 if vr > 1.2 else -0.1
                ev.append("Near BSL $%.1f (%.1f%%) — possible sweep + reverse" % (bsl_nearest, dist_bsl))
        
        if ssl_nearest:
            dist_ssl = (price - ssl_nearest) / price * 100
            if dist_ssl < 0.5:  # Close to SSL
                # Price might sweep SSL then reverse
                score = 0.2 if vr > 1.2 else 0.1
                ev.append("Near SSL $%.1f (%.1f%%) — possible sweep + bounce" % (ssl_nearest, dist_ssl))
        
        # If both exist, use the closer one
        if bsl_nearest and ssl_nearest:
            dist_b = (bsl_nearest - price) / price
            dist_s = (price - ssl_nearest) / price
            if dist_b < dist_s:
                ev.append("BSL is closer — bias SELL")
                score = min(score, -0.1)
            else:
                ev.append("SSL is closer — bias BUY")
                score = max(score, 0.1)
        
        d = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "NEUTRAL")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev or ["No liquidity levels"])
