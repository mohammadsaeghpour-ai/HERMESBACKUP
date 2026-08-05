"""
Smart Action Agent v2 — Enhanced with confluence scoring
Second best agent: +0.11 correlation
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind


class SmartActionAgentV2:
    """
    Enhanced smart action with confluence scoring:
    1. Trend alignment (EMA)
    2. Support/Resistance levels
    3. Volume profile
    4. Risk/Reward calculation
    """
    name = "SmartAction"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0, evidence=["Not enough data"])
        
        score = 0
        ev = []
        confluence = 0
        
        # EMA alignment
        ema8 = ind.ema(df["close"], 8).iloc[-1]
        ema20 = ind.ema(df["close"], 20).iloc[-1]
        ema50 = ind.ema(df["close"], 50).iloc[-1] if len(df) > 50 else ema20
        price = df["close"].iloc[-1]
        
        if ema8 > ema20 > ema50:
            score += 0.3
            confluence += 1
            ev.append("Strong uptrend (EMA8>20>50)")
        elif ema8 < ema20 < ema50:
            score -= 0.3
            confluence += 1
            ev.append("Strong downtrend (EMA8<20<50)")
        
        # Supertrend
        st_dir, st_val = ind.supertrend(df)
        if st_dir.iloc[-1] == 1:
            score += 0.2
            confluence += 1
            ev.append("Supertrend bullish")
        else:
            score -= 0.2
            confluence += 1
            ev.append("Supertrend bearish")
        
        # Bollinger position
        upper, mid, lower = ind.bollinger(df)
        bb_pos = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
        
        if bb_pos < 0.2:
            score += 0.2
            ev.append("Near lower BB (discount)")
        elif bb_pos > 0.8:
            score -= 0.2
            ev.append("Near upper BB (premium)")
        
        # Volume confirmation
        vol_ratio = ind.volume_ratio(df).iloc[-1]
        if vol_ratio > 1.2:
            confluence += 1
            ev.append("Volume confirms (%.1fx)" % vol_ratio)
        
        # ADX trend strength
        adx_v, dip, dim = ind.adx(df)
        if adx_v.iloc[-1] > 25:
            confluence += 1
            ev.append("ADX strong: %.1f" % adx_v.iloc[-1])
        
        # R:R calculation
        atr = ind.atr(df, 14).iloc[-1]
        if score > 0:
            tp = price + 2 * atr
            sl = price - 1 * atr
        else:
            tp = price - 2 * atr
            sl = price + 1 * atr
        
        rr = abs(tp - price) / abs(sl - price + 1e-10)
        
        # Direction
        if score > 0.1:
            d = "BUY"
        elif score < -0.1:
            d = "SELL"
        else:
            d = "NEUTRAL"
        
        # Confidence based on confluence
        conf = min(confluence * 25, 100)
        
        ev.append("Confluence: %d/4 | R:R: 1:%.1f" % (confluence, rr))
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev)
