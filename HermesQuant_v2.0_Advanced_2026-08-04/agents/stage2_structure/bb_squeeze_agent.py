"""Bollinger Squeeze Agent — Volatility Breakout"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class BBSqueezeAgent:
    """
    Detects Bollinger Band Squeeze:
    - Bands contract → squeeze building
    - Bands expand + price breaks → breakout
    
    Squeeze = low volatility before big move
    """
    name = "BB_Squeeze"
    weight = 1.1
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        upper, mid, lower = ind.bollinger(df)
        price = df["close"]
        
        # BB Width
        width = (upper - lower) / mid
        width_avg = width.iloc[-20:].mean()
        width_ratio = width.iloc[-1] / (width_avg + 1e-10)
        
        # Current position
        bb_pos = (price.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
        
        # Squeeze detection
        is_squeeze = width_ratio < 0.7  # Bands are narrow
        
        # Breakout detection
        prev_close = price.iloc[-2]
        if prev_close < upper.iloc[-2] and price.iloc[-1] > upper.iloc[-1]:
            # Bullish breakout
            score = 0.25 if is_squeeze else 0.1
            d = "BUY"
            ev = ["BB Bullish breakout (squeeze=%s)" % is_squeeze]
        elif prev_close > lower.iloc[-2] and price.iloc[-1] < lower.iloc[-1]:
            # Bearish breakout
            score = -0.25 if is_squeeze else -0.1
            d = "SELL"
            ev = ["BB Bearish breakout (squeeze=%s)" % is_squeeze]
        elif is_squeeze and bb_pos < 0.3:
            score = 0.1
            d = "BUY"
            ev = ["BB Squeeze + low position"]
        elif is_squeeze and bb_pos > 0.7:
            score = -0.1
            d = "SELL"
            ev = ["BB Squeeze + high position"]
        else:
            score = 0
            d = "NEUTRAL"
            ev = ["No BB setup"]
        
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev)
