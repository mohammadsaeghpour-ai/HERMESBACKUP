"""RSI Divergence Agent — replaces broken SMC"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class RSIDivergenceAgent:
    """
    Detects Bullish/Bearish RSI Divergence:
    - Bullish: Price makes lower low, RSI makes higher low → BUY
    - Bearish: Price makes higher high, RSI makes lower high → SELL
    
    This is one of the most reliable reversal signals.
    """
    name = "RSI_Divergence"
    weight = 1.2
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        rsi_v = ind.rsi(df)
        price = df["close"]
        
        # Find swing points
        highs_idx, lows_idx = [], []
        for i in range(3, len(df) - 3):
            if price.iloc[i] == price.iloc[i-3:i+4].max():
                highs_idx.append(i)
            if price.iloc[i] == price.iloc[i-3:i+4].min():
                lows_idx.append(i)
        
        score = 0
        ev = []
        
        # Bullish Divergence: lower low in price, higher low in RSI
        if len(lows_idx) >= 2:
            l1, l2 = lows_idx[-2], lows_idx[-1]
            if price.iloc[l2] < price.iloc[l1] and rsi_v.iloc[l2] > rsi_v.iloc[l1]:
                score = 0.3
                ev.append("Bullish RSI divergence (price ↓, RSI ↑)")
        
        # Bearish Divergence: higher high in price, lower high in RSI
        if len(highs_idx) >= 2:
            h1, h2 = highs_idx[-2], highs_idx[-1]
            if price.iloc[h2] > price.iloc[h1] and rsi_v.iloc[h2] < rsi_v.iloc[h1]:
                score = -0.3
                ev.append("Bearish RSI divergence (price ↑, RSI ↓)")
        
        d = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "NEUTRAL")
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                          score=score, weight=self.weight, evidence=ev or ["No divergence"])
