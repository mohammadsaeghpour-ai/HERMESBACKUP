"""
Momentum Agent v2 — Improved with multi-indicator confirmation
Best agent: +0.19 correlation with actual direction
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind


class MomentumAgentV2:
    """
    Enhanced momentum detection:
    1. RSI confirmation (oversold/overbought)
    2. MACD crossover
    3. Rate of change
    4. Volume confirmation
    """
    name = "Momentum"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0, evidence=["Not enough data"])
        
        score = 0
        ev = []
        
        # RSI
        rsi = ind.rsi(df, 14).iloc[-1]
        rsi_prev = ind.rsi(df, 14).iloc[-2] if len(df) > 1 else rsi
        
        if rsi < 30:
            score += 0.3
            ev.append("RSI oversold: %.1f" % rsi)
        elif rsi > 70:
            score -= 0.3
            ev.append("RSI overbought: %.1f" % rsi)
        
        # RSI divergence (price down, RSI up = bullish)
        if len(df) > 10:
            price_chg = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
            rsi_chg = (rsi - ind.rsi(df, 14).iloc[-5]) / max(ind.rsi(df, 14).iloc[-5], 1)
            
            if price_chg < -0.01 and rsi_chg > 0.05:
                score += 0.25
                ev.append("Bullish RSI divergence")
            elif price_chg > 0.01 and rsi_chg < -0.05:
                score -= 0.25
                ev.append("Bearish RSI divergence")
        
        # MACD
        macd_line, signal_line, histogram = ind.macd(df)
        macd_now = histogram.iloc[-1]
        macd_prev = histogram.iloc[-2] if len(histogram) > 1 else macd_now
        
        if macd_now > 0 and macd_prev <= 0:
            score += 0.2
            ev.append("MACD bullish crossover")
        elif macd_now < 0 and macd_prev >= 0:
            score -= 0.2
            ev.append("MACD bearish crossover")
        elif macd_now > macd_prev and macd_now > 0:
            score += 0.1
            ev.append("MACD accelerating up")
        elif macd_now < macd_prev and macd_now < 0:
            score -= 0.1
            ev.append("MACD accelerating down")
        
        # Rate of Change
        roc = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10] * 100
        if roc > 2:
            score += 0.15
            ev.append("Strong positive ROC: %.2f%%" % roc)
        elif roc < -2:
            score -= 0.15
            ev.append("Strong negative ROC: %.2f%%" % roc)
        
        # Volume confirmation
        vol_ratio = ind.volume_ratio(df).iloc[-1]
        if vol_ratio > 1.5 and score > 0:
            score += 0.1
            ev.append("Volume confirms bullish (%.1fx)" % vol_ratio)
        elif vol_ratio > 1.5 and score < 0:
            score -= 0.1
            ev.append("Volume confirms bearish (%.1fx)" % vol_ratio)
        
        # Direction
        if score > 0.1:
            d = "BUY"
        elif score < -0.1:
            d = "SELL"
        else:
            d = "NEUTRAL"
        
        conf = min(abs(score) * 200, 100)
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev or ["No signal"])
