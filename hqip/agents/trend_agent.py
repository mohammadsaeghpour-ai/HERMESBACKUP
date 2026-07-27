"""
Trend Agent
===========
Analyzes trend direction, strength, and alignment across EMAs.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class TrendAgent(BaseAgent):
    name = "Trend"
    weight = 1.5

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 50:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        close = df["close"].iloc[-1]
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]
        ema100 = df["ema100"].iloc[-1]
        ema200 = df["ema200"].iloc[-1]
        adx_val = df["adx"].iloc[-1]
        plus_di = df["plus_di"].iloc[-1]
        minus_di = df["minus_di"].iloc[-1]
        st_dir = df["st_dir"].iloc[-1]

        # EMA Stack
        bullish_stack = ema20 > ema50 > ema100
        bearish_stack = ema20 < ema50 < ema100
        evidence.append(f"EMA Stack: {'Bullish' if bullish_stack else 'Bearish' if bearish_stack else 'Mixed'}")
        evidence.append(f"Price vs EMA20: {'Above' if close > ema20 else 'Below'} ({(close/ema20-1)*100:.2f}%)")

        # ADX Trend Strength
        if adx_val > 40:
            trend_str = "Very Strong"
        elif adx_val > 25:
            trend_str = "Strong"
        elif adx_val > 20:
            trend_str = "Moderate"
        else:
            trend_str = "Weak/Range"
        evidence.append(f"ADX: {adx_val:.1f} ({trend_str})")

        # DI Direction
        evidence.append(f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}")

        # SuperTrend
        evidence.append(f"SuperTrend: {'Bullish' if st_dir == 1 else 'Bearish'}")

        # Score calculation
        if bullish_stack:
            score += 0.4
        elif bearish_stack:
            score -= 0.4

        if adx_val > 25:
            score += 0.2 if plus_di > minus_di else -0.2
        else:
            score *= 0.5  # Reduce score in ranging market

        if st_dir == 1:
            score += 0.15
        else:
            score -= 0.15

        if close > ema20:
            score += 0.1
        else:
            score -= 0.1

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 100 * (1 + adx_val / 100))

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            reasoning=f"Trend {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} | ADX={adx_val:.0f} | EMA stack {'up' if bullish_stack else 'down' if bearish_stack else 'mixed'}"
        )
