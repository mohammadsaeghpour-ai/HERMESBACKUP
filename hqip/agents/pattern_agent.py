"""
Pattern Agent
=============
Detects candlestick patterns: engulfing, pin bar, hammer, etc.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class PatternAgent(BaseAgent):
    name = "Pattern"
    weight = 1.0

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 5:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0
        patterns_found = []

        o = df["open"].iloc[-1]
        h = df["high"].iloc[-1]
        l = df["low"].iloc[-1]
        c = df["close"].iloc[-1]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l
        atr_val = df["atr"].iloc[-1] if "atr" in df.columns else total_range

        # ── Bullish Engulfing ──
        if len(df) >= 2:
            prev_o = df["open"].iloc[-2]
            prev_c = df["close"].iloc[-2]
            if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
                evidence.append("🟢 Bullish Engulfing")
                patterns_found.append("bullish_engulfing")
                score += 0.4

        # ── Bearish Engulfing ──
        if len(df) >= 2:
            prev_o = df["open"].iloc[-2]
            prev_c = df["close"].iloc[-2]
            if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
                evidence.append("🔴 Bearish Engulfing")
                patterns_found.append("bearish_engulfing")
                score -= 0.4

        # ── Hammer (bullish) ──
        if total_range > 0:
            if lower_wick > 2 * body and upper_wick < body * 0.3 and body > 0:
                evidence.append("🟢 Hammer (bullish)")
                patterns_found.append("hammer")
                score += 0.3

        # ── Shooting Star (bearish) ──
        if total_range > 0:
            if upper_wick > 2 * body and lower_wick < body * 0.3 and body > 0:
                evidence.append("🔴 Shooting Star (bearish)")
                patterns_found.append("shooting_star")
                score -= 0.3

        # ── Doji ──
        if total_range > 0 and body / total_range < 0.1:
            evidence.append("Doji (indecision)")
            patterns_found.append("doji")
            # Slight pull toward continuation
            if df["close"].iloc[-2] > df["open"].iloc[-2]:
                score += 0.05
            else:
                score -= 0.05

        # ── Inside Bar ──
        if len(df) >= 2:
            prev_h = df["high"].iloc[-2]
            prev_l = df["low"].iloc[-2]
            if h < prev_h and l > prev_l:
                evidence.append("Inside Bar (consolidation)")
                patterns_found.append("inside_bar")

        # ── Pin Bar ──
        if total_range > 0:
            if lower_wick > total_range * 0.65:
                evidence.append("🟢 Bullish Pin Bar")
                patterns_found.append("bullish_pin")
                score += 0.25
            elif upper_wick > total_range * 0.65:
                evidence.append("🔴 Bearish Pin Bar")
                patterns_found.append("bearish_pin")
                score -= 0.25

        # ── Morning Star (3-candle bullish reversal) ──
        if len(df) >= 3:
            c3, o3 = df["close"].iloc[-1], df["open"].iloc[-1]
            c2, o2 = df["close"].iloc[-2], df["open"].iloc[-2]
            c1, o1 = df["close"].iloc[-3], df["open"].iloc[-3]
            if c1 < o1 and abs(c2 - o2) < abs(c1 - o1) * 0.3 and c3 > o3 and c3 > (c1 + o1) / 2:
                evidence.append("🟢 Morning Star!")
                patterns_found.append("morning_star")
                score += 0.45

        # ── Evening Star (3-candle bearish reversal) ──
        if len(df) >= 3:
            c3, o3 = df["close"].iloc[-1], df["open"].iloc[-1]
            c2, o2 = df["close"].iloc[-2], df["open"].iloc[-2]
            c1, o1 = df["close"].iloc[-3], df["open"].iloc[-3]
            if c1 > o1 and abs(c2 - o2) < abs(c1 - o1) * 0.3 and c3 < o3 and c3 < (c1 + o1) / 2:
                evidence.append("🔴 Evening Star!")
                patterns_found.append("evening_star")
                score -= 0.45

        if not patterns_found:
            evidence.append("No significant patterns detected")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 120)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            data={"patterns": patterns_found},
            reasoning=f"Patterns: {', '.join(patterns_found) if patterns_found else 'none'} | Score={score:.2f}"
        )
