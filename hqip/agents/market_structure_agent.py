"""
Market Structure Agent
======================
Analyzes swing highs/lows, support/resistance, break of structure.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class MarketStructureAgent(BaseAgent):
    name = "MarketStructure"
    weight = 1.4

    def _find_swings(self, series, lookback=5):
        highs, lows = [], []
        for i in range(lookback, len(series) - lookback):
            if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].max():
                highs.append((i, series.iloc[i]))
            if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].min():
                lows.append((i, series.iloc[i]))
        return highs, lows

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        close = df["close"].iloc[-1]
        swing_highs, swing_lows = self._find_swings(df["high"], lookback=5)
        _, swing_lows = self._find_swings(df["low"], lookback=5)

        # Recent swing points
        recent_highs = [h[1] for h in swing_highs[-3:]] if swing_highs else []
        recent_lows = [l[1] for l in swing_lows[-3:]] if swing_lows else []

        # Higher Highs / Higher Lows = Bullish
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1] > recent_highs[-2]
            hl = recent_lows[-1] > recent_lows[-2]
            ll = recent_lows[-1] < recent_lows[-2]
            lh = recent_highs[-1] < recent_highs[-2]

            if hh and hl:
                evidence.append("🟢 Higher Highs + Higher Lows = UPTREND")
                score += 0.4
            elif ll and lh:
                evidence.append("🔴 Lower Lows + Lower Highs = DOWNTREND")
                score -= 0.4
            elif hh and ll:
                evidence.append("Mixed: Higher High + Lower Low = expanding range")
            elif lh and hl:
                evidence.append("Mixed: Lower High + Higher Low = contracting range (squeeze)")

        # Support/Resistance levels
        if recent_lows:
            nearest_support = max([s for s in recent_lows if s < close], default=close * 0.99)
            support_dist = (close - nearest_support) / close * 100
            evidence.append(f"Nearest support: {nearest_support:.2f} ({support_dist:.2f}% below)")

            if support_dist < 0.5:
                evidence.append("⚠️ Very close to support - watch for bounce or break")
                score += 0.15

        if recent_highs:
            nearest_resistance = min([r for r in recent_highs if r > close], default=close * 1.01)
            resistance_dist = (nearest_resistance - close) / close * 100
            evidence.append(f"Nearest resistance: {nearest_resistance:.2f} ({resistance_dist:.2f}% above)")

            if resistance_dist < 0.5:
                evidence.append("⚠️ Very close to resistance - watch for rejection or break")
                score -= 0.15

        # Break of Structure (BOS)
        if len(df) > 5:
            # Check if last candle broke above recent high
            if recent_highs and close > recent_highs[-1]:
                evidence.append("🟢 Break of Structure (BOS) ABOVE")
                score += 0.3
            elif recent_lows and close < recent_lows[-1]:
                evidence.append("🔴 Break of Structure (BOS) BELOW")
                score -= 0.3

        # Trend structure via Donchian
        dc_upper = df["dc_upper"].iloc[-1]
        dc_lower = df["dc_lower"].iloc[-1]
        dc_mid = df["dc_mid"].iloc[-1]

        if close > dc_mid:
            evidence.append(f"Price in upper Donchian channel ({(close-dc_mid)/(dc_upper-dc_mid)*100:.0f}%)")
            score += 0.1
        else:
            evidence.append(f"Price in lower Donchian channel")
            score -= 0.1

        evidence.append(f"Swing structure: {len(swing_highs)} highs, {len(swing_lows)} lows detected")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 90)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            reasoning=f"Structure {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} | BOS={'above' if close > (recent_highs[-1] if recent_highs else 0) else 'below' if recent_lows and close < recent_lows[-1] else 'none'}"
        )
