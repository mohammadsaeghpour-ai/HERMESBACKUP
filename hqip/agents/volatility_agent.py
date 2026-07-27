"""
Volatility Agent
================
Analyzes ATR percentile, Bollinger width, squeeze detection.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class VolatilityAgent(BaseAgent):
    name = "Volatility"
    weight = 1.0

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        atr_pct = df["atr_pct"].iloc[-1]
        bb_width = df["bb_width"].iloc[-1]
        bb_pct = df["bb_pct"].iloc[-1]
        close = df["close"].iloc[-1]

        # ATR Percentile
        atr_hist = df["atr_pct"].rolling(50).apply(lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False)
        atr_pctile = atr_hist.iloc[-1] if not np.isnan(atr_hist.iloc[-1]) else 50
        evidence.append(f"ATR: {atr_pct:.2f}% (percentile: {atr_pctile:.0f}%)")

        if atr_pctile > 80:
            evidence.append("🔴 Very high volatility - caution")
            score -= 0.1
        elif atr_pctile < 20:
            evidence.append("🟢 Low volatility - potential breakout setup")
            score += 0.1

        # Bollinger Band Position
        if bb_pct > 0.95:
            evidence.append("Price at upper BB = overextended")
            score -= 0.2
        elif bb_pct < 0.05:
            evidence.append("Price at lower BB = oversold")
            score += 0.2
        elif bb_pct > 0.7:
            evidence.append("Price in upper BB zone")
            score += 0.05
        elif bb_pct < 0.3:
            evidence.append("Price in lower BB zone")
            score -= 0.05

        # BB Squeeze Detection
        bb_width_hist = df["bb_width"].rolling(50).apply(lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False)
        bb_pctile = bb_width_hist.iloc[-1] if not np.isnan(bb_width_hist.iloc[-1]) else 0.5
        evidence.append(f"BB Width: {bb_width:.4f} (percentile: {bb_pctile*100:.0f}%)")

        if bb_pctile < 0.1:
            evidence.append("⚡ BB SQUEEZE detected - breakout imminent!")
            # Squeeze itself is neutral - direction comes from other agents
        elif bb_pctile > 0.9:
            evidence.append("BB expansion - volatility high")

        # Keltner Channel vs BB
        ema20 = df["ema20"].iloc[-1]
        atr_val = df["atr"].iloc[-1]
        kc_upper = ema20 + 2 * atr_val
        kc_lower = ema20 - 2 * atr_val
        bb_upper = df["bb_upper"].iloc[-1]
        bb_lower = df["bb_lower"].iloc[-1]

        if bb_upper < kc_upper and bb_lower > kc_lower:
            evidence.append("Inside Keltner = Squeeze CONFIRMED")
        else:
            evidence.append("BB outside Keltner = no squeeze")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 100 + (10 if bb_pctile < 0.1 else 0))

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            reasoning=f"Volatility: ATR={atr_pct:.2f}% | BB position={bb_pct:.2f} | Squeeze={'YES' if bb_pctile < 0.1 else 'NO'}"
        )
