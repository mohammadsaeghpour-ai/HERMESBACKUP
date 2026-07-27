"""
Regime Agent
============
Detects market regime: Trending, Ranging, Volatile, Calm.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class RegimeAgent(BaseAgent):
    name = "Regime"
    weight = 0.8

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        adx_val = df["adx"].iloc[-1]
        atr_pct = df["atr_pct"].iloc[-1]
        bb_width = df["bb_width"].iloc[-1]
        close = df["close"].iloc[-1]
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]

        # ADX-based regime
        if adx_val > 30:
            regime = "TRENDING"
            evidence.append(f"ADX={adx_val:.0f} > 30 = TRENDING market")
        elif adx_val > 20:
            regime = "MILD_TREND"
            evidence.append(f"ADX={adx_val:.0f} = mild trend")
        else:
            regime = "RANGING"
            evidence.append(f"ADX={adx_val:.0f} < 20 = RANGING market")

        # Volatility regime
        atr_hist = df["atr_pct"].rolling(50).apply(lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False)
        atr_pctile = atr_hist.iloc[-1] if not np.isnan(atr_hist.iloc[-1]) else 0.5

        if atr_pctile > 0.8:
            vol_regime = "HIGH_VOLATILITY"
            evidence.append(f"ATR percentile {atr_pctile*100:.0f}% = HIGH volatility")
        elif atr_pctile < 0.2:
            vol_regime = "LOW_VOLATILITY"
            evidence.append(f"ATR percentile {atr_pctile*100:.0f}% = LOW volatility")
        else:
            vol_regime = "NORMAL"
            evidence.append(f"ATR percentile {atr_pctile*100:.0f}% = normal volatility")

        # Price vs moving averages
        if close > ema50:
            evidence.append("Price above EMA50 = bullish bias")
            score += 0.1
        else:
            evidence.append("Price below EMA50 = bearish bias")
            score -= 0.1

        # Regime confidence
        if regime == "TRENDING":
            confidence = min(100, adx_val * 2.5)
        elif regime == "RANGING":
            confidence = min(100, (30 - adx_val) * 5)
        else:
            confidence = 50

        evidence.append(f"Regime: {regime} | Volatility: {vol_regime}")

        return self._out(
            direction="NEUTRAL",  # Regime agent doesn't trade
            confidence=confidence,
            score=score,
            evidence=evidence,
            data={"regime": regime, "volatility_regime": vol_regime},
            reasoning=f"Market regime: {regime}, Volatility: {vol_regime}"
        )
