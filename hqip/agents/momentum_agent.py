"""
Momentum Agent
==============
Analyzes RSI, MACD, Stochastic, ROC, and divergences.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class MomentumAgent(BaseAgent):
    name = "Momentum"
    weight = 1.3

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        rsi = df["rsi"].iloc[-1]
        rsi7 = df["rsi7"].iloc[-1]
        macd_hist = df["macd_hist"].iloc[-1]
        macd_hist_prev = df["macd_hist"].iloc[-2]
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_d = df["stoch_d"].iloc[-1]
        roc = df["roc"].iloc[-1]

        # RSI
        if rsi < 30:
            evidence.append(f"RSI({rsi:.0f}) = Oversold 🟢")
            score += 0.3
        elif rsi > 70:
            evidence.append(f"RSI({rsi:.0f}) = Overbought 🔴")
            score -= 0.3
        else:
            evidence.append(f"RSI({rsi:.0f}) = Neutral")

        # RSI Momentum
        if rsi7 > rsi:
            evidence.append("RSI7 > RSI14 = Short-term momentum rising")
            score += 0.1
        elif rsi7 < rsi:
            evidence.append("RSI7 < RSI14 = Short-term momentum falling")
            score -= 0.1

        # MACD
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            evidence.append("MACD histogram rising above zero 🟢")
            score += 0.25
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            evidence.append("MACD histogram falling below zero 🔴")
            score -= 0.25
        elif macd_hist > 0:
            evidence.append("MACD histogram positive but momentum slowing")
            score += 0.05
        else:
            evidence.append("MACD histogram negative but momentum slowing")
            score -= 0.05

        # MACD Crossover
        if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]:
            evidence.append("🟢 MACD Bullish Crossover!")
            score += 0.3
        elif df["macd"].iloc[-1] < df["macd_signal"].iloc[-1] and df["macd"].iloc[-2] >= df["macd_signal"].iloc[-2]:
            evidence.append("🔴 MACD Bearish Crossover!")
            score -= 0.3

        # Stochastic
        if stoch_k < 20:
            evidence.append(f"Stochastic({stoch_k:.0f}) = Oversold")
            score += 0.15
        elif stoch_k > 80:
            evidence.append(f"Stochastic({stoch_k:.0f}) = Overbought")
            score -= 0.15

        if stoch_k > stoch_d and df["stoch_k"].iloc[-2] <= df["stoch_d"].iloc[-2]:
            evidence.append("Stochastic Bullish Cross")
            score += 0.15
        elif stoch_k < stoch_d and df["stoch_k"].iloc[-2] >= df["stoch_d"].iloc[-2]:
            evidence.append("Stochastic Bearish Cross")
            score -= 0.15

        # ROC
        evidence.append(f"ROC: {roc:.2f}%")
        score += np.clip(roc / 10, -0.2, 0.2)

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            reasoning=f"Momentum {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} | RSI={rsi:.0f} | MACD={'↑' if macd_hist > 0 else '↓'}"
        )
