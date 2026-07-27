"""
Risk Agent
==========
Position sizing, stop loss, take profit, circuit breakers.
Only produces risk parameters. Never outputs BUY/SELL.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class RiskAgent(BaseAgent):
    name = "Risk"
    weight = 0.0  # Doesn't vote, only calculates

    def analyze(self, df=None, symbol="", timeframe="", direction="NEUTRAL",
                capital=10000, max_loss=100, leverage=5, **kwargs):
        if df is None or df.empty:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["No data"])

        evidence = []
        data = {}
        close = df["close"].iloc[-1]
        atr_val = df["atr"].iloc[-1]

        if direction == "NEUTRAL":
            evidence.append("No position to risk - NO TRADE")
            return self._out(direction="NEUTRAL", confidence=0, evidence=evidence, reasoning="No trade")

        # Position sizing
        risk_amount = min(max_loss, capital * 0.01)
        sl_distance = atr_val * 1.5  # 1.5 ATR stop loss
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        position_value = position_size * close

        evidence.append(f"Capital: ${capital:,.0f}")
        evidence.append(f"Risk per trade: ${risk_amount:.2f} ({risk_amount/capital*100:.1f}%)")
        evidence.append(f"ATR: {atr_val:.2f}")
        evidence.append(f"SL distance: {sl_distance:.2f} (1.5x ATR)")

        # Entry, SL, TP levels
        if direction == "BUY":
            entry = close
            sl = close - sl_distance
            tp1 = close + sl_distance * 1.5  # 1.5:1 RR
            tp2 = close + sl_distance * 2.5  # 2.5:1 RR
            tp3 = close + sl_distance * 4.0  # 4:1 RR
        else:
            entry = close
            sl = close + sl_distance
            tp1 = close - sl_distance * 1.5
            tp2 = close - sl_distance * 2.5
            tp3 = close - sl_distance * 4.0

        rr1 = abs(tp1 - entry) / max(abs(sl - entry), 0.01)
        rr2 = abs(tp2 - entry) / max(abs(sl - entry), 0.01)
        rr3 = abs(tp3 - entry) / max(abs(sl - entry), 0.01)

        evidence.append(f"Entry: {entry:.2f}")
        evidence.append(f"Stop Loss: {sl:.2f}")
        evidence.append(f"TP1: {tp1:.2f} (RR={rr1:.1f})")
        evidence.append(f"TP2: {tp2:.2f} (RR={rr2:.1f})")
        evidence.append(f"TP3: {tp3:.2f} (RR={rr3:.1f})")
        evidence.append(f"Position Size: {position_size:.6f}")
        evidence.append(f"Position Value: ${position_value:.2f} ({position_value/capital*100:.1f}% of capital)")

        # Liquidation estimate
        liq_distance = (close * 0.9) / leverage
        if direction == "BUY":
            liq = close - liq_distance
        else:
            liq = close + liq_distance
        evidence.append(f"Liquidation (est): {liq:.2f}")

        data = {
            "entry": round(entry, 2), "sl": round(sl, 2),
            "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
            "position_size": round(position_size, 6),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "rr1": round(rr1, 2), "rr2": round(rr2, 2), "rr3": round(rr3, 2),
            "liquidation": round(liq, 2),
            "leverage": leverage
        }

        return self._out(
            direction="NEUTRAL",
            confidence=100,
            score=0,
            evidence=evidence,
            data=data,
            reasoning=f"Risk: SL={sl_distance:.2f} | RR1={rr1:.1f} | Size={position_size:.6f}"
        )
