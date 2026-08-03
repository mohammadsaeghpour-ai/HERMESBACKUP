"""
Risk Agent — Enhanced Risk Management
=======================================
Position sizing, stop loss, take profit, drawdown tracking.

Features:
- Kelly Criterion for position sizing (half-Kelly for safety)
- Dynamic SL based on ATR (1.0% for 15m, 1.0% for 1h+)
- Fixed R:R = 1:2 minimum (SL=1%, TP1=2%, TP2=3%, TP3=4%)
- Trailing stop: after TP1 hit, move SL to breakeven
- Partial take profit: 50% at TP1, 30% at TP2, 20% at TP3
- Session-based adjustment: lower size in low-volatility sessions
- Max drawdown tracking per session

Weight: 0 (doesn't vote)
"""
from hqip.agents.base import BaseAgent
import numpy as np


# Kelly Criterion constants
KELLY_FRACTION = 0.5  # Use half-Kelly for safety

# Partial take profit allocations
TP1_CLOSE_PCT = 0.50
TP2_CLOSE_PCT = 0.30
TP3_CLOSE_PCT = 0.20

# Session drawdown state (per-process)
_session_max_drawdown = 0.0
_session_peak_equity = 0.0


def _kelly_criterion(win_prob, risk_reward):
    """
    Kelly Criterion: Kelly% = (W * R - (1-W)) / R
    Returns fraction of capital to risk, capped at 25%.
    """
    if risk_reward <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    kelly = (win_prob * risk_reward - (1 - win_prob)) / risk_reward
    kelly *= KELLY_FRACTION
    return max(0.0, min(kelly, 0.25))


class RiskAgent(BaseAgent):
    name = "Risk"
    weight = 0  # doesn't vote

    def analyze(self, df=None, symbol="", timeframe="", direction="NEUTRAL",
                capital=10000, max_loss=100, leverage=5,
                win_probability=None, session_equity=None,
                **kwargs):
        if df is None or df.empty:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["No data"],
                reasoning="No data for risk calculation",
            )

        evidence = []
        data = {}
        close = float(df["close"].iloc[-1])

        if direction == "NEUTRAL":
            evidence.append("No position to risk — NO TRADE")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence, reasoning="No trade",
            )

        # ── ATR Calculation ──
        if "atr" in df.columns:
            atr_val = float(df["atr"].iloc[-1])
        else:
            # Compute ATR manually
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            c = df["close"].astype(float)
            tr = np.maximum(
                h - l,
                np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1)))
            )
            atr_val = float(tr.rolling(14).mean().iloc[-1])

        # ── Dynamic SL based on ATR ──
        # Fixed 1.0% for both 15m and 1h+ as specified
        sl_pct = 0.01  # 1.0%
        sl_distance = close * sl_pct

        # Also compute ATR-based SL for comparison
        atr_multiplier = 1.0
        avg_volatility = float(df["atr"].mean()) if "atr" in df.columns and len(df) > 5 else atr_val

        if atr_val > avg_volatility * 1.2:
            atr_multiplier = 1.5
            evidence.append(f"📈 High volatility → ATR multiplier: {atr_multiplier}x")
        else:
            evidence.append(f"📊 Normal volatility → ATR multiplier: {atr_multiplier}x")

        atr_sl = atr_val * atr_multiplier

        # Use the smaller of ATR-based and percentage-based SL
        # (conservative: don't let SL be too wide)
        actual_sl = min(sl_distance, atr_sl)
        actual_sl = max(actual_sl, close * 0.005)  # Minimum 0.5% SL

        evidence.append(f"ATR: {atr_val:.2f} | SL%: {sl_pct * 100:.1f}% | ATR SL: {atr_sl:.2f}")

        # ── Kelly Criterion Position Sizing ──
        if win_probability is None:
            win_probability = 0.55  # Default moderate estimate

        risk_reward = 2.0  # Minimum 1:2 R:R
        kelly_pct = _kelly_criterion(win_probability, risk_reward)
        risk_amount_kelly = capital * kelly_pct

        # Traditional 1% risk
        risk_amount_traditional = capital * 0.01

        # Use the more conservative of the two
        risk_amount = min(risk_amount_kelly, risk_amount_traditional)
        risk_amount = max(risk_amount, capital * 0.005)  # Minimum 0.5%

        position_size = risk_amount / actual_sl if actual_sl > 0 else 0
        position_value = position_size * close

        evidence.append(f"Capital: ${capital:,.0f}")
        evidence.append(
            f"Kelly Criterion: {kelly_pct * 100:.1f}% "
            f"(W={win_probability:.0%}, RR={risk_reward:.1f})"
        )
        evidence.append(
            f"Risk per trade: ${risk_amount:.2f} "
            f"({risk_amount / capital * 100:.1f}% of capital)"
        )

        # ── Entry, SL, TP levels (R:R = 1:2 minimum) ──
        if direction == "BUY":
            entry = close
            sl = close - actual_sl
            tp1 = close + actual_sl * 2.0   # 1:2 R:R
            tp2 = close + actual_sl * 3.0   # 1:3 R:R
            tp3 = close + actual_sl * 4.0   # 1:4 R:R
        else:
            entry = close
            sl = close + actual_sl
            tp1 = close - actual_sl * 2.0
            tp2 = close - actual_sl * 3.0
            tp3 = close - actual_sl * 4.0

        rr1 = abs(tp1 - entry) / max(abs(sl - entry), 0.01)
        rr2 = abs(tp2 - entry) / max(abs(sl - entry), 0.01)
        rr3 = abs(tp3 - entry) / max(abs(sl - entry), 0.01)

        # ── Partial Take Profit Plan ──
        tp_plan = [
            {
                "target": "TP1",
                "level": round(tp1, 2),
                "rr": round(rr1, 2),
                "close_pct": TP1_CLOSE_PCT,
                "close_amount_usd": round(position_value * TP1_CLOSE_PCT, 2),
                "action": f"Close {TP1_CLOSE_PCT * 100:.0f}% of position",
            },
            {
                "target": "TP2",
                "level": round(tp2, 2),
                "rr": round(rr2, 2),
                "close_pct": TP2_CLOSE_PCT,
                "close_amount_usd": round(position_value * TP2_CLOSE_PCT, 2),
                "action": f"Close {TP2_CLOSE_PCT * 100:.0f}% of position",
            },
            {
                "target": "TP3",
                "level": round(tp3, 2),
                "rr": round(rr3, 2),
                "close_pct": TP3_CLOSE_PCT,
                "close_amount_usd": round(position_value * TP3_CLOSE_PCT, 2),
                "action": f"Close remaining {TP3_CLOSE_PCT * 100:.0f}% (runner)",
            },
        ]

        # ── Trailing Stop ──
        trailing_stop = {
            "initial_sl": round(sl, 2),
            "breakeven_sl": round(entry, 2),
            "trigger": f"Move SL to breakeven ({entry:.2f}) after TP1 ({tp1:.2f}) hit",
            "description": (
                f"After {TP1_CLOSE_PCT * 100:.0f}% closed at TP1, "
                f"move stop loss to breakeven ({entry:.2f}) to eliminate risk "
                f"on remaining {1 - TP1_CLOSE_PCT:.0%} of position"
            ),
        }

        evidence.append(f"Entry: {entry:.2f}")
        evidence.append(f"Stop Loss: {sl:.2f} ({actual_sl / close * 100:.2f}%)")
        evidence.append(
            f"TP1: {tp1:.2f} (RR={rr1:.1f}) — Close {TP1_CLOSE_PCT * 100:.0f}%"
        )
        evidence.append(
            f"TP2: {tp2:.2f} (RR={rr2:.1f}) — Close {TP2_CLOSE_PCT * 100:.0f}%"
        )
        evidence.append(
            f"TP3: {tp3:.2f} (RR={rr3:.1f}) — Close remaining {TP3_CLOSE_PCT * 100:.0f}%"
        )
        evidence.append(f"Position Size: {position_size:.6f}")
        evidence.append(
            f"Position Value: ${position_value:.2f} "
            f"({position_value / capital * 100:.1f}% of capital)"
        )
        evidence.append("Trailing Stop: Move SL to breakeven after TP1 hit")

        # ── Liquidation estimate ──
        liq_distance = (close * 0.9) / leverage
        if direction == "BUY":
            liq = close - liq_distance
        else:
            liq = close + liq_distance
        evidence.append(f"Liquidation (est): {liq:.2f}")

        # ── Session-based adjustment (low-vol sessions get smaller size) ──
        session_adjusted = False
        if atr_val < avg_volatility * 0.7 and avg_volatility > 0:
            reduction = 0.70  # 30% size reduction in low-vol sessions
            position_size *= reduction
            position_value *= reduction
            risk_amount *= reduction
            evidence.append(
                f"⚠️ Low volatility session — position reduced by "
                f"{(1 - reduction) * 100:.0f}%"
            )
            session_adjusted = True

        # ── Max Drawdown Tracking ──
        global _session_max_drawdown, _session_peak_equity

        drawdown_pct = 0.0
        drawdown_amount = 0.0
        if session_equity and len(session_equity) > 0:
            current_equity = session_equity[-1]
            peak = max(session_equity)
            _session_peak_equity = max(_session_peak_equity, peak)
            drawdown_amount = _session_peak_equity - current_equity
            drawdown_pct = (drawdown_amount / max(_session_peak_equity, 1)) * 100
            _session_max_drawdown = max(_session_max_drawdown, drawdown_pct)

            evidence.append(
                f"📉 Session Drawdown: {drawdown_pct:.1f}% "
                f"(${drawdown_amount:.2f})"
            )
            evidence.append(f"📉 Max Session Drawdown: {_session_max_drawdown:.1f}%")

            if drawdown_pct > 5.0:
                evidence.append(
                    f"🚨 WARNING: Drawdown {drawdown_pct:.1f}% > 5% threshold. "
                    f"Consider reducing size or pausing."
                )
            if drawdown_pct > 10.0:
                evidence.append(
                    f"🚨🚨 CRITICAL: Drawdown {drawdown_pct:.1f}% > 10% limit. "
                    f"STOP TRADING for this session."
                )
        else:
            evidence.append("📉 No session equity data — drawdown tracking inactive")
            if _session_peak_equity == 0:
                _session_peak_equity = capital

        # ── Build output data ──
        data = {
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "position_size": round(position_size, 6),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "rr1": round(rr1, 2),
            "rr2": round(rr2, 2),
            "rr3": round(rr3, 2),
            "liquidation": round(liq, 2),
            "leverage": leverage,
            "sl_distance": round(actual_sl, 2),
            "sl_pct": round(actual_sl / close * 100, 2),
            "kelly_pct": round(kelly_pct * 100, 2),
            "win_probability": win_probability,
            "risk_reward": risk_reward,
            "tp_plan": tp_plan,
            "trailing_stop": trailing_stop,
            "session_drawdown_pct": round(drawdown_pct, 2),
            "session_max_drawdown_pct": round(_session_max_drawdown, 2),
            "risk_budget_used_pct": round(risk_amount / capital * 100, 2),
            "session_adjusted": session_adjusted,
        }

        reasoning = (
            f"Risk: SL={actual_sl:.2f} ({actual_sl / close * 100:.2f}%) | "
            f"Kelly={kelly_pct * 100:.1f}% | RR1={rr1:.1f} | "
            f"Size={position_size:.6f} (${position_value:.2f}) | "
            f"Partial TP: 50/30/20"
        )
        if session_adjusted:
            reasoning += " | Size reduced for low-vol session"
        if drawdown_pct > 5.0:
            reasoning += f" | ⚠️ Drawdown {drawdown_pct:.1f}%"

        return self._out(
            direction="NEUTRAL",
            confidence=100,
            score=0,
            evidence=evidence,
            data=data,
            reasoning=reasoning,
        )
