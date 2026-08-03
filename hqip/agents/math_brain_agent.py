"""
HQIP Mathematical Brain Agent
===============================
Uses Fibonacci, Elliott Wave, Gann, Pivot Points, VWAP Bands,
standard deviations, Z-score, Bollinger Bandwidth breakout,
mean reversion formulas, and all math applicable to trading.
"""
from hqip.agents.base import BaseAgent
import numpy as np


class MathBrainAgent(BaseAgent):
    name = "MathBrain"
    weight = 1.4

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out("NEUTRAL", 0, evidence=["insufficient data"])

        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        p = closes[-1]
        evidence = []
        score = 0.0

        # ═══════════════════════════════════════════════════
        # 1. FIBONACCI RETRACEMENT
        # ═══════════════════════════════════════════════════
        high_50 = np.max(highs[-50:])
        low_50 = np.min(lows[-50:])
        fib_range = high_50 - low_50

        fib_236 = high_50 - fib_range * 0.236
        fib_382 = high_50 - fib_range * 0.382
        fib_500 = high_50 - fib_range * 0.500
        fib_618 = high_50 - fib_range * 0.618
        fib_786 = high_50 - fib_range * 0.786

        # Where are we in Fibonacci?
        fib_levels = [fib_236, fib_382, fib_500, fib_618, fib_786]
        fib_names = ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]

        closest_fib = None
        closest_dist = float("inf")
        for i, (fl, fn) in enumerate(zip(fib_levels, fib_names)):
            dist = abs(p - fl) / fl
            if dist < closest_dist:
                closest_dist = dist
                closest_fib = (fl, fn)

        if closest_dist < 0.005:  # Within 0.5% of a fib level
            fl, fn = closest_fib
            if p > fib_500:
                score += 0.15
                evidence.append(f"🟢 نزدیک فیبو {fn} (${fl:,.0f}) — منطقه حمایت")
            else:
                score -= 0.15
                evidence.append(f"🔴 نزدیک فیبو {fn} (${fl:,.0f}) — منطقه مقاومت")
        else:
            # Price position in fib range
            fib_pos = (p - low_50) / fib_range * 100
            evidence.append(f"📐 موقعیت فیبو: {fib_pos:.0f}% از محدوده")

        # OTE zone (61.8% - 78.6%)
        if fib_786 <= p <= fib_618:
            score -= 0.20
            evidence.append(f"🔴 منطقه OTE (61.8-78.6%) — احتمال برگشت")
        elif fib_236 <= p <= fib_382:
            score += 0.15
            evidence.append(f"🟢 منطقه اصلاح سالم (23.6-38.2%)")

        # ═══════════════════════════════════════════════════
        # 2. ELLIOTT WAVE (simplified)
        # ═══════════════════════════════════════════════════
        # Detect wave structure: impulse = 5 waves, correction = 3 waves
        swing_highs = []
        swing_lows = []
        for i in range(2, min(len(highs) - 2, 50)):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append((i, lows[i]))

        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # Count waves
            waves = sorted(swing_highs[-3:] + swing_lows[-3:], key=lambda x: x[0])
            if len(waves) >= 4:
                up_moves = sum(1 for i in range(1, len(waves)) if waves[i][1] > waves[i-1][1])
                down_moves = len(waves) - 1 - up_moves

                if up_moves >= 3:
                    evidence.append(f"🌊 موج صعودی ({up_moves}/{up_moves+down_moves}) — احتمال اصلاح")
                    score -= 0.10
                elif down_moves >= 3:
                    evidence.append(f"🌊 موج نزولی ({down_moves}/{up_moves+down_moves}) — احتمال برگشت")
                    score += 0.10

        # ═══════════════════════════════════════════════════
        # 3. PIVOT POINTS (Classic)
        # ═══════════════════════════════════════════════════
        h = highs[-1]
        l = lows[-1]
        c = closes[-1]
        pivot = (h + l + c) / 3
        s1 = 2 * pivot - h
        s2 = pivot - (h - l)
        r1 = 2 * pivot - l
        r2 = pivot + (h - l)

        if p <= s2:
            score += 0.20
            evidence.append(f"🟢 زیر S2 (${s2:,.0f}) — اشباع فروش شدید")
        elif p <= s1:
            score += 0.10
            evidence.append(f"🟢 نزدیک S1 (${s1:,.0f})")
        elif p >= r2:
            score -= 0.20
            evidence.append(f"🔴 بالای R2 (${r2:,.0f}) — اشباع خرید شدید")
        elif p >= r1:
            score -= 0.10
            evidence.append(f"🔴 نزدیک R1 (${r1:,.0f})")

        # ═══════════════════════════════════════════════════
        # 4. VWAP BANDS (Institutional)
        # ═══════════════════════════════════════════════════
        typical_price = (highs + lows + closes) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volumes)
        cumulative_vol = np.cumsum(volumes)
        vwap = cumulative_tp_vol[-1] / cumulative_vol[-1] if cumulative_vol[-1] > 0 else p

        vwap_dist = (p - vwap) / vwap * 100

        # VWAP standard deviation bands
        variance = np.cumsum(volumes * (typical_price - vwap) ** 2) / cumulative_vol
        std_dev = np.sqrt(variance[-1])

        vwap_1s = std_dev / vwap * 100
        vwap_2s = 2 * std_dev / vwap * 100

        if p < vwap:
            if vwap_dist < -vwap_2s:
                score += 0.20
                evidence.append(f"🟢 زیر VWAP باند 2σ ({vwap_dist:+.2f}%) — خرید قوی")
            elif vwap_dist < -vwap_1s:
                score += 0.10
                evidence.append(f"🟢 زیر VWAP باند 1σ ({vwap_dist:+.2f}%)")
            else:
                score += 0.05
                evidence.append(f"🟢 زیر VWAP ({vwap_dist:+.2f}%)")
        else:
            if vwap_dist > vwap_2s:
                score -= 0.20
                evidence.append(f"🔴 بالای VWAP باند 2σ ({vwap_dist:+.2f}%) — فروش قوی")
            elif vwap_dist > vwap_1s:
                score -= 0.10
                evidence.append(f"🔴 بالای VWAP باند 1σ ({vwap_dist:+.2f}%)")
            else:
                score -= 0.05
                evidence.append(f"🔴 بالای VWAP ({vwap_dist:+.2f}%)")

        # ═══════════════════════════════════════════════════
        # 5. Z-SCORE (Mean Reversion)
        # ═══════════════════════════════════════════════════
        mean_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        z_score = (p - mean_20) / std_20 if std_20 > 0 else 0

        if z_score > 2.0:
            score -= 0.20
            evidence.append(f"🔴 Z-Score={z_score:.2f} — خرید افراطی (mean reversion)")
        elif z_score > 1.5:
            score -= 0.10
            evidence.append(f"🔴 Z-Score={z_score:.2f} — بالای میانگین")
        elif z_score < -2.0:
            score += 0.20
            evidence.append(f"🟢 Z-Score={z_score:.2f} — فروش افراطی (mean reversion)")
        elif z_score < -1.5:
            score += 0.10
            evidence.append(f"🟢 Z-Score={z_score:.2f} — زیر میانگین")

        # ═══════════════════════════════════════════════════
        # 6. BOLLINGER BANDWIDTH SQUEEZE → BREAKOUT
        # ═══════════════════════════════════════════════════
        bb_mid = mean_20
        bb_width_pct = (2 * std_20 / bb_mid * 100) if bb_mid > 0 else 0
        bb_width_20 = []
        for i in range(-20, 0):
            m = np.mean(closes[i-20:i]) if i-20 >= 0 else np.mean(closes[:i])
            s = np.std(closes[i-20:i]) if i-20 >= 0 else np.std(closes[:i])
            bb_width_20.append(2 * s / m * 100 if m > 0 else 0)

        if bb_width_20:
            bb_percentile = sum(1 for w in bb_width_20 if w < bb_width_pct) / len(bb_width_20) * 100
            if bb_percentile > 90:
                evidence.append(f"📊 BB عرض بالا ({bb_width_pct:.2f}%) — پتانسیل breakout")
            elif bb_percentile < 10:
                evidence.append(f"📊 BB squeeze ({bb_width_pct:.2f}%) — آماده انفجار")
                score *= 1.2

        # ═══════════════════════════════════════════════════
        # 7. STANDARD DEVIATION CHANNELS
        # ═══════════════════════════════════════════════════
        upper_2sd = mean_20 + 2 * std_20
        lower_2sd = mean_20 - 2 * std_20
        upper_1sd = mean_20 + std_20
        lower_1sd = mean_20 - std_20

        if p > upper_2sd:
            score -= 0.15
            evidence.append(f"🔴 بالای کانال 2σ (${upper_2sd:,.0f})")
        elif p < lower_2sd:
            score += 0.15
            evidence.append(f"🟢 زیر کانال 2σ (${lower_2sd:,.0f})")

        # ═══════════════════════════════════════════════════
        # 8. GANN ANGLES (simplified 45-degree)
        # ═══════════════════════════════════════════════════
        # 1x1 angle = 45 degrees = price moves same as time
        time_units = 20
        price_per_unit = fib_range / time_units

        if closes[-1] > closes[-20]:
            angle = "1x1 صعودی"
            score += 0.05
        elif closes[-1] < closes[-20]:
            angle = "1x1 نزولی"
            score -= 0.05

        # ═══════════════════════════════════════════════════
        # 9. PROBABILITY CONES
        # ═══════════════════════════════════════════════════
        daily_vol = np.std(np.diff(np.log(closes[-30:]))) if len(closes) > 30 else 0.02
        expected_1d = daily_vol * 100
        expected_5d = daily_vol * np.sqrt(5) * 100

        evidence.append(f"📊 نوسان روزانه: {expected_1d:.2f}% | ۵ روزه: {expected_5d:.2f}%")

        # ═══════════════════════════════════════════════════
        # FINAL DECISION
        # ═══════════════════════════════════════════════════
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NO_TRADE"
        confidence = min(100, abs(score) * 130 + 25)

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=round(score, 3),
            evidence=evidence[:8],
            reasoning=f"MathBrain: score={score:.3f}, z={z_score:.2f}, fib={closest_fib[1] if closest_fib else 'none'}, vwap={vwap_dist:+.2f}%",
            data={
                "fib_levels": {fn: round(fl, 2) for fn, fl in zip(fib_names, fib_levels)},
                "pivot": {"P": round(pivot, 2), "S1": round(s1, 2), "S2": round(s2, 2), "R1": round(r1, 2), "R2": round(r2, 2)},
                "vwap": round(vwap, 2),
                "z_score": round(z_score, 2),
                "bb_width": round(bb_width_pct, 2),
                "std_dev": round(std_dev, 2),
            },
        )
