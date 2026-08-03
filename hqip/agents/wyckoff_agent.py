"""
Wyckoff Agent — Accumulation/Distribution Detection
===================================================
Detects 4 phases: Accumulation, Markup, Distribution, Markdown.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class WyckoffAgent(BaseAgent):
    name = "Wyckoff"
    weight = 1.4

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or len(df) < 50:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Need 50+ candles for Wyckoff"])

        evidence = []
        score = 0.0
        close = df["close"].iloc[-1]
        vol = df["volume"].values
        prices = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # ── Phase Detection ──
        recent_20 = prices[-20:]
        recent_vol = vol[-20:]
        avg_vol = np.mean(vol[-50:])
        price_range = np.max(recent_20) - np.min(recent_20)
        price_position = (close - np.min(recent_20)) / max(price_range, 0.01)

        # Selling Climax (SC): sharp drop with huge volume
        for i in range(len(df)-20, len(df)):
            if i < 2: continue
            drop = (prices[i] - prices[i-3]) / prices[i-3] * 100
            if drop < -2 and vol[i] > avg_vol * 2:
                evidence.append(f"📉 Selling Climax at {prices[i]:.2f} ({drop:.1f}%, vol {vol[i]/avg_vol:.1f}x)")
                evidence.append("   🏦 Capitulation — weak hands selling to institutions")

        # Accumulation signs: tight range + high volume at bottom
        recent_range_pct = price_range / np.mean(recent_20) * 100
        vol_at_bottom = np.mean(recent_vol[recent_20 < np.median(recent_20)])
        vol_at_top = np.mean(recent_vol[recent_20 > np.median(recent_20)])

        if recent_range_pct < 5:
            evidence.append(f"Tight range: {recent_range_pct:.1f}% — potential accumulation/distribution base")

        # Phase C: Spring/Shakeout (false breakdown below support)
        recent_low = np.min(lows[-10:])
        prev_low = np.min(lows[-30:-10]) if len(lows) > 30 else recent_low
        if recent_low < prev_low and close > prev_low:
            evidence.append(f"🟢 SPRING detected: Price dipped below support ({prev_low:.2f}) then recovered!")
            evidence.append("   🏦 Classic Wyckoff accumulation Phase C — institutions shook out weak holders")
            score += 0.45

        # Phase D: Sign of Strength (SOS)
        if close > np.max(highs[-20:-10]) if len(highs) > 20 else False:
            evidence.append("🟢 SIGN OF STRENGTH (SOS): Price broke above range resistance")
            evidence.append("   🏦 Institutions now buying aggressively — markup beginning")
            score += 0.35

        # Distribution signs: high volume at top, narrowing range
        if vol_at_bottom > 0 and vol_at_top > vol_at_bottom * 1.5 and price_position > 0.7:
            evidence.append("🔴 DISTRIBUTION: High volume at highs — institutions selling to retail")
            score -= 0.35

        # Spring/Shakeout above (distribution)
        recent_high = np.max(highs[-10:])
        prev_high = np.max(highs[-30:-10]) if len(highs) > 30 else recent_high
        if recent_high > prev_high and close < prev_high:
            evidence.append(f"🔴 UPTRAP detected: Price spiked above resistance ({prev_high:.2f}) then fell back!")
            evidence.append("   🏦 Classic Wyckoff distribution — institutions trapped breakout buyers")
            score -= 0.45

        # Volume climax at top = distribution
        if recent_vol[-1] > avg_vol * 2.5 and price_position > 0.8:
            evidence.append("🔴 Volume climax at highs — potential distribution top")
            score -= 0.2

        # ── Phase Summary ──
        if score > 0.3:
            phase = "ACCUMULATION (Phase C/D)"
            evidence.append(f"📊 Wyckoff Phase: {phase} — institutions accumulating")
        elif score < -0.3:
            phase = "DISTRIBUTION (Phase C/D)"
            evidence.append(f"📊 Wyckoff Phase: {phase} — institutions distributing")
        elif price_position < 0.3 and recent_range_pct < 5:
            phase = "ACCUMULATION_BASE"
            evidence.append("📊 Potential ACCUMULATION base forming")
        elif price_position > 0.7 and recent_range_pct < 5:
            phase = "DISTRIBUTION_BASE"
            evidence.append("📊 Potential DISTRIBUTION base forming")
        else:
            phase = "UNCLEAR"
            evidence.append("📊 Phase unclear — no clear Wyckoff pattern")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 15)

        return self._out(
            direction=direction, confidence=confidence,
            score=np.clip(score, -1, 1), evidence=evidence,
            data={"phase": phase if score > 0.3 or score < -0.3 else "UNCLEAR", "price_position": price_position},
            reasoning=f"Wyckoff: {'accumulation' if score > 0 else 'distribution' if score < 0 else 'neutral'}"
        )
