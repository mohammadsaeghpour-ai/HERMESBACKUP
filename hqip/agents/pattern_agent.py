"""
Pattern Agent — Enhanced Candlestick Pattern Detection
=======================================================
Detects 20+ candlestick patterns (single, two-candle, three-candle) with:

- Pattern quality scoring based on volume confirmation
- Trend alignment bonus (patterns aligned with trend are stronger)
- Position-in-range context (hammer at support = strong, hammer mid-air = weak)
- Rarity-weighted confidence
- Sensitivity-tuned thresholds

Weight: 1.1
"""
from hqip.agents.base import BaseAgent
import numpy as np


class PatternAgent(BaseAgent):
    name = "Pattern"
    weight = 1.1

    # ── Helpers ─────────────────────────────────────────────────
    def _get_trend(self, df):
        """Determine current trend from EMAs. Returns 'bull', 'bear', or 'none'."""
        if "ema20" not in df.columns or "ema50" not in df.columns:
            return "none"
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]
        close = df["close"].iloc[-1]
        if close > ema20 > ema50:
            return "bull"
        elif close < ema20 < ema50:
            return "bear"
        return "none"

    def _get_vol_spike(self, df):
        """Volume ratio relative to average. Returns float >= 0."""
        if "vol_ratio" in df.columns:
            return float(df["vol_ratio"].iloc[-1])
        if "vol_sma" in df.columns and df["vol_sma"].iloc[-1] > 0:
            return float(df["volume"].iloc[-1] / df["vol_sma"].iloc[-1])
        return 1.0

    def _vol_bonus(self, vol_ratio):
        """Confidence bonus from volume spike (0 to 0.15)."""
        if vol_ratio > 2.0:
            return 0.15
        elif vol_ratio > 1.5:
            return 0.10
        elif vol_ratio > 1.2:
            return 0.05
        return 0.0

    def _trend_bonus(self, pattern_direction, trend):
        """
        Bonus if pattern aligns with trend, penalty if counter-trend.
        pattern_direction: 'bull' or 'bear'
        """
        if trend == "none":
            return 0.0
        if pattern_direction == trend:
            return 0.10  # aligned = stronger
        return -0.05  # counter-trend = weaker

    def _position_bonus(self, df):
        """
        Pattern quality bonus based on position in recent range.
        Patterns at extremes (support/resistance) are more meaningful.
        Returns 0 to 0.10 bonus.
        """
        if len(df) < 20 or "high" not in df.columns or "low" not in df.columns:
            return 0.0

        recent = df.iloc[-20:]
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        rng = range_high - range_low
        if rng <= 0:
            return 0.0

        close = df["close"].iloc[-1]
        position = (close - range_low) / rng

        # Near support (bottom 15%) or resistance (top 15%) = meaningful context
        if position < 0.15 or position > 0.85:
            return 0.10
        elif position < 0.25 or position > 0.75:
            return 0.05
        return 0.0

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 5:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data for pattern analysis"],
                reasoning="Pattern: No data",
            )

        evidence = []
        score = 0.0
        patterns_found = []

        trend = self._get_trend(df)
        vol_ratio = self._get_vol_spike(df)
        vol_bonus = self._vol_bonus(vol_ratio)
        pos_bonus = self._position_bonus(df)

        # ── Candle accessors ──
        def c(idx): return df["close"].iloc[idx]
        def o(idx): return df["open"].iloc[idx]
        def h(idx): return df["high"].iloc[idx]
        def l(idx): return df["low"].iloc[idx]
        def body(idx): return abs(c(idx) - o(idx))
        def range_(idx): return h(idx) - l(idx)
        def is_bull(idx): return c(idx) > o(idx)
        def is_bear(idx): return c(idx) < o(idx)
        def upper_wick(idx): return h(idx) - max(o(idx), c(idx))
        def lower_wick(idx): return min(o(idx), c(idx)) - l(idx)

        atr_val = df["atr"].iloc[-1] if "atr" in df.columns else range_(-1)

        def add_pattern(name, direction, base_conf):
            """Register a detected pattern with full quality adjustments."""
            nonlocal score
            direction_key = "bull" if direction == "BUY" else "bear"
            tb = self._trend_bonus(direction_key, trend)
            final_conf = base_conf + vol_bonus + tb + pos_bonus
            final_conf = max(0.0, min(1.0, final_conf))

            if direction == "BUY":
                score += final_conf
            else:
                score -= final_conf

            trend_note = f" ({trend} aligned)" if tb > 0 else ""
            vol_note = f" [vol×{vol_ratio:.1f}]" if vol_ratio > 1.2 else ""
            pos_note = f" [range extrema]" if pos_bonus > 0 else ""
            label = "🟢" if direction == "BUY" else "🔴"
            evidence.append(f"{label} {name}{trend_note}{vol_note}{pos_note}")
            patterns_found.append(name)

        # ════════════════════════════════════════════════════════
        #  SINGLE-CANDLE PATTERNS
        # ════════════════════════════════════════════════════════

        if range_(-1) > 0 and body(-1) > 0:
            # Hammer (bullish reversal — long lower wick)
            if lower_wick(-1) > 1.8 * body(-1) and upper_wick(-1) < body(-1) * 0.4:
                add_pattern("Hammer", "BUY", 0.35)

            # Inverted Hammer (bullish reversal — long upper wick after decline)
            if upper_wick(-1) > 1.8 * body(-1) and lower_wick(-1) < body(-1) * 0.4:
                add_pattern("Inverted Hammer", "BUY", 0.25)

            # Shooting Star (bearish reversal — long upper wick)
            if upper_wick(-1) > 1.8 * body(-1) and lower_wick(-1) < body(-1) * 0.4:
                add_pattern("Shooting Star", "SELL", 0.35)

            # Hanging Man (bearish reversal — long lower wick at top)
            if lower_wick(-1) > 1.8 * body(-1) and upper_wick(-1) < body(-1) * 0.4:
                add_pattern("Hanging Man", "SELL", 0.30)

        # Doji (indecision)
        if range_(-1) > 0:
            body_ratio = body(-1) / range_(-1)
            if body_ratio < 0.10:
                # Direction: look at prior candle trend
                if len(df) >= 2:
                    add_pattern("Doji", "BUY" if is_bear(-2) else "SELL", 0.15)

            # Spinning Top (small body, long wicks both sides)
            elif 0.10 < body_ratio < 0.25:
                upper_pct = upper_wick(-1) / range_(-1)
                lower_pct = lower_wick(-1) / range_(-1)
                if upper_pct > 0.30 and lower_pct > 0.30:
                    if len(df) >= 2:
                        add_pattern("Spinning Top", "BUY" if is_bear(-2) else "SELL", 0.12)

        # Marubozu (strong single candle, almost no wicks)
        if range_(-1) > 0 and body(-1) > 0:
            body_ratio = body(-1) / range_(-1)
            if body_ratio > 0.85:
                direction = "BUY" if is_bull(-1) else "SELL"
                add_pattern("Bullish Marubozu" if direction == "BUY" else "Bearish Marubozu",
                           direction, 0.30)

        # ════════════════════════════════════════════════════════
        #  TWO-CANDLE PATTERNS
        # ════════════════════════════════════════════════════════

        if len(df) >= 2:
            # Bullish Engulfing
            if is_bear(-2) and is_bull(-1):
                if c(-1) > o(-2) and o(-1) < c(-2):
                    add_pattern("Bullish Engulfing", "BUY", 0.40)

            # Bearish Engulfing
            if is_bull(-2) and is_bear(-1):
                if c(-1) < o(-2) and o(-1) > c(-2):
                    add_pattern("Bearish Engulfing", "SELL", 0.40)

            # Piercing Line (bullish)
            if is_bear(-2) and is_bull(-1):
                prev_body = abs(c(-2) - o(-2))
                if prev_body > 0:
                    mid_prev = (o(-2) + c(-2)) / 2
                    penetration = (c(-1) - mid_prev) / prev_body
                    if o(-1) < c(-2) and penetration > 0.4:
                        add_pattern("Piercing Line", "BUY", 0.35)

            # Dark Cloud Cover (bearish)
            if is_bull(-2) and is_bear(-1):
                prev_body = abs(c(-2) - o(-2))
                if prev_body > 0:
                    mid_prev = (o(-2) + c(-2)) / 2
                    penetration = (mid_prev - c(-1)) / prev_body
                    if o(-1) > c(-2) and penetration > 0.4:
                        add_pattern("Dark Cloud Cover", "SELL", 0.35)

            # Bullish Harami
            if is_bear(-2):
                prev_body = abs(c(-2) - o(-2))
                curr_body = body(-1)
                if prev_body > 0 and curr_body < prev_body * 0.6:
                    if o(-1) > c(-2) and c(-1) < o(-2):
                        add_pattern("Bullish Harami", "BUY", 0.30)

            # Bearish Harami
            if is_bull(-2):
                prev_body = abs(c(-2) - o(-2))
                curr_body = body(-1)
                if prev_body > 0 and curr_body < prev_body * 0.6:
                    if o(-1) < c(-2) and c(-1) > o(-2):
                        add_pattern("Bearish Harami", "SELL", 0.30)

            # Tweezer Bottom (bullish)
            if abs(l(-1) - l(-2)) < atr_val * 0.05 and is_bull(-1):
                add_pattern("Tweezer Bottom", "BUY", 0.28)

            # Tweezer Top (bearish)
            if abs(h(-1) - h(-2)) < atr_val * 0.05 and is_bear(-1):
                add_pattern("Tweezer Top", "SELL", 0.28)

        # ════════════════════════════════════════════════════════
        #  THREE-CANDLE PATTERNS
        # ════════════════════════════════════════════════════════

        if len(df) >= 3:
            # Morning Star (bullish reversal)
            prev_body = abs(c(-3) - o(-3))
            mid_body = abs(c(-2) - o(-2))
            if is_bear(-3) and prev_body > 0 and mid_body < prev_body * 0.4:
                if is_bull(-1) and c(-1) > (o(-3) + c(-3)) / 2:
                    add_pattern("Morning Star", "BUY", 0.48)

            # Evening Star (bearish reversal)
            if is_bull(-3) and prev_body > 0 and mid_body < prev_body * 0.4:
                if is_bear(-1) and c(-1) < (o(-3) + c(-3)) / 2:
                    add_pattern("Evening Star", "SELL", 0.48)

            # Three White Soldiers (bullish)
            if is_bull(-3) and is_bull(-2) and is_bull(-1):
                if c(-1) > c(-2) > c(-3):
                    if o(-2) > o(-3) and o(-1) > o(-2):
                        if body(-1) > 0 and body(-2) > 0 and body(-3) > 0:
                            add_pattern("Three White Soldiers", "BUY", 0.50)

            # Three Black Crows (bearish)
            if is_bear(-3) and is_bear(-2) and is_bear(-1):
                if c(-1) < c(-2) < c(-3):
                    if o(-2) < o(-3) and o(-1) < o(-2):
                        if body(-1) > 0 and body(-2) > 0 and body(-3) > 0:
                            add_pattern("Three Black Crows", "SELL", 0.50)

            # Rising Three Methods (bullish continuation)
            if is_bull(-3) and body(-3) > 0:
                # Small bearish candles inside the first candle's range
                if (is_bear(-2) and l(-2) > l(-3) and h(-2) < h(-3)):
                    if is_bull(-1) and c(-1) > h(-3):
                        add_pattern("Rising Three Methods", "BUY", 0.45)

        # ════════════════════════════════════════════════════════
        #  SUMMARY
        # ════════════════════════════════════════════════════════

        if not patterns_found:
            evidence.append("No significant patterns detected")

        evidence.insert(0, f"Trend: {trend} | Vol ratio: {vol_ratio:.2f}× | Position bonus: {pos_bonus:.2f}")

        # Direction and confidence from accumulated score
        direction = "BUY" if score > 0.10 else "SELL" if score < -0.10 else "NEUTRAL"

        # Scale confidence: more patterns = higher ceiling
        pattern_count = len(patterns_found)
        if pattern_count >= 3:
            conf_scale = 150
        elif pattern_count == 2:
            conf_scale = 130
        else:
            conf_scale = 110

        confidence = min(100.0, abs(score) * conf_scale)

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=float(np.clip(score, -1.0, 1.0)),
            evidence=evidence,
            data={
                "patterns": patterns_found,
                "trend": trend,
                "vol_ratio": round(vol_ratio, 2),
                "position_bonus": round(pos_bonus, 2),
                "pattern_count": pattern_count,
            },
            reasoning=(
                f"Patterns: {', '.join(patterns_found) if patterns_found else 'none'} | "
                f"Score={score:.2f} | Trend={trend} | "
                f"Quality: vol_bonus={vol_bonus:.2f}, pos_bonus={pos_bonus:.2f}"
            ),
        )
