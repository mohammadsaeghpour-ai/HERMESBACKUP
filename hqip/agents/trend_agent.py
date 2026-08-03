"""
Advanced Trend Analysis Agent — Multi-Indicator Trend Consensus
================================================================
Comprehensive trend analysis combining:

1. **Multi-EMA Stack** (9, 21, 50, 200) — alignment and fan spread
2. **EMA Fan Spread** — bullish when 9>21>50>200 with widening gaps
3. **Trend Acceleration** — fan spread widening (accelerating) vs narrowing
4. **ADX Trend Strength** with +DI/-DI directional confirmation
5. **SuperTrend** — ATR-based trailing stop for trend direction
6. **Ichimoku Cloud** — Tenkan-sen, Kijun-sen, Senkou A/B for trend + support/resistance
7. **Volume-Weighted Trend** — OBV and volume slope confirmation

Scoring: weighted consensus of all indicators, each normalized to [-1, +1].

Weight: 1.5
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TrendAgent(BaseAgent):
    """Advanced multi-indicator trend analysis with consensus scoring.

    Evaluates trend direction, strength, and acceleration using seven
    independent subsystems and produces a consensus direction/score.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.5).
    """

    name = "Trend"
    weight = 1.5

    # ------------------------------------------------------------------ #
    #  Internal: Compute EMA fan spread metrics
    # ------------------------------------------------------------------ #
    def _ema_fan_analysis(self, close, ema9, ema21, ema50, ema200):
        """Evaluate EMA fan alignment and spread.

        Parameters
        ----------
        close, ema9, ema21, ema50, ema200 : numpy.ndarray or float
            Price and EMA values.

        Returns
        -------
        tuple[float, str, dict]
            Score in [-1, 1], human-readable label, and detail dict.
        """
        # Convert to arrays for consistency
        emas = np.array([float(ema9), float(ema21), float(ema50), float(ema200)])
        c = float(close)

        # Perfect bullish order: ema9 > ema21 > ema50 > ema200
        bullish_order = emas[0] > emas[1] > emas[2] > emas[3]
        bearish_order = emas[0] < emas[1] < emas[2] < emas[3]

        # Spread: normalized distance between fastest and slowest
        avg_price = np.mean(emas) + 1e-10
        spread = (emas[0] - emas[3]) / avg_price * 100.0

        # Fan score: +1 for perfect bullish, -1 for perfect bearish
        score = 0.0
        if bullish_order:
            score = min(1.0, abs(spread) / 5.0)  # Wider spread = stronger
        elif bearish_order:
            score = -min(1.0, abs(spread) / 5.0)
        else:
            # Partial alignment
            pairs_bull = sum(1 for i in range(3) if emas[i] > emas[i + 1])
            pairs_bear = sum(1 for i in range(3) if emas[i] < emas[i + 1])
            score = (pairs_bull - pairs_bear) / 3.0 * 0.5

        # Price vs fan
        price_above = c > emas[0]
        price_below = c < emas[3]
        if price_above:
            score = min(1.0, score + 0.1)
        elif price_below:
            score = max(-1.0, score - 0.1)

        label = "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Mixed"
        detail = {
            "spread_pct": round(float(spread), 4),
            "bullish_order": bool(bullish_order),
            "bearish_order": bool(bearish_order),
        }
        return score, label, detail

    # ------------------------------------------------------------------ #
    #  Internal: Trend acceleration
    # ------------------------------------------------------------------ #
    def _trend_acceleration(self, ema9_arr, ema21_arr, ema50_arr, ema200_arr, lookback=20):
        """Detect if EMA fan spread is widening (accelerating) or narrowing.

        Parameters
        ----------
        ema9_arr, ema21_arr, ema50_arr, ema200_arr : numpy.ndarray
            Historical EMA values.
        lookback : int
            Number of candles to compare.

        Returns
        -------
        tuple[float, str]
            Acceleration score in [-1, 1] and label.
        """
        n = min(lookback, len(ema9_arr))
        if n < 5:
            return 0.0, "Unknown"

        recent_spread = ema9_arr[-n:] - ema200_arr[-n:]
        older_spread = ema9_arr[-2 * n:-n] - ema200_arr[-2 * n:-n] if len(ema9_arr) >= 2 * n else recent_spread

        recent_avg = np.mean(recent_spread)
        older_avg = np.mean(older_spread)
        delta = recent_avg - older_avg

        score = np.clip(delta / (abs(older_avg) + 1e-10), -1.0, 1.0)
        label = "Accelerating" if abs(delta) > 1e-6 else "Stable"
        return float(score), label

    # ------------------------------------------------------------------ #
    #  Internal: Ichimoku Cloud
    # ------------------------------------------------------------------ #
    def _ichimoku(self, high, low, close, tenkan_n=9, kijun_n=26, senkou_b_n=52):
        """Compute Ichimoku Cloud components and signal.

        Parameters
        ----------
        high, low, close : numpy.ndarray
            Price arrays.
        tenkan_n, kijun_n, senkou_b_n : int
            Periods for each component.

        Returns
        -------
        tuple[float, str, dict]
            Score in [-1, 1], label, and component values.
        """
        n = len(high)
        if n < senkou_b_n:
            return 0.0, "Insufficient data", {}

        # Tenkan-sen: (highest high + lowest low) / 2 over tenkan_n
        tenkan = (np.max(high[-tenkan_n:]) + np.min(low[-tenkan_n:])) / 2.0

        # Kijun-sen: same over kijun_n
        kijun = (np.max(high[-kijun_n:]) + np.min(low[-kijun_n:])) / 2.0

        # Senkou Span A: (Tenkan + Kijun) / 2 (current cloud edge)
        senkou_a = (tenkan + kijun) / 2.0

        # Senkou Span B: midpoint of kijun_n-period high/low
        senkou_b = (np.max(high[-senkou_b_n:]) + np.min(low[-senkou_b_n:])) / 2.0

        # Cloud top/bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)

        c = float(close[-1])

        # Signal scoring
        score = 0.0

        # 1. Price vs cloud
        if c > cloud_top:
            score += 0.4
        elif c < cloud_bot:
            score -= 0.4
        else:
            score += 0.0  # In the cloud

        # 2. Tenkan/Kijun cross
        if tenkan > kijun:
            score += 0.2
        elif tenkan < kijun:
            score -= 0.2

        # 3. Cloud color (future direction)
        if senkou_a > senkou_b:
            score += 0.2  # Bullish cloud
        elif senkou_a < senkou_b:
            score -= 0.2  # Bearish cloud

        score = float(np.clip(score, -1.0, 1.0))
        label = "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral"

        detail = {
            "tenkan": round(float(tenkan), 2),
            "kijun": round(float(kijun), 2),
            "senkou_a": round(float(senkou_a), 2),
            "senkou_b": round(float(senkou_b), 2),
            "cloud_top": round(float(cloud_top), 2),
            "cloud_bot": round(float(cloud_bot), 2),
            "price_in_cloud": cloud_bot <= c <= cloud_top,
        }
        return score, label, detail

    # ------------------------------------------------------------------ #
    #  Internal: SuperTrend
    # ------------------------------------------------------------------ #
    @staticmethod
    def _supertrend(high, low, close, atr_period=10, multiplier=3.0):
        """Compute SuperTrend direction.

        Parameters
        ----------
        high, low, close : numpy.ndarray
            Price arrays.
        atr_period : int
            ATR lookback period.
        multiplier : float
            ATR multiplier for band width.

        Returns
        -------
        tuple[int, float]
            Direction (+1 bullish, -1 bearish) and SuperTrend value.
        """
        n = len(close)
        if n < atr_period + 1:
            return 0, 0.0

        # Compute ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = np.zeros(n)
        atr[1:] = np.convolve(tr, np.ones(atr_period) / atr_period, mode="full")[:n - 1]
        atr[atr == 0] = 1e-10

        # HL2
        hl2 = (high + low) / 2.0

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        st = np.zeros(n)
        direction = np.zeros(n, dtype=int)
        direction[0] = 1

        for i in range(1, n):
            if close[i] > st[i - 1]:
                st[i] = lower_band[i]
                direction[i] = 1
            else:
                st[i] = upper_band[i]
                direction[i] = -1

            # Prevent band reversal
            if direction[i] == 1 and lower_band[i] < st[i - 1]:
                st[i] = max(st[i - 1], lower_band[i])
            elif direction[i] == -1 and upper_band[i] > st[i - 1]:
                st[i] = min(st[i - 1], upper_band[i])

        return int(direction[-1]), float(st[-1])

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run multi-indicator trend analysis.

        Parameters
        ----------
        df : pandas.DataFrame or None
            OHLCV + indicator data.
        symbol : str, optional
            Trading pair symbol.
        timeframe : str, optional
            Candle timeframe.
        **kwargs : dict
            Extra parameters (unused).

        Returns
        -------
        AgentOutput
            Direction, confidence, score, evidence, reasoning, and data.
        """
        if df is None or df.empty or len(df) < 50:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data (need 50+ candles)"],
                reasoning="Trend: Insufficient data",
            )

        evidence: list = []
        indicators_bull = 0
        indicators_bear = 0
        total_weight = 0.0
        weighted_score = 0.0

        try:
            close_arr = df["close"].values.astype(float)
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            volume_arr = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(close_arr))
            close = close_arr[-1]

            # ── 1. Multi-EMA Analysis ──
            ema_periods = [9, 21, 50, 200]
            ema_vals = {}
            ema_arrays = {}
            for p in ema_periods:
                col = f"ema{p}"
                if col in df.columns:
                    ema_vals[p] = float(df[col].iloc[-1])
                    ema_arrays[p] = df[col].values.astype(float)
                else:
                    # Compute EMA from scratch
                    span = 2 * p + 1
                    ema_vals[p] = float(np.mean(close_arr[-p:]))
                    ema_arrays[p] = np.full(len(close_arr), ema_vals[p])

            # ── 2. EMA Fan Spread ──
            fan_score, fan_label, fan_detail = self._ema_fan_analysis(
                close,
                ema_vals.get(9, close),
                ema_vals.get(21, close),
                ema_vals.get(50, close),
                ema_vals.get(200, close),
            )
            w = 1.5
            weighted_score += fan_score * w
            total_weight += w
            evidence.append(f"EMA Fan: {fan_label} (score={fan_score:+.2f}, "
                            f"spread={fan_detail['spread_pct']:.2f}%)")

            # ── 3. Trend Acceleration ──
            accel_score, accel_label = self._trend_acceleration(
                ema_arrays.get(9, close_arr),
                ema_arrays.get(21, close_arr),
                ema_arrays.get(50, close_arr),
                ema_arrays.get(200, close_arr),
            )
            w = 0.8
            weighted_score += accel_score * w
            total_weight += w
            evidence.append(f"Acceleration: {accel_label} ({accel_score:+.2f})")

            # ── 4. ADX with +DI/-DI ──
            adx_val = float(df["adx"].iloc[-1]) if "adx" in df.columns else 20.0
            plus_di = float(df["plus_di"].iloc[-1]) if "plus_di" in df.columns else 25.0
            minus_di = float(df["minus_di"].iloc[-1]) if "minus_di" in df.columns else 25.0

            # ADX strength modifier
            if adx_val > 40:
                trend_str = "Very Strong"
            elif adx_val > 25:
                trend_str = "Strong"
            elif adx_val > 20:
                trend_str = "Moderate"
            else:
                trend_str = "Weak/Range"

            # DI signal
            di_range = plus_di + minus_di + 1e-10
            di_score = (plus_di - minus_di) / di_range * 2.0  # Normalize to ~[-1, 1]
            adx_multiplier = min(1.0, adx_val / 40.0)  # Weight by ADX strength
            di_score *= adx_multiplier

            w = 1.2
            weighted_score += di_score * w
            total_weight += w
            evidence.append(f"ADX: {adx_val:.1f} ({trend_str}) | "
                            f"+DI={plus_di:.1f} -DI={minus_di:.1f} "
                            f"→ {di_score:+.2f}")

            # ── 5. SuperTrend ──
            st_dir, st_val = self._supertrend(high_arr, low_arr, close_arr)
            st_score = float(st_dir)  # +1 or -1
            w = 1.0
            weighted_score += st_score * w
            total_weight += w
            evidence.append(f"SuperTrend: {'Bullish' if st_dir == 1 else 'Bearish'} "
                            f"(level={st_val:.2f})")

            # ── 6. Ichimoku Cloud ──
            ich_score, ich_label, ich_detail = self._ichimoku(
                high_arr, low_arr, close_arr
            )
            w = 1.3
            weighted_score += ich_score * w
            total_weight += w
            ich_str = (f"Ichimoku: {ich_label} | "
                       f"Tenkan={ich_detail.get('tenkan', 0):.2f} "
                       f"Kijun={ich_detail.get('kijun', 0):.2f}")
            if ich_detail.get("price_in_cloud"):
                ich_str += " (in cloud)"
            evidence.append(ich_str)

            # ── 7. Volume-Weighted Trend ──
            vol_score = 0.0
            if "obv" in df.columns and "obv_ema" in df.columns:
                obv_val = float(df["obv"].iloc[-1])
                obv_ema_val = float(df["obv_ema"].iloc[-1])
                vol_score += 0.5 if obv_val > obv_ema_val else -0.5
                evidence.append(f"OBV: {'Above' if obv_val > obv_ema_val else 'Below'} EMA")

            # Volume slope (20-period)
            if len(volume_arr) >= 20:
                vol_slope = np.polyfit(range(20), volume_arr[-20:], 1)[0]
                vol_trend = vol_slope / (np.mean(volume_arr[-20:]) + 1e-10)
                vol_score += np.clip(vol_trend * 10, -0.5, 0.5)

            w = 0.8
            weighted_score += vol_score * w
            total_weight += w
            evidence.append(f"Volume trend: {vol_score:+.2f}")

            # ── Count agreement ──
            all_scores = [fan_score, di_score, st_score, ich_score, vol_score]
            for s in all_scores:
                if s > 0.1:
                    indicators_bull += 1
                elif s < -0.1:
                    indicators_bear += 1

            # ── Final consensus ──
            if total_weight > 0:
                consensus_score = weighted_score / total_weight
            else:
                consensus_score = 0.0

            agreement = max(indicators_bull, indicators_bear)
            total_indicators = len(all_scores)

            # Direction
            if consensus_score > 0.15:
                direction = "BUY"
            elif consensus_score < -0.15:
                direction = "SELL"
            else:
                direction = "NEUTRAL"

            # Confidence: based on agreement and ADX
            agreement_pct = agreement / total_indicators if total_indicators > 0 else 0
            confidence = agreement_pct * 80.0 + (adx_val / 40.0) * 20.0
            confidence = float(np.clip(confidence, 0, 100))
            score = float(np.clip(consensus_score, -1.0, 1.0))

            evidence.append(
                f"Consensus: {agreement}/{total_indicators} indicators agree "
                f"→ {direction}"
            )

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "consensus_score": round(score, 4),
                    "ema_fan_score": round(fan_score, 3),
                    "adx_value": round(adx_val, 1),
                    "adx_trend": trend_str,
                    "super_trend": "bullish" if st_dir == 1 else "bearish",
                    "ichimoku": ich_label,
                    "ichimoku_detail": ich_detail,
                    "indicators_bull": indicators_bull,
                    "indicators_bear": indicators_bear,
                    "acceleration": accel_label,
                    "fan_spread_pct": fan_detail["spread_pct"],
                },
                reasoning=(
                    f"Trend consensus: {direction} "
                    f"({agreement}/{total_indicators} agree) | "
                    f"ADX={adx_val:.0f} ({trend_str}) | "
                    f"Fan: {fan_label} | Ichimoku: {ich_label} | "
                    f"SuperTrend: {'↑' if st_dir == 1 else '↓'}"
                ),
            )

        except Exception as e:
            evidence.append(f"Trend error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"Trend analysis failed: {str(e)[:80]}",
            )
