"""
Advanced Volatility Analysis Agent — Regime Detection & Breakout Signals
========================================================================
Comprehensive volatility analysis combining:

1. **ATR Percentile & Trend** — where current volatility sits in its history
2. **Bollinger Band Width** — band expansion/contraction tracking
3. **Bollinger Band Position** — price position within bands (bb_pct)
4. **Bollinger Squeeze Detection** — compression → breakout precursor
5. **Keltner Channel Squeeze** — BB inside KC = squeeze confirmed
6. **Historical Volatility Percentile** — annualized vol vs rolling history
7. **Implied Volatility Proxy** — options-model-inspired vol estimate from
   recent returns distribution (skew/kurtosis-adjusted)
8. **Volatility Regime Detection** — transitions between low/high regimes
9. **Choppiness Index** — trend vs range classification from ATR path

Weight: 1.0
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class VolatilityAgent(BaseAgent):
    """Multi-dimensional volatility analysis for regime detection and signals.

    Combines 9 volatility subsystems to characterize the current volatility
    environment and generate directional/neutral signals.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.0).
    """

    name = "Volatility"
    weight = 1.0

    # ------------------------------------------------------------------ #
    #  Internal: Historical Volatility Percentile
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hv_percentile(close_arr, window=20, lookback=252):
        """Compute historical volatility and its percentile rank.

        Parameters
        ----------
        close_arr : numpy.ndarray
            Close prices.
        window : int
            Return calculation window.
        lookback : int
            How many historical observations for percentile.

        Returns
        -------
        tuple[float, float, float]
            Current HV, percentile (0-100), HV trend direction.
        """
        n = len(close_arr)
        if n < window + 10:
            return 0.0, 50.0, 0.0

        # Log returns
        returns = np.diff(np.log(close_arr + 1e-10))

        # Rolling HV
        hv_series = np.array([
            np.std(returns[max(0, i - window + 1):i + 1]) * np.sqrt(365 * 24)
            for i in range(len(returns))
        ])

        current_hv = hv_series[-1]
        n_hist = min(lookback, len(hv_series))
        hist = hv_series[-n_hist:]

        percentile = float(np.sum(hist < current_hv) / len(hist) * 100)

        # HV trend: is vol expanding or contracting?
        hv_trend = 0.0
        if len(hv_series) >= 10:
            recent_avg = np.mean(hv_series[-5:])
            older_avg = np.mean(hv_series[-15:-5]) if len(hv_series) >= 15 else np.mean(hv_series[:5])
            hv_trend = (recent_avg - older_avg) / (older_avg + 1e-10)

        return float(current_hv), percentile, float(hv_trend)

    # ------------------------------------------------------------------ #
    #  Internal: Implied Volatility Proxy
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iv_proxy(close_arr, window=20):
        """Estimate implied volatility using options-model-inspired approach.

        Uses a modified Garman-Klass estimator which accounts for OHLC
        relationships, adjusted with skewness and kurtosis corrections.

        Parameters
        ----------
        close_arr : numpy.ndarray
            Close prices.
        window : int
            Lookback window.

        Returns
        -------
        tuple[float, float]
            IV proxy value and skewness indicator.
        """
        n = len(close_arr)
        if n < window + 5:
            return 0.0, 0.0

        # Returns distribution properties
        returns = np.diff(np.log(close_arr + 1e-10))[-window:]

        vol = np.std(returns) * np.sqrt(365 * 24)
        skew = float(np.mean((returns - np.mean(returns)) ** 3) / (np.std(returns) + 1e-10) ** 3)
        kurt = float(np.mean((returns - np.mean(returns)) ** 4) / (np.std(returns) + 1e-10) ** 4)

        # Adjust for fat tails (kurtosis > 3 means fatter tails)
        kurt_adj = 1.0 + 0.1 * max(0, kurt - 3)
        iv = vol * kurt_adj

        return float(iv), float(skew)

    # ------------------------------------------------------------------ #
    #  Internal: Choppiness Index
    # ------------------------------------------------------------------ #
    @staticmethod
    def _choppiness_index(high, low, close, period=14):
        """Compute Choppiness Index: 0-100 scale where higher = choppier.

        CI = 100 * LOG10(SUM(ATR, n) / (Max(n) - Min(n))) / LOG10(n)

        CI > 61.8 → choppy (range-bound market)
        CI < 38.2 → trending market

        Parameters
        ----------
        high, low, close : numpy.ndarray
            Price arrays.
        period : int
            Lookback period.

        Returns
        -------
        tuple[float, str]
            CI value and market state label.
        """
        n = len(close)
        if n < period + 1:
            return 50.0, "Unknown"

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # Sum of ATR over period
        atr_sum = np.sum(tr[-period:])

        # Price range over period
        price_high = np.max(high[-period - 1:])
        price_low = np.min(low[-period - 1:])
        price_range = price_high - price_low

        if price_range == 0:
            return 50.0, "Unknown"

        ci = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        ci = float(np.clip(ci, 0, 100))

        if ci > 61.8:
            label = "Choppy/Range"
        elif ci < 38.2:
            label = "Trending"
        else:
            label = "Transitioning"

        return ci, label

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run advanced volatility analysis.

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
        if df is None or df.empty or len(df) < 30:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data (need 30+ candles)"],
                reasoning="Volatility: Insufficient data",
            )

        evidence: list = []
        signals = []

        try:
            close_arr = df["close"].values.astype(float)
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            close = close_arr[-1]

            # ── 1. ATR Percentile & Trend ──
            atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else (
                np.mean(np.abs(np.diff(close_arr[-15:])))
            )
            atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else atr_val / close * 100

            # ATR percentile (how extreme is current ATR)
            if "atr_pct" in df.columns:
                atr_hist = df["atr_pct"].values.astype(float)
                n_hist = min(100, len(atr_hist))
                atr_percentile = float(np.sum(atr_hist[-n_hist:] < atr_hist[-1]) / n_hist * 100)
            else:
                atr_percentile = 50.0

            atr_score = 0.0
            if atr_percentile > 85:
                atr_score = -0.2  # High vol = caution
                evidence.append(f"ATR percentile {atr_percentile:.0f}% = Very high volatility 🔴")
            elif atr_percentile < 15:
                atr_score = 0.15  # Low vol = potential breakout
                evidence.append(f"ATR percentile {atr_percentile:.0f}% = Low vol (breakout setup) 🟢")
            else:
                evidence.append(f"ATR: {atr_pct:.2f}% (percentile: {atr_percentile:.0f}%)")

            signals.append((atr_score, 0.8))

            # ── 2. Bollinger Band Width & Squeeze ──
            bb_width = float(df["bb_width"].iloc[-1]) if "bb_width" in df.columns else 0.0
            bb_pct = float(df["bb_pct"].iloc[-1]) if "bb_pct" in df.columns else 0.5

            # BB Width percentile
            if "bb_width" in df.columns:
                bbw_hist = df["bb_width"].values.astype(float)
                n_bbw = min(100, len(bbw_hist))
                bbw_percentile = float(np.sum(bbw_hist[-n_bbw:] < bbw_hist[-1]) / n_bbw * 100)
            else:
                bbw_percentile = 50.0

            bb_score = 0.0

            # BB Position
            if bb_pct > 0.95:
                bb_score -= 0.25
                evidence.append(f"Price at upper BB ({bb_pct:.2f}) = overextended 🔴")
            elif bb_pct < 0.05:
                bb_score += 0.25
                evidence.append(f"Price at lower BB ({bb_pct:.2f}) = oversold 🟢")
            elif bb_pct > 0.7:
                bb_score += 0.05
                evidence.append(f"Price in upper BB zone ({bb_pct:.2f})")
            elif bb_pct < 0.3:
                bb_score -= 0.05
                evidence.append(f"Price in lower BB zone ({bb_pct:.2f})")
            else:
                evidence.append(f"BB position: {bb_pct:.2f} (mid-range)")

            # BB Squeeze
            is_squeeze = bbw_percentile < 10
            is_expansion = bbw_percentile > 90

            if is_squeeze:
                evidence.append(f"⚡ BB SQUEEZE (width percentile: {bbw_percentile:.0f}%) — breakout imminent!")
                # Squeeze itself is neutral for direction, but increases confidence
            elif is_expansion:
                evidence.append(f"BB Expansion (width percentile: {bbw_percentile:.0f}%)")
            else:
                evidence.append(f"BB Width: {bb_width:.4f} (percentile: {bbw_percentile:.0f}%)")

            signals.append((bb_score, 0.9))

            # ── 3. Keltner Channel Squeeze ──
            ema20 = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else np.mean(close_arr[-20:])
            kc_upper = ema20 + 2.0 * atr_val
            kc_lower = ema20 - 2.0 * atr_val
            bb_upper = float(df["bb_upper"].iloc[-1]) if "bb_upper" in df.columns else ema20 + 2 * atr_val
            bb_lower = float(df["bb_lower"].iloc[-1]) if "bb_lower" in df.columns else ema20 - 2 * atr_val

            kc_squeeze = bb_upper < kc_upper and bb_lower > kc_lower
            if kc_squeeze:
                evidence.append("Inside Keltner Channel = Squeeze CONFIRMED ⚡")
            else:
                evidence.append("BB outside Keltner = No squeeze")

            # ── 4. Historical Volatility Percentile ──
            hv, hv_pct, hv_trend = self._hv_percentile(close_arr)

            if hv_pct > 80:
                evidence.append(f"HV percentile: {hv_pct:.0f}% = Historically high")
            elif hv_pct < 20:
                evidence.append(f"HV percentile: {hv_pct:.0f}% = Historically low")
            else:
                evidence.append(f"HV percentile: {hv_pct:.0f}% (normal)")

            # ── 5. Implied Volatility Proxy ──
            iv_val, iv_skew = self._iv_proxy(close_arr)

            iv_score = 0.0
            if iv_skew < -0.5:
                iv_score += 0.15  # Negative skew → downside risk priced in → contrarian bullish
                evidence.append(f"IV proxy skew: {iv_skew:.2f} (negative → downside priced in)")
            elif iv_skew > 0.5:
                iv_score -= 0.15
                evidence.append(f"IV proxy skew: {iv_skew:.2f} (positive → upside euphoria)")

            signals.append((iv_score, 0.5))

            # ── 6. Volatility Regime Detection ──
            regime_score = 0.0
            if hv_trend > 0.2:
                regime_score = -0.1  # Vol expanding → caution
                evidence.append(f"⚠️ Vol EXPANDING (trend={hv_trend:+.3f})")
            elif hv_trend < -0.2:
                regime_score = 0.1  # Vol contracting → calm
                evidence.append(f"Vol contracting (trend={hv_trend:+.3f}) — consolidation")
            else:
                evidence.append(f"Vol stable (trend={hv_trend:+.3f})")

            signals.append((regime_score, 0.5))

            # ── 7. Choppiness Index ──
            ci, ci_label = self._choppiness_index(high_arr, low_arr, close_arr)

            ci_score = 0.0
            if ci > 61.8:
                ci_score = -0.1  # Choppy = reduce conviction
                evidence.append(f"Choppiness Index: {ci:.1f} — {ci_label}")
            elif ci < 38.2:
                ci_score = 0.1  # Trending = increase conviction
                evidence.append(f"Choppiness Index: {ci:.1f} — {ci_label}")
            else:
                evidence.append(f"Choppiness Index: {ci:.1f} — {ci_label}")

            signals.append((ci_score, 0.6))

            # ── Consensus ──
            total_weight = sum(w for _, w in signals)
            consensus = sum(s * w for s, w in signals) / (total_weight + 1e-10)

            if consensus > 0.1:
                direction = "BUY"
            elif consensus < -0.1:
                direction = "SELL"
            else:
                direction = "NEUTRAL"

            # Special signals
            special_boost = 0.0
            special_reasons = []
            if is_squeeze and kc_squeeze:
                special_boost += 10
                special_reasons.append("Squeeze confirmed (BB+KC)")
            if is_squeeze:
                special_boost += 5
                special_reasons.append("BB Squeeze active")

            confidence = abs(consensus) * 60.0 + special_boost
            confidence = float(np.clip(confidence, 0, 100))
            score = float(np.clip(consensus, -1.0, 1.0))

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "atr_pct": round(float(atr_pct), 4),
                    "atr_percentile": round(float(atr_percentile), 1),
                    "bb_width": round(float(bb_width), 6),
                    "bb_width_percentile": round(float(bbw_percentile), 1),
                    "bb_position": round(float(bb_pct), 4),
                    "bb_squeeze": bool(is_squeeze),
                    "kc_squeeze": bool(kc_squeeze),
                    "hv": round(float(hv), 4),
                    "hv_percentile": round(float(hv_pct), 1),
                    "hv_trend": round(float(hv_trend), 4),
                    "iv_proxy": round(float(iv_val), 4),
                    "iv_skew": round(float(iv_skew), 4),
                    "choppiness": round(float(ci), 2),
                    "choppiness_state": ci_label,
                    "vol_regime": "HIGH" if atr_percentile > 80 else (
                        "LOW" if atr_percentile < 20 else "NORMAL"
                    ),
                },
                reasoning=(
                    f"Volatility: {direction} ({confidence:.0f}%) | "
                    f"ATR pct={atr_percentile:.0f}% | BB squeeze={'YES' if is_squeeze else 'NO'} | "
                    f"KC squeeze={'YES' if kc_squeeze else 'NO'} | "
                    f"CI={ci:.0f} ({ci_label}) | "
                    f"{'SPECIAL: ' + ', '.join(special_reasons) if special_reasons else ''}"
                ),
            )

        except Exception as e:
            evidence.append(f"Volatility error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"Volatility failed: {str(e)[:80]}",
            )
