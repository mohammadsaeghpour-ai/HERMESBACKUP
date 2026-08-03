"""
Advanced Momentum Agent — Multi-Oscillator Analysis with Divergence Detection
==============================================================================
State-of-the-art momentum analysis combining:

1. **RSI** with dynamic overbought/oversold levels that adapt to trend context
2. **MACD Histogram** analysis — crossovers, zero-line, momentum direction
3. **Stochastic RSI** — RSI of RSI for refined overbought/oversold signals
4. **Rate of Change (ROC)** — multi-period momentum measurement
5. **CCI (Commodity Channel Index)** — cyclical turn detection
6. **Williams %R** — price position relative to range
7. **MFI (Money Flow Index)** — volume-weighted RSI for institutional flow
8. **RSI Divergence Detection** — price new high + RSI lower high = bearish
9. **MACD Divergence Detection** — price vs MACD histogram divergences

Weight: 1.3
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class MomentumAgent(BaseAgent):
    """Multi-oscillator momentum analysis with divergence detection.

    Combines 9 independent momentum subsystems into a consensus score.
    Includes both standard oscillator analysis and divergence pattern
    detection for early reversal identification.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.3).
    """

    name = "Momentum"
    weight = 1.3

    # ------------------------------------------------------------------ #
    #  Divergence Detection Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_divergence(price, indicator, lookback=60):
        """Detect bullish/bearish divergence between price and indicator.

        Bullish divergence: price makes lower low, indicator makes higher low.
        Bearish divergence: price makes higher high, indicator makes lower high.

        Parameters
        ----------
        price : numpy.ndarray
            Price series (close).
        indicator : numpy.ndarray
            Indicator series (RSI, MACD, etc.).
        lookback : int
            Number of candles to search.

        Returns
        -------
        tuple[float, str]
            Divergence score in [-1, +1] and label string.
        """
        n = min(lookback, len(price))
        if n < 20:
            return 0.0, "Insufficient data"

        p = price[-n:]
        ind = indicator[-n:]

        # Find local extrema using a window
        window = max(5, n // 10)

        # Bullish divergence: price lower low, indicator higher low
        bull_div = False
        bear_div = False

        # Find two troughs in price
        p_troughs = []
        ind_troughs = []
        for i in range(window, n - window):
            if p[i] == min(p[i - window:i + window + 1]):
                p_troughs.append(i)
                ind_troughs.append(ind[i])

        if len(p_troughs) >= 2:
            # Check last two troughs
            t1, t2 = p_troughs[-2], p_troughs[-1]
            if p[t2] < p[t1] and ind[t2] > ind[t1]:
                bull_div = True

        # Find two peaks in price
        p_peaks = []
        ind_peaks = []
        for i in range(window, n - window):
            if p[i] == max(p[i - window:i + window + 1]):
                p_peaks.append(i)
                ind_peaks.append(ind[i])

        if len(p_peaks) >= 2:
            p1, p2 = p_peaks[-2], p_peaks[-1]
            if p[p2] > p[p1] and ind[p2] < ind[p1]:
                bear_div = True

        if bull_div and not bear_div:
            return 0.7, "Bullish Divergence"
        elif bear_div and not bull_div:
            return -0.7, "Bearish Divergence"
        elif bull_div and bear_div:
            return 0.0, "Conflicting Divergences"
        else:
            return 0.0, "No Divergence"

    # ------------------------------------------------------------------ #
    #  Internal: Dynamic RSI levels
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dynamic_rsi_levels(close, rsi_val, lookback=50):
        """Adjust RSI overbought/oversold levels based on recent trend.

        In a strong uptrend, RSI can legitimately stay above 70 (OB = 80+).
        In a strong downtrend, RSI can stay below 30 (OS = 20-).

        Parameters
        ----------
        close : numpy.ndarray
            Recent close prices.
        rsi_val : float
            Current RSI value.
        lookback : int
            Number of candles for trend estimation.

        Returns
        -------
        tuple[float, float, str]
            OB level, OS level, and trend label.
        """
        n = min(lookback, len(close))
        if n < 5:
            return 70.0, 30.0, "Unknown"

        # Simple linear regression slope
        x = np.arange(n)
        slope = np.polyfit(x, close[-n:], 1)[0]
        norm_slope = slope / (np.mean(close[-n:]) + 1e-10) * 100

        if norm_slope > 0.1:  # Uptrend
            ob_level, os_level = 80.0, 25.0
            label = "Uptrend"
        elif norm_slope < -0.1:  # Downtrend
            ob_level, os_level = 75.0, 20.0
            label = "Downtrend"
        else:
            ob_level, os_level = 70.0, 30.0
            label = "Range"

        return ob_level, os_level, label

    # ------------------------------------------------------------------ #
    #  Internal: Stochastic RSI
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stochastic_rsi(rsi_arr, k_period=14, smooth=3):
        """Compute Stochastic RSI: stoch(RSI).

        Parameters
        ----------
        rsi_arr : numpy.ndarray
            RSI values.
        k_period : int
            Lookback for the stochastic.
        smooth : int
            Smoothing period for %K.

        Returns
        -------
        tuple[float, float]
            StochRSI %K and %D (smoothed).
        """
        n = len(rsi_arr)
        if n < k_period + smooth:
            return 50.0, 50.0

        stoch = np.zeros(n)
        for i in range(k_period - 1, n):
            window = rsi_arr[i - k_period + 1:i + 1]
            rsi_min = np.min(window)
            rsi_max = np.max(window)
            rng = rsi_max - rsi_min
            stoch[i] = ((rsi_arr[i] - rsi_min) / rng * 100) if rng > 0 else 50.0

        # Smooth to get %K and %D
        k_vals = stoch[-smooth * 3:] if len(stoch) >= smooth * 3 else stoch
        k_smooth = np.convolve(k_vals, np.ones(smooth) / smooth, mode="valid")
        k_val = float(k_smooth[-1]) if len(k_smooth) > 0 else 50.0
        d_val = float(np.mean(k_smooth[-smooth:])) if len(k_smooth) >= smooth else k_val

        return float(np.clip(k_val, 0, 100)), float(np.clip(d_val, 0, 100))

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run advanced momentum analysis.

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
                reasoning="Momentum: Insufficient data",
            )

        evidence: list = []
        signals = []  # (score, weight) pairs

        try:
            close_arr = df["close"].values.astype(float)

            # ── 1. RSI with Dynamic Levels ──
            rsi_val = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
            rsi_arr = df["rsi"].values.astype(float) if "rsi" in df.columns else np.full(len(close_arr), 50.0)

            ob_level, os_level, trend_label = self._dynamic_rsi_levels(
                close_arr, rsi_val
            )

            if rsi_val > ob_level:
                rsi_score = -0.5 * (rsi_val - ob_level) / (100 - ob_level + 1e-10)
                evidence.append(f"RSI({rsi_val:.1f}) > OB({ob_level:.0f}) = Overbought 🔴")
            elif rsi_val < os_level:
                rsi_score = 0.5 * (os_level - rsi_val) / (os_level + 1e-10)
                evidence.append(f"RSI({rsi_val:.1f}) < OS({os_level:.0f}) = Oversold 🟢")
            else:
                rsi_score = (rsi_val - 50) / 50.0 * 0.2
                evidence.append(f"RSI({rsi_val:.1f}) = Neutral (trend: {trend_label})")

            # RSI momentum (RSI direction)
            if len(rsi_arr) >= 5:
                rsi_slope = rsi_arr[-1] - rsi_arr[-5]
                rsi_score += np.clip(rsi_slope / 30, -0.2, 0.2)
                if rsi_slope > 3:
                    evidence.append(f"RSI rising ({rsi_slope:+.1f})")
                elif rsi_slope < -3:
                    evidence.append(f"RSI falling ({rsi_slope:+.1f})")

            signals.append((rsi_score, 1.2))

            # ── 2. MACD Histogram Analysis ──
            macd_hist = float(df["macd_hist"].iloc[-1]) if "macd_hist" in df.columns else 0.0
            macd_hist_prev = float(df["macd_hist"].iloc[-2]) if len(df) > 1 and "macd_hist" in df.columns else macd_hist
            macd_val = float(df["macd"].iloc[-1]) if "macd" in df.columns else 0.0
            macd_sig = float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else 0.0
            macd_sig_prev = float(df["macd_signal"].iloc[-2]) if len(df) > 1 and "macd_signal" in df.columns else macd_sig

            macd_score = 0.0
            # Histogram direction
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                macd_score += 0.3
                evidence.append("MACD histogram ↑ above zero 🟢")
            elif macd_hist < 0 and macd_hist < macd_hist_prev:
                macd_score -= 0.3
                evidence.append("MACD histogram ↓ below zero 🔴")
            elif macd_hist > 0:
                macd_score += 0.1
                evidence.append("MACD histogram positive but decelerating")
            else:
                macd_score -= 0.1
                evidence.append("MACD histogram negative but decelerating")

            # MACD crossover detection
            macd_curr_diff = macd_val - macd_sig
            macd_prev_diff = macd_val - macd_sig if len(df) <= 2 else (
                float(df["macd"].iloc[-2]) - float(df["macd_signal"].iloc[-2])
                if "macd" in df.columns and "macd_signal" in df.columns else 0
            )

            if macd_curr_diff > 0 and macd_prev_diff <= 0:
                macd_score += 0.3
                evidence.append("🟢 MACD Bullish Crossover!")
            elif macd_curr_diff < 0 and macd_prev_diff >= 0:
                macd_score -= 0.3
                evidence.append("🔴 MACD Bearish Crossover!")

            signals.append((macd_score, 1.0))

            # ── 3. Stochastic RSI ──
            srsi_k, srsi_d = self._stochastic_rsi(rsi_arr)
            srsi_score = 0.0

            if srsi_k < 20:
                srsi_score += 0.4
                evidence.append(f"StochRSI({srsi_k:.1f}) = Oversold 🟢")
            elif srsi_k > 80:
                srsi_score -= 0.4
                evidence.append(f"StochRSI({srsi_k:.1f}) = Overbought 🔴")
            else:
                srsi_score = (srsi_k - 50) / 100.0

            # StochRSI cross
            if srsi_k > srsi_d and srsi_k < 30:
                srsi_score += 0.2
                evidence.append("StochRSI bullish cross (oversold)")
            elif srsi_k < srsi_d and srsi_k > 70:
                srsi_score -= 0.2
                evidence.append("StochRSI bearish cross (overbought)")

            signals.append((srsi_score, 0.8))

            # ── 4. Rate of Change (ROC) — Multi-period ──
            roc_score = 0.0
            for period, weight in [(5, 0.4), (10, 0.4), (20, 0.2)]:
                if len(close_arr) > period:
                    roc_val = (close_arr[-1] - close_arr[-period - 1]) / close_arr[-period - 1] * 100
                    roc_score += np.clip(roc_val / 5.0, -0.5, 0.5) * weight
                    evidence.append(f"ROC({period}): {roc_val:+.2f}%")

            signals.append((roc_score, 0.7))

            # ── 5. CCI (Commodity Channel Index) ──
            cci_val = float(df["cci"].iloc[-1]) if "cci" in df.columns else None
            if cci_val is None:
                # Compute from scratch
                tp = (df["high"] + df["low"] + df["close"]).values / 3.0
                tp_sma = np.convolve(tp, np.ones(20) / 20, mode="full")[:len(tp)]
                tp_mad = np.array([
                    np.mean(np.abs(tp[max(0, i - 19):i + 1] - np.mean(tp[max(0, i - 19):i + 1])))
                    for i in range(len(tp))
                ])
                cci_val = (tp[-1] - tp_sma[-1]) / (0.015 * tp_mad[-1] + 1e-10)

            cci_score = np.clip(-cci_val / 200, -0.5, 0.5)
            if cci_val > 200:
                evidence.append(f"CCI({cci_val:.0f}) = Extremely overbought 🔴")
            elif cci_val > 100:
                evidence.append(f"CCI({cci_val:.0f}) = Overbought")
            elif cci_val < -200:
                evidence.append(f"CCI({cci_val:.0f}) = Extremely oversold 🟢")
            elif cci_val < -100:
                evidence.append(f"CCI({cci_val:.0f}) = Oversold")
            else:
                evidence.append(f"CCI({cci_val:.0f}) = Neutral")
            signals.append((cci_score, 0.6))

            # ── 6. Williams %R ──
            willr_val = float(df["willr"].iloc[-1]) if "willr" in df.columns else -50.0
            willr_score = 0.0
            if willr_val > -20:
                willr_score = -0.3
                evidence.append(f"Williams %R({willr_val:.0f}) = Overbought 🔴")
            elif willr_val < -80:
                willr_score = 0.3
                evidence.append(f"Williams %R({willr_val:.0f}) = Oversold 🟢")
            else:
                evidence.append(f"Williams %R({willr_val:.0f}) = Neutral")
            signals.append((willr_score, 0.5))

            # ── 7. MFI (Money Flow Index) ──
            mfi_val = float(df["mfi"].iloc[-1]) if "mfi" in df.columns else 50.0
            mfi_score = 0.0
            if mfi_val > 80:
                mfi_score = -0.4
                evidence.append(f"MFI({mfi_val:.0f}) = Overbought 🔴")
            elif mfi_val < 20:
                mfi_score = 0.4
                evidence.append(f"MFI({mfi_val:.0f}) = Oversold 🟢")
            else:
                mfi_score = (mfi_val - 50) / 100.0
                evidence.append(f"MFI({mfi_val:.0f}) = Neutral")
            signals.append((mfi_score, 0.7))

            # ── 8. RSI Divergence ──
            rsi_div_score, rsi_div_label = self._find_divergence(close_arr, rsi_arr)
            signals.append((rsi_div_score, 1.0))
            if "Divergence" in rsi_div_label:
                evidence.append(f"RSI Divergence: {rsi_div_label}")

            # ── 9. MACD Divergence ──
            if "macd_hist" in df.columns:
                macd_hist_arr = df["macd_hist"].values.astype(float)
                macd_div_score, macd_div_label = self._find_divergence(
                    close_arr, macd_hist_arr
                )
                signals.append((macd_div_score, 0.9))
                if "Divergence" in macd_div_label:
                    evidence.append(f"MACD Divergence: {macd_div_label}")

            # ── Weighted consensus ──
            total_weight = sum(w for _, w in signals)
            consensus = sum(s * w for s, w in signals) / (total_weight + 1e-10)

            # Direction
            if consensus > 0.15:
                direction = "BUY"
            elif consensus < -0.15:
                direction = "SELL"
            else:
                direction = "NEUTRAL"

            # Count bullish vs bearish signals
            bull_count = sum(1 for s, _ in signals if s > 0.1)
            bear_count = sum(1 for s, _ in signals if s < -0.1)

            confidence = (max(bull_count, bear_count) / len(signals)) * 80.0
            confidence = float(np.clip(confidence, 0, 100))
            score = float(np.clip(consensus, -1.0, 1.0))

            evidence.append(
                f"Momentum consensus: {bull_count} bullish / {bear_count} "
                f"bearish / {len(signals) - bull_count - bear_count} neutral"
            )

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "rsi": round(rsi_val, 2),
                    "rsi_levels": {"ob": ob_level, "os": os_level, "trend": trend_label},
                    "macd_hist": round(macd_hist, 4),
                    "stoch_rsi_k": round(srsi_k, 2),
                    "stoch_rsi_d": round(srsi_d, 2),
                    "cci": round(cci_val, 2),
                    "willr": round(willr_val, 2),
                    "mfi": round(mfi_val, 2),
                    "rsi_divergence": rsi_div_label,
                    "signals_bull": bull_count,
                    "signals_bear": bear_count,
                    "consensus": round(score, 4),
                },
                reasoning=(
                    f"Momentum: {direction} ({confidence:.0f}%) | "
                    f"RSI={rsi_val:.0f} | MACD={macd_hist:+.4f} | "
                    f"StochRSI={srsi_k:.0f} | CCI={cci_val:.0f} | "
                    f"Div: {rsi_div_label}"
                ),
            )

        except Exception as e:
            evidence.append(f"Momentum error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"Momentum failed: {str(e)[:80]}",
            )
