"""
Advanced Volume Analysis Agent — Institutional Flow Detection
==============================================================
Comprehensive volume analysis combining:

1. **OBV Trend & Divergence** — on-balance volume slope and price/OBV divergence
2. **VWAP Position** — price relative to volume-weighted average price
3. **Volume Profile** — Point of Control (POC), Value Area High/Low (VAH/VAL)
4. **Accumulation/Distribution Line** — money flow multiplier-based line
5. **Volume-Weighted Momentum** — momentum signals weighted by volume confirmation
6. **Volume Absorption** — high volume + small price change = absorption
7. **Volume Climax Detection** — extreme volume spikes signaling exhaustion
8. **Chaikin Money Flow** — multi-period money flow oscillator
9. **OBV Divergence** — divergence between price and on-balance volume

Weight: 1.3
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class VolumeAgent(BaseAgent):
    """Advanced volume analysis combining 9 independent subsystems.

    Identifies accumulation/distribution patterns, institutional flow,
    absorption events, and volume divergences for high-conviction signals.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.3).
    """

    name = "Volume"
    weight = 1.3

    # ------------------------------------------------------------------ #
    #  Internal: Volume Profile (POC/VAH/VAL)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _volume_profile(high, low, close, volume, n_bins=50):
        """Compute Volume Profile: POC, VAH, VAL.

        POC = price level with highest traded volume.
        VAH/VAL = boundaries of the 70% value area.

        Parameters
        ----------
        high, low, close, volume : numpy.ndarray
            OHLCV arrays.
        n_bins : int
            Number of price bins for the profile.

        Returns
        -------
        tuple[float, float, float]
            POC, VAH, VAL prices.
        """
        price_min = np.min(low)
        price_max = np.max(high)
        if price_max == price_min:
            return float(close[-1]), float(close[-1]), float(close[-1])

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        profile = np.zeros(n_bins)

        # Distribute volume across bins based on price range
        for i in range(len(close)):
            bar_low, bar_high = low[i], high[i]
            for b in range(n_bins):
                if bar_low <= bin_edges[b + 1] and bar_high >= bin_edges[b]:
                    # Overlap fraction
                    overlap_low = max(bar_low, bin_edges[b])
                    overlap_high = min(bar_high, bin_edges[b + 1])
                    bar_range = bar_high - bar_low + 1e-10
                    fraction = (overlap_high - overlap_low) / bar_range
                    profile[b] += volume[i] * fraction

        # POC
        poc_idx = np.argmax(profile)
        poc = float(bin_centers[poc_idx])

        # Value Area (70% of volume)
        total_vol = profile.sum()
        if total_vol == 0:
            return poc, float(price_max), float(price_min)

        sorted_idx = np.argsort(profile)[::-1]
        cum_vol = 0.0
        va_indices = []
        for idx in sorted_idx:
            va_indices.append(idx)
            cum_vol += profile[idx]
            if cum_vol >= 0.7 * total_vol:
                break

        vah = float(bin_centers[max(va_indices)])
        val = float(bin_centers[min(va_indices)])

        return poc, vah, val

    # ------------------------------------------------------------------ #
    #  Internal: Divergence Detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_divergence(price, indicator, lookback=40):
        """Detect bullish/bearish divergence between price and volume indicator.

        Parameters
        ----------
        price : numpy.ndarray
            Price series.
        indicator : numpy.ndarray
            Volume-based indicator (OBV, AD, etc.).
        lookback : int
            Candles to examine.

        Returns
        -------
        tuple[float, str]
            Score in [-1, +1] and label.
        """
        n = min(lookback, len(price))
        if n < 15:
            return 0.0, "Insufficient data"

        p = price[-n:]
        ind = indicator[-n:]
        window = max(3, n // 8)

        # Find peaks and troughs
        p_troughs, ind_troughs = [], []
        p_peaks, ind_peaks = [], []

        for i in range(window, n - window):
            local_p = p[i - window:i + window + 1]
            if p[i] == np.min(local_p):
                p_troughs.append(i)
                ind_troughs.append(ind[i])
            if p[i] == np.max(local_p):
                p_peaks.append(i)
                ind_peaks.append(ind[i])

        # Bullish divergence: price lower low, indicator higher low
        if len(p_troughs) >= 2:
            t1, t2 = p_troughs[-2], p_troughs[-1]
            if p[t2] < p[t1] * 0.995 and ind[t2] > ind[t1] * 1.005:
                return 0.6, "Bullish OBV Divergence"

        # Bearish divergence: price higher high, indicator lower high
        if len(p_peaks) >= 2:
            p1, p2 = p_peaks[-2], p_peaks[-1]
            if p[p2] > p[p1] * 1.005 and ind[p2] < ind[p1] * 0.995:
                return -0.6, "Bearish OBV Divergence"

        return 0.0, "No Divergence"

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run advanced volume analysis.

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
                reasoning="Volume: Insufficient data",
            )

        evidence: list = []
        signals = []  # (score, weight) pairs

        try:
            close_arr = df["close"].values.astype(float)
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            open_arr = df["open"].values.astype(float) if "open" in df.columns else close_arr
            vol_arr = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(close_arr))
            close = close_arr[-1]

            # ── 1. OBV Trend & Slope ──
            if "obv" in df.columns:
                obv_arr = df["obv"].values.astype(float)
            else:
                # Compute OBV from scratch
                obv_arr = np.zeros(len(close_arr))
                for i in range(1, len(close_arr)):
                    if close_arr[i] > close_arr[i - 1]:
                        obv_arr[i] = obv_arr[i - 1] + vol_arr[i]
                    elif close_arr[i] < close_arr[i - 1]:
                        obv_arr[i] = obv_arr[i - 1] - vol_arr[i]
                    else:
                        obv_arr[i] = obv_arr[i - 1]

            obv_val = obv_arr[-1]
            if "obv_ema" in df.columns:
                obv_ema_val = float(df["obv_ema"].iloc[-1])
            else:
                # Simple 20-period EMA of OBV
                alpha = 2.0 / 21.0
                obv_ema_val = obv_arr[-1]
                for v in obv_arr[-20:]:
                    obv_ema_val = alpha * v + (1 - alpha) * obv_ema_val

            obv_score = 0.0
            if obv_val > obv_ema_val:
                obv_score += 0.3
                evidence.append("OBV above EMA = bullish volume flow")
            else:
                obv_score -= 0.3
                evidence.append("OBV below EMA = bearish volume flow")

            # OBV slope (20-period)
            if len(obv_arr) >= 20:
                obv_slope = np.polyfit(range(20), obv_arr[-20:], 1)[0]
                obv_norm = obv_slope / (np.abs(np.mean(obv_arr[-20:])) + 1e-10)
                obv_score += np.clip(obv_norm * 5, -0.3, 0.3)
                evidence.append(f"OBV slope: {'↑' if obv_norm > 0 else '↓'}")

            signals.append((obv_score, 1.0))

            # ── 2. OBV Divergence ──
            obv_div_score, obv_div_label = self._detect_divergence(
                close_arr, obv_arr, lookback=60
            )
            signals.append((obv_div_score, 1.0))
            if "Divergence" in obv_div_label:
                evidence.append(f"⚠️ {obv_div_label}")

            # ── 3. VWAP Position ──
            vwap_val = float(df["vwap"].iloc[-1]) if "vwap" in df.columns else close
            vwap_dist = (close / vwap_val - 1.0) * 100.0 if vwap_val > 0 else 0.0

            vwap_score = 0.0
            if vwap_dist > 1.0:
                vwap_score = 0.15  # Above VWAP = bullish in trending market
                evidence.append(f"Price {vwap_dist:.2f}% above VWAP (bullish)")
            elif vwap_dist < -1.0:
                vwap_score = -0.15
                evidence.append(f"Price {vwap_dist:.2f}% below VWAP (bearish)")
            else:
                evidence.append(f"Price near VWAP ({vwap_dist:+.2f}%)")

            signals.append((vwap_score, 0.6))

            # ── 4. Volume Profile (POC/VAH/VAL) ──
            poc, vah, val = self._volume_profile(
                high_arr, low_arr, close_arr, vol_arr
            )

            vp_score = 0.0
            if close > vah:
                vp_score = 0.3
                evidence.append(f"Price above VAH ({vah:.2f}) = strong bullish")
            elif close < val:
                vp_score = -0.3
                evidence.append(f"Price below VAL ({val:.2f}) = strong bearish")
            elif close > poc:
                vp_score = 0.1
                evidence.append(f"Price above POC ({poc:.2f}) = mild bullish")
            else:
                vp_score = -0.1
                evidence.append(f"Price below POC ({poc:.2f}) = mild bearish")

            signals.append((vp_score, 0.8))

            # ── 5. Accumulation/Distribution Line ──
            hl_range = (high_arr - low_arr)
            hl_range_safe = np.where(hl_range == 0, 1e-10, hl_range)
            mfm = ((close_arr - low_arr) - (high_arr - close_arr)) / hl_range_safe
            mfv = mfm * vol_arr
            ad_line = np.cumsum(mfv)
            ad_val = ad_line[-1]

            # AD slope
            ad_score = 0.0
            if len(ad_line) >= 20:
                ad_slope = np.polyfit(range(20), ad_line[-20:], 1)[0]
                ad_norm = ad_slope / (np.abs(np.mean(ad_line[-20:])) + 1e-10)
                ad_score = np.clip(ad_norm * 10, -0.5, 0.5)
                evidence.append(
                    f"AD Line: {'accumulation' if ad_norm > 0 else 'distribution'} "
                    f"(slope={ad_norm:+.4f})"
                )

            signals.append((ad_score, 0.7))

            # ── 6. Volume-Weighted Momentum ──
            vwm_score = 0.0
            if len(close_arr) >= 10:
                price_change = (close_arr[-1] - close_arr[-10]) / close_arr[-10]
                avg_vol = np.mean(vol_arr[-10:])
                recent_vol = np.mean(vol_arr[-3:])
                vol_mult = recent_vol / (avg_vol + 1e-10)

                # Amplify momentum by volume confirmation
                vwm_score = np.clip(price_change * vol_mult * 20, -0.5, 0.5)
                evidence.append(
                    f"Vol-Weighted Momentum: {vwm_score:+.2f} "
                    f"(vol_mult={vol_mult:.2f}x)"
                )

            signals.append((vwm_score, 0.7))

            # ── 7. Volume Absorption ──
            # High volume + small price change = absorption (stealth activity)
            if len(close_arr) >= 10:
                price_range_10 = np.max(close_arr[-10:]) - np.min(close_arr[-10:])
                price_range_pct = price_range_10 / (np.mean(close_arr[-10:]) + 1e-10)
                vol_avg = np.mean(vol_arr[-10:])
                vol_last = vol_arr[-1]
                vol_ratio_last = vol_last / (vol_avg + 1e-10)

                # Absorption = high volume but small price movement
                if vol_ratio_last > 1.5 and price_range_pct < 0.02:
                    # Determine direction of absorption
                    recent_direction = close_arr[-1] - open_arr[-1]
                    abs_score = 0.3 if recent_direction >= 0 else -0.3
                    signals.append((abs_score, 0.8))
                    evidence.append(
                        f"🔄 VOLUME ABSORPTION detected ({vol_ratio_last:.1f}x vol, "
                        f"{price_range_pct * 100:.2f}% range) "
                        f"{'bullish' if abs_score > 0 else 'bearish'}"
                    )
                else:
                    evidence.append("No absorption pattern")

            # ── 8. Volume Climax Detection ──
            if len(vol_arr) >= 20:
                vol_mean = np.mean(vol_arr[-20:])
                vol_std = np.std(vol_arr[-20:])
                vol_z = (vol_arr[-1] - vol_mean) / (vol_std + 1e-10)

                if vol_z > 2.5:
                    # Climax: extreme volume can signal exhaustion or breakout
                    bullish_candle = close_arr[-1] > open_arr[-1]
                    climax_score = -0.3 if bullish_candle else 0.3  # Exhaustion
                    signals.append((climax_score, 0.5))
                    evidence.append(
                        f"⚡ VOLUME CLIMAX ({vol_z:.1f}σ) — possible "
                        f"{'exhaustion top' if climax_score < 0 else 'exhaustion bottom'}"
                    )
                else:
                    evidence.append(f"Volume z-score: {vol_z:.2f}σ (normal)")

            # ── 9. Chaikin Money Flow ──
            cmf_period = 20
            if len(close_arr) >= cmf_period:
                hl_r = np.where(hl_range == 0, 1e-10, hl_range)
                mfm_cmf = ((close_arr - low_arr) - (high_arr - close_arr)) / hl_r
                mfv_cmf = mfm_cmf * vol_arr

                cmf_val = np.sum(mfv_cmf[-cmf_period:]) / (np.sum(vol_arr[-cmf_period:]) + 1e-10)

                cmf_score = np.clip(cmf_val * 3, -0.5, 0.5)
                if cmf_val > 0.1:
                    evidence.append(f"CMF({cmf_val:.3f}) = Strong buying pressure 🟢")
                elif cmf_val < -0.1:
                    evidence.append(f"CMF({cmf_val:.3f}) = Strong selling pressure 🔴")
                else:
                    evidence.append(f"CMF({cmf_val:.3f}) = Neutral")

                signals.append((cmf_score, 0.8))

            # ── Consensus ──
            total_weight = sum(w for _, w in signals)
            consensus = sum(s * w for s, w in signals) / (total_weight + 1e-10)

            # Direction
            if consensus > 0.15:
                direction = "BUY"
            elif consensus < -0.15:
                direction = "SELL"
            else:
                direction = "NEUTRAL"

            bull_count = sum(1 for s, _ in signals if s > 0.05)
            bear_count = sum(1 for s, _ in signals if s < -0.05)

            confidence = (max(bull_count, bear_count) / max(len(signals), 1)) * 85.0
            confidence = float(np.clip(confidence, 0, 100))
            score = float(np.clip(consensus, -1.0, 1.0))

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "obv_val": round(float(obv_val), 2),
                    "obv_above_ema": bool(obv_val > obv_ema_val),
                    "vwap_dist_pct": round(float(vwap_dist), 4),
                    "volume_profile": {
                        "poc": round(poc, 2),
                        "vah": round(vah, 2),
                        "val": round(val, 2),
                    },
                    "ad_line_slope": round(float(
                        np.polyfit(range(20), ad_line[-20:], 1)[0]
                    ) if len(ad_line) >= 20 else 0, 2),
                    "obv_divergence": obv_div_label,
                    "cmf": round(float(cmf_val), 4) if "cmf_val" in dir() else 0,
                    "signals_bull": bull_count,
                    "signals_bear": bear_count,
                },
                reasoning=(
                    f"Volume: {direction} ({confidence:.0f}%) | "
                    f"OBV {'↑' if obv_val > obv_ema_val else '↓'} | "
                    f"VWAP dist={vwap_dist:+.2f}% | "
                    f"POC={poc:.2f} | CMF={cmf_val:.3f}" if "cmf_val" in dir() else
                    f"Volume: {direction} ({confidence:.0f}%) | "
                    f"OBV {'↑' if obv_val > obv_ema_val else '↓'} | "
                    f"VWAP dist={vwap_dist:+.2f}%"
                ),
            )

        except Exception as e:
            evidence.append(f"Volume error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"Volume failed: {str(e)[:80]}",
            )
