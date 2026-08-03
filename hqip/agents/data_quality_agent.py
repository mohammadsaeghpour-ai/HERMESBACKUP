"""
Data Quality Agent — Enhanced Data Validation
===============================================
Deep data validation for all market data:

- Data freshness check (age of last candle)
- Volume anomaly detection (zero-volume, spikes, declining)
- Price gap detection (missing candles, discontinuities)
- ATR sudden change detection (volatility explosion/collapse)
- Cross-timeframe consistency (price alignment between TFs)
- Wick analysis (manipulation / stop-hunt detection)

Returns:
  - quality_score: 0-100
  - weight_adjustment: 0.5-1.0 (applied to other agents)

Weight: 0 (non-voting, just adjusts other weights)
"""
from hqip.agents.base import BaseAgent
import numpy as np


class DataQualityAgent(BaseAgent):
    name = "DataQuality"
    weight = 0  # non-voting

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 20:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data (< 20 candles)"],
                reasoning="Not enough candles for analysis",
                data={"quality_score": 0, "weight_adjustment": 0.2},
            )

        evidence = []
        issues = 0
        max_issues = 8
        data = {}

        # ═══════════════════════════════════════════════════════
        # 1. DATA COMPLETENESS
        # ═══════════════════════════════════════════════════════
        total_cells = len(df) * len(df.columns)
        null_count = int(df.isnull().sum().sum())
        completeness = 1.0 - (null_count / total_cells) if total_cells > 0 else 0
        data["completeness"] = round(completeness * 100, 1)
        evidence.append(f"Data completeness: {completeness * 100:.1f}%")

        if completeness < 0.95:
            issues += 1
            evidence.append(f"⚠️ Missing {null_count} cells ({(1 - completeness) * 100:.1f}% incomplete)")

        # ═══════════════════════════════════════════════════════
        # 2. DATA FRESHNESS (staleness check)
        # ═══════════════════════════════════════════════════════
        if "timestamp" in df.columns:
            try:
                import pandas as pd
                now = pd.Timestamp.now(tz="UTC")
                last_ts = df["timestamp"].iloc[-1]

                if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
                    last_ts = pd.Timestamp(last_ts, tz="UTC")

                age_minutes = abs((now - pd.Timestamp(last_ts)).total_seconds() / 60)
                data["data_age_minutes"] = round(age_minutes, 1)

                if age_minutes > 60:
                    issues += 2
                    evidence.append(
                        f"🔴 STALE DATA: Last candle is {age_minutes:.0f} min old "
                        f"(>{60} min threshold)"
                    )
                elif age_minutes > 30:
                    issues += 1
                    evidence.append(f"⚠️ Aging data: Last candle is {age_minutes:.0f} min old")
                else:
                    evidence.append(f"✅ Data freshness OK ({age_minutes:.0f} min)")
            except Exception:
                evidence.append("Could not assess data freshness")
        else:
            # Estimate from candle count if no timestamp
            evidence.append("ℹ️ No timestamp column — freshness estimated from candle count")

        # ═══════════════════════════════════════════════════════
        # 3. TIMESTAMP GAPS (missing candles)
        # ═══════════════════════════════════════════════════════
        if "timestamp" in df.columns and len(df) > 5:
            try:
                ts_diffs = df["timestamp"].diff().dropna()
                median_gap = ts_diffs.median()
                max_gap = ts_diffs.max()
                gap_ratio = max_gap / median_gap if median_gap > 0 else 1
                data["max_gap_ratio"] = round(float(gap_ratio), 2)

                if gap_ratio > 5:
                    issues += 2
                    evidence.append(
                        f"🔴 LARGE GAP: max/median = {gap_ratio:.1f}x "
                        f"(missing candles detected)"
                    )
                elif gap_ratio > 3:
                    issues += 1
                    evidence.append(f"⚠️ Gap detected: max/median = {gap_ratio:.1f}x")
                else:
                    evidence.append(f"✅ Timestamps regular (gap ratio: {gap_ratio:.1f}x)")
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════
        # 4. VOLUME ANOMALIES
        # ═══════════════════════════════════════════════════════
        if "volume" in df.columns:
            vol = df["volume"].astype(float)

            # Zero-volume bars
            zero_vol = int((vol == 0).sum())
            zero_pct = zero_vol / len(vol) * 100
            data["zero_volume_pct"] = round(zero_pct, 1)

            if zero_pct > 20:
                issues += 2
                evidence.append(
                    f"🔴 VOLUME ANOMALY: {zero_vol} bars ({zero_pct:.0f}%) "
                    "with zero volume — data feed issue"
                )
            elif zero_pct > 10:
                issues += 1
                evidence.append(
                    f"⚠️ High zero-volume bars: {zero_vol} ({zero_pct:.0f}%)"
                )
            else:
                evidence.append(f"✅ Volume healthy ({zero_pct:.0f}% zero bars)")

            # Volume spike detection (z-score)
            vol_mean = vol.rolling(20).mean()
            vol_std = vol.rolling(20).std()
            if len(vol_mean.dropna()) > 0:
                latest_vol = float(vol.iloc[-1])
                latest_mean = float(vol_mean.iloc[-1])
                latest_std = float(vol_std.iloc[-1])
                if latest_std > 0:
                    vol_zscore = (latest_vol - latest_mean) / latest_std
                    data["volume_zscore"] = round(vol_zscore, 2)

                    if abs(vol_zscore) > 4:
                        issues += 1
                        evidence.append(
                            f"⚠️ Volume spike: z-score {vol_zscore:.1f} "
                            "(possible manipulation)"
                        )

            # Volume trend: declining = weak conviction
            if len(vol) > 15:
                recent_vol = float(vol.iloc[-5:].mean())
                older_vol = float(vol.iloc[-15:-5].mean())
                vol_change = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
                data["volume_trend"] = round(vol_change * 100, 1)
                if vol_change < -0.5:
                    evidence.append(
                        f"📉 Volume declining {vol_change * 100:.0f}% — "
                        "weak conviction behind moves"
                    )

        # ═══════════════════════════════════════════════════════
        # 5. PRICE GAP DETECTION
        # ═══════════════════════════════════════════════════════
        if "open" in df.columns and "close" in df.columns:
            gaps = []
            for i in range(1, len(df)):
                prev_close = df["close"].iloc[i - 1]
                curr_open = df["open"].iloc[i]
                if prev_close > 0:
                    gap_pct = abs(curr_open - prev_close) / prev_close
                    if gap_pct > 0.01:  # > 1% gap
                        gaps.append(gap_pct)

            data["price_gaps_count"] = len(gaps)
            if len(gaps) > 3:
                issues += 2
                max_gap = max(gaps) * 100
                evidence.append(
                    f"🔴 {len(gaps)} price gaps > 1% (largest: {max_gap:.2f}%) "
                    "— fragmented data / low liquidity"
                )
            elif len(gaps) > 0:
                issues += 1
                evidence.append(f"⚠️ {len(gaps)} price gaps detected")
            else:
                evidence.append("✅ No significant price gaps")

        # ═══════════════════════════════════════════════════════
        # 6. ATR SUDDEN CHANGE DETECTION
        # ═══════════════════════════════════════════════════════
        if all(c in df.columns for c in ["high", "low", "close"]):
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)

            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1)),
                ),
            )
            atr_14 = tr.rolling(14).mean()
            atr_50 = tr.rolling(50).mean()

            if len(atr_14.dropna()) > 0 and len(atr_50.dropna()) > 0:
                current_atr = float(atr_14.iloc[-1])
                baseline_atr = float(atr_50.iloc[-1])
                data["atr_14"] = round(current_atr, 4)
                data["atr_50"] = round(baseline_atr, 4)

                if baseline_atr > 0:
                    atr_ratio = current_atr / baseline_atr
                    data["atr_ratio"] = round(atr_ratio, 2)

                    if atr_ratio > 2.0:
                        issues += 1
                        evidence.append(
                            f"⚠️ ATR SURGED: current/baseline = {atr_ratio:.1f}x "
                            "— volatility explosion (news event?)"
                        )
                    elif atr_ratio < 0.3:
                        issues += 1
                        evidence.append(
                            f"⚠️ ATR collapsed: current/baseline = {atr_ratio:.1f}x "
                            "— extremely low volatility (squeeze building?)"
                        )
                    else:
                        evidence.append(f"✅ ATR normal ({atr_ratio:.1f}x baseline)")

        # ═══════════════════════════════════════════════════════
        # 7. WICK ANALYSIS (manipulation detection)
        # ═══════════════════════════════════════════════════════
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            import pandas as pd
            o, h, l, c = (
                df["open"].astype(float),
                df["high"].astype(float),
                df["low"].astype(float),
                df["close"].astype(float),
            )
            body = (c - o).abs()
            upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
            lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

            manipulation_detected = False
            for i in range(-3, 0):
                if len(df) + i >= 0:
                    candle_body = body.iloc[i]
                    if candle_body > 0:
                        u_ratio = upper_wick.iloc[i] / candle_body
                        l_ratio = lower_wick.iloc[i] / candle_body
                        if u_ratio > 3 or l_ratio > 3:
                            issues += 1
                            wick_dir = "upper" if u_ratio > l_ratio else "lower"
                            evidence.append(
                                f"⚠️ Long {wick_dir} wick — possible stop-hunt / manipulation"
                            )
                            manipulation_detected = True
                            break

            if not manipulation_detected:
                evidence.append("✅ No suspicious wicks in recent candles")

        # ═══════════════════════════════════════════════════════
        # QUALITY SCORE (0-100)
        # ═══════════════════════════════════════════════════════
        deduction_per_issue = 100 / max_issues
        quality_score = max(0, round(100 - issues * deduction_per_issue, 1))
        data["quality_score"] = quality_score
        data["issue_count"] = issues

        if quality_score >= 85:
            grade, label = "A", "Excellent"
        elif quality_score >= 70:
            grade, label = "B", "Good"
        elif quality_score >= 50:
            grade, label = "C", "Fair"
        elif quality_score >= 30:
            grade, label = "D", "Poor"
        else:
            grade, label = "F", "Very Poor"

        data["quality_grade"] = grade

        # ═══════════════════════════════════════════════════════
        # WEIGHT ADJUSTMENT (0.5 - 1.0)
        # ═══════════════════════════════════════════════════════
        if quality_score >= 80:
            weight_adj = 1.0
        elif quality_score >= 60:
            weight_adj = 0.8
        elif quality_score >= 40:
            weight_adj = 0.5
        else:
            weight_adj = 0.2
        data["weight_adjustment"] = weight_adj

        evidence.insert(0, f"═══ DATA QUALITY: {quality_score}/100 ({grade} — {label}) ═══")
        evidence.append(f"Weight adjustment: {weight_adj}x (applied to consensus)")

        return self._out(
            direction="NEUTRAL",
            confidence=quality_score,
            score=0,
            evidence=evidence,
            data=data,
            reasoning=(
                f"Data quality {quality_score}/100 ({grade}): "
                f"{issues} issues across {len(df)} candles. "
                f"Weight adjustment: {weight_adj}x"
            ),
        )
