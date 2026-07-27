"""
Data Quality Agent
==================
Validates data completeness, gaps, anomalies before analysis.
Only produces evidence. Never outputs BUY/SELL.
"""
from hqip.agents.base import BaseAgent, AgentOutput
import numpy as np

class DataQualityAgent(BaseAgent):
    name = "DataQuality"
    weight = 0.5

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 20:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"], reasoning="Not enough candles")

        evidence = []
        score = 0.0

        # Check completeness
        expected_rows = len(df)
        null_count = df.isnull().sum().sum()
        completeness = 1 - (null_count / (expected_rows * len(df.columns)))
        evidence.append(f"Data completeness: {completeness*100:.1f}%")

        # Check for gaps
        if "timestamp" in df.columns:
            ts = df["timestamp"].diff()
            median_gap = ts.median()
            max_gap = ts.max()
            if max_gap > median_gap * 3:
                evidence.append(f"⚠️ Large gap detected: {max_gap} vs median {median_gap}")
                score -= 0.2
            else:
                evidence.append(f"Timestamps regular (median gap: {median_gap})")

        # Check for anomalies (price spikes)
        if "close" in df.columns:
            rets = df["close"].pct_change().abs()
            spike_threshold = rets.mean() + 3 * rets.std()
            spikes = (rets > spike_threshold).sum()
            evidence.append(f"Price spikes (>3σ): {spikes}")
            if spikes > 5:
                score -= 0.15
                evidence.append("⚠️ Many price spikes detected")

        # Volume consistency
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            evidence.append(f"Zero-volume bars: {zero_vol}")
            if zero_vol > len(df) * 0.1:
                score -= 0.1

        confidence = max(0, min(100, completeness * 80 + (10 if score > -0.1 else 0)))

        return self._out(
            direction="NEUTRAL",
            confidence=confidence,
            score=score,
            evidence=evidence,
            reasoning=f"Data quality check: {completeness*100:.0f}% complete, {len(df)} candles"
        )
