"""
Feature Engineering Agent
=========================
Creates ML-ready features from raw indicators.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class FeatureEngineeringAgent(BaseAgent):
    name = "FeatureEngineering"
    weight = 0.3

    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        features = {}

        # Key features for ML
        features["rsi"] = df["rsi"].iloc[-1]
        features["rsi_change"] = df["rsi"].iloc[-1] - df["rsi"].iloc[-2] if len(df) > 1 else 0
        features["macd_hist"] = df["macd_hist"].iloc[-1]
        features["macd_hist_change"] = df["macd_hist"].iloc[-1] - df["macd_hist"].iloc[-2] if len(df) > 1 else 0
        features["adx"] = df["adx"].iloc[-1]
        features["bb_pct"] = df["bb_pct"].iloc[-1]
        features["bb_width"] = df["bb_width"].iloc[-1]
        features["atr_pct"] = df["atr_pct"].iloc[-1]
        features["vol_ratio"] = df["vol_ratio"].iloc[-1]
        features["vwap_dist"] = df["vwap_dist"].iloc[-1]
        features["stoch_k"] = df["stoch_k"].iloc[-1]
        features["stoch_d"] = df["stoch_d"].iloc[-1]
        features["roc"] = df["roc"].iloc[-1]
        features["price_vs_ema20"] = (df["close"].iloc[-1] / df["ema20"].iloc[-1] - 1) * 100
        features["price_vs_ema50"] = (df["close"].iloc[-1] / df["ema50"].iloc[-1] - 1) * 100
        features["plus_di"] = df["plus_di"].iloc[-1]
        features["minus_di"] = df["minus_di"].iloc[-1]
        features["bullish_candle"] = float(df["bullish_candle"].iloc[-1])
        features["body_pct"] = df["body_pct"].iloc[-1]
        features["obv_trend"] = 1.0 if df["obv"].iloc[-1] > df["obv_ema"].iloc[-1] else -1.0
        features["st_dir"] = float(df["st_dir"].iloc[-1])

        evidence.append(f"Generated {len(features)} features")
        evidence.append(f"Key: RSI={features['rsi']:.0f}, ADX={features['adx']:.0f}, BB%={features['bb_pct']:.2f}")

        return self._out(
            direction="NEUTRAL",
            confidence=100,
            score=0,
            evidence=evidence,
            data=features,
            reasoning=f"Feature engineering complete: {len(features)} features"
        )
