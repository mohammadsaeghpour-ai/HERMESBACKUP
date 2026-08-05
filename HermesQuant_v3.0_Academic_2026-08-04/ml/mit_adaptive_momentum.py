"""
MIT Adaptive Momentum — Andrew Lo (Adaptive Markets Hypothesis)
Market adapts like evolution — momentum changes based on regime
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind


class AdaptiveMomentum:
    """
    Adaptive momentum based on Andrew Lo's Adaptive Markets Hypothesis:
    1. Detect market regime (bull/bear/sideways)
    2. Adjust momentum lookback based on regime
    3. Use different strategies for different regimes
    
    Key insight: Markets are not efficient — they adapt like species
    """
    
    def __init__(self):
        self.regime_history = []
    
    def detect_regime(self, df, lookback=50):
        """
        Detect market regime using:
        1. Trend strength (ADX)
        2. Volatility clustering (GARCH-like)
        3. Mean reversion speed
        """
        if df is None or len(df) < lookback:
            return "unknown", 0
        
        # ADX for trend strength
        adx_v, _, _ = ind.adx(df)
        adx = adx_v.iloc[-1]
        
        # Volatility
        returns = df["close"].pct_change()
        vol = returns.std() * np.sqrt(252) * 100
        
        # Mean reversion (Hurst exponent approximation)
        prices = df["close"].values[-lookback:]
        lags = range(2, min(20, len(prices)))
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        
        if len(tau) < 2 or any(t == 0 for t in tau):
            hurst = 0.5
        else:
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            hurst = poly[0]
        
        # Regime detection
        if adx > 25 and hurst > 0.5:
            regime = "trending"
            strength = adx / 100
        elif adx < 20 and hurst < 0.5:
            regime = "mean_reverting"
            strength = 1 - hurst
        else:
            regime = "sideways"
            strength = 0.5
        
        return regime, strength
    
    def adaptive_lookback(self, regime, base_lookback=20):
        """
        Adjust lookback based on regime:
        - Trending: longer lookback (ride the trend)
        - Mean-reverting: shorter lookback (quick reversals)
        - Sideways: medium lookback
        """
        multipliers = {
            "trending": 1.5,
            "mean_reverting": 0.5,
            "sideways": 1.0,
            "unknown": 1.0
        }
        
        return int(base_lookback * multipliers.get(regime, 1.0))
    
    def analyze(self, df, symbol="", timeframe=""):
        """
        Adaptive momentum analysis:
        1. Detect regime
        2. Adjust parameters
        3. Generate signal
        """
        if df is None or len(df) < 50:
            return {"direction": "NEUTRAL", "confidence": 0, "regime": "unknown"}
        
        # Detect regime
        regime, strength = self.detect_regime(df)
        
        # Adaptive lookback
        lookback = self.adaptive_lookback(regime, 20)
        
        # Momentum with adaptive lookback
        returns = df["close"].pct_change(lookback).iloc[-1]
        
        # Volatility-adjusted momentum
        vol = df["close"].pct_change().rolling(20).std().iloc[-1]
        adj_momentum = returns / (vol + 1e-10)
        
        # RSI for confirmation
        rsi = ind.rsi(df, 14).iloc[-1]
        
        # Signal
        if regime == "trending":
            if adj_momentum > 0.5 and rsi < 70:
                direction = "BUY"
                confidence = min(abs(adj_momentum) * 30, 90)
            elif adj_momentum < -0.5 and rsi > 30:
                direction = "SELL"
                confidence = min(abs(adj_momentum) * 30, 90)
            else:
                direction = "NEUTRAL"
                confidence = 0
        
        elif regime == "mean_reverting":
            if rsi < 30:
                direction = "BUY"
                confidence = (30 - rsi) * 2
            elif rsi > 70:
                direction = "SELL"
                confidence = (rsi - 70) * 2
            else:
                direction = "NEUTRAL"
                confidence = 0
        
        else:  # sideways
            direction = "NEUTRAL"
            confidence = 0
        
        return {
            "direction": direction,
            "confidence": confidence,
            "regime": regime,
            "strength": strength,
            "lookback": lookback,
            "adj_momentum": adj_momentum,
        }
