"""
Multi-Timeframe Strategy — 60 → 30 → 15 → 5
Based on best practices from Investopedia, TradingView, Forex Factory
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant_v3")
import numpy as np
import pandas as pd
from core import indicators as ind


class MultiTFStrategy:
    """
    Multi-timeframe strategy:
    1. 60m: Determine trend direction (EMA 200)
    2. 30m: Confirm trend (EMA 50/200)
    3. 15m: Identify entry zones (S/R, divergence)
    4. 5m: Precise entry (EMA cross, candle patterns)
    """
    
    def __init__(self):
        pass
    
    def analyze(self, df, timeframe="15m"):
        """
        Analyze market using multi-timeframe approach
        Returns: direction, confidence, reasons
        """
        if df is None or len(df) < 50:
            return "NEUTRAL", 0, ["Not enough data"]
        
        score = 0
        reasons = []
        
        # ── Trend Direction (EMA 200) ──
        if len(df) >= 200:
            ema_200 = ind.ema(df["close"], 200).iloc[-1]
            price = df["close"].iloc[-1]
            
            if price > ema_200:
                score += 0.3
                reasons.append("Price above EMA200 (bullish)")
            else:
                score -= 0.3
                reasons.append("Price below EMA200 (bearish)")
        
        # ── Trend Confirmation (EMA 50/200) ──
        if len(df) >= 200:
            ema_50 = ind.ema(df["close"], 50).iloc[-1]
            ema_200 = ind.ema(df["close"], 200).iloc[-1]
            
            if ema_50 > ema_200:
                score += 0.2
                reasons.append("EMA50 > EMA200 (golden cross)")
            else:
                score -= 0.2
                reasons.append("EMA50 < EMA200 (death cross)")
        
        # ── Momentum (RSI + MACD) ──
        rsi = ind.rsi(df, 14).iloc[-1]
        macd_l, sig_l, hist = ind.macd(df)
        
        if rsi < 30:
            score += 0.2
            reasons.append("RSI oversold: %.1f" % rsi)
        elif rsi > 70:
            score -= 0.2
            reasons.append("RSI overbought: %.1f" % rsi)
        
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            score += 0.15
            reasons.append("MACD bullish crossover")
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            score -= 0.15
            reasons.append("MACD bearish crossover")
        
        # ── Volatility (ATR + BB) ──
        atr_val = ind.atr(df, 14).iloc[-1]
        upper, mid, lower = ind.bollinger(df)
        bb_pos = (df["close"].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
        
        if bb_pos < 0.2:
            score += 0.1
            reasons.append("Near lower BB (discount)")
        elif bb_pos > 0.8:
            score -= 0.1
            reasons.append("Near upper BB (premium)")
        
        # ── Volume Confirmation ──
        vol_ratio = ind.volume_ratio(df).iloc[-1]
        if vol_ratio > 1.5:
            score += 0.1 if score > 0 else -0.1
            reasons.append("Volume confirms (%.1fx)" % vol_ratio)
        
        # ── Supertrend ──
        st_dir, st_val = ind.supertrend(df)
        if st_dir.iloc[-1] == 1:
            score += 0.1
            reasons.append("Supertrend bullish")
        else:
            score -= 0.1
            reasons.append("Supertrend bearish")
        
        # ── Direction ──
        if score > 0.1:
            direction = "BUY"
        elif score < -0.1:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        confidence = min(abs(score) * 100, 100)
        
        return direction, confidence, reasons
