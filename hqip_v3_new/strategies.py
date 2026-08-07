"""
HQIP Multi-Timeframe Strategy Engine
======================================
Each timeframe gets its OWN strategy optimized for that timeframe.
Signals are independent per timeframe.
"""
from hqip.agents.base import BaseAgent
import numpy as np


class TimeframeStrategy:
    """Base class for timeframe-specific strategies."""
    
    def __init__(self, name, tf):
        self.name = name
        self.tf = tf
    
    def analyze(self, df, symbol=""):
        raise NotImplementedError


class ScalpingStrategy15m(TimeframeStrategy):
    """
    15-Minute SCALPING Strategy
    ============================
    Goal: Catch $100-400 moves on BTC within 1-4 hours
    Approach: Mean reversion + momentum + support/resistance
    
    Key indicators:
    - RSI oversold/overbought for entries
    - Bollinger Bands for mean reversion
    - EMA crossover for direction
    - Volume confirmation
    - ATR for SL/TP sizing
    
    SL: 0.3-0.5% of price (tight)
    TP: 0.5-1.0% of price
    Hold time: 1-4 hours
    """
    
    def __init__(self):
        super().__init__("Scalping15m", "15m")
    
    def analyze(self, df, symbol="BTCUSDT"):
        if df is None or df.empty or len(df) < 30:
            return {"direction": "NO_TRADE", "confidence": 0, "evidence": ["Insufficient data"]}
        
        evidence = []
        score = 0
        price = float(df.iloc[-1]["close"])
        
        # 1. RSI — Mean Reversion
        rsi = df.get("rsi", None)
        if rsi is not None and len(rsi) > 0:
            current_rsi = float(rsi.iloc[-1])
            if current_rsi < 30:
                score += 0.3
                evidence.append(f"🟢 RSI oversold ({current_rsi:.0f}) — bounce expected")
            elif current_rsi > 70:
                score -= 0.3
                evidence.append(f"🔴 RSI overbought ({current_rsi:.0f}) — pullback expected")
            elif current_rsi < 40:
                score += 0.15
                evidence.append(f"🟡 RSI low ({current_rsi:.0f}) — approaching oversold")
            elif current_rsi > 60:
                score -= 0.15
                evidence.append(f"🟡 RSI high ({current_rsi:.0f}) — approaching overbought")
        
        # 2. Bollinger Bands — Mean Reversion
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            bb_lower = float(df.iloc[-1]["bb_lower"])
            bb_upper = float(df.iloc[-1]["bb_upper"])
            bb_mid = (bb_lower + bb_upper) / 2
            bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
            
            if price <= bb_lower * 1.002:
                score += 0.25
                evidence.append(f"🟢 Price at lower BB — mean reversion buy")
            elif price >= bb_upper * 0.998:
                score -= 0.25
                evidence.append(f"🔴 Price at upper BB — mean reversion sell")
            
            # Squeeze detection
            if bb_width < 0.02:
                evidence.append(f"⚡ BB Squeeze detected — breakout imminent")
        
        # 3. EMA Crossover — Direction
        if 'ema9' in df.columns and 'ema21' in df.columns:
            ema9 = float(df.iloc[-1]["ema9"])
            ema21 = float(df.iloc[-1]["ema21"])
            ema9_prev = float(df.iloc[-2]["ema9"])
            ema21_prev = float(df.iloc[-2]["ema21"])
            
            # Fresh crossover
            if ema9 > ema21 and ema9_prev <= ema21_prev:
                score += 0.3
                evidence.append(f"🟢 EMA 9/21 bullish CROSS — momentum shift up")
            elif ema9 < ema21 and ema9_prev >= ema21_prev:
                score -= 0.3
                evidence.append(f"🔴 EMA 9/21 bearish CROSS — momentum shift down")
            elif ema9 > ema21:
                score += 0.1
                evidence.append(f"🟢 EMA 9 above 21 — bullish")
            elif ema9 < ema21:
                score -= 0.1
                evidence.append(f"🔴 EMA 9 below 21 — bearish")
        
        # 4. Volume Confirmation
        if 'volume' in df.columns:
            vol = df["volume"].values
            avg_vol = np.mean(vol[-20:])
            current_vol = vol[-1]
            if current_vol > avg_vol * 1.5:
                evidence.append(f"📊 Volume spike ({current_vol/avg_vol:.1f}x) — confirms move")
                score *= 1.2  # Boost score with volume
        
        # 5. ATR-based SL/TP
        if 'atr' in df.columns:
            atr = float(df.iloc[-1]["atr"])
            atr_pct = atr / price * 100
            sl_pct = max(0.3, min(0.5, atr_pct * 0.5))
            tp_pct = max(0.5, min(1.0, atr_pct * 0.8))
        else:
            sl_pct = 0.4
            tp_pct = 0.7
        
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NO_TRADE"
        confidence = min(100, abs(score) * 150 + 30)
        
        # Calculate entry/SL/TP
        if direction == "BUY":
            entry = price
            sl = price * (1 - sl_pct/100)
            tp1 = price * (1 + tp_pct/100)
            tp2 = price * (1 + tp_pct*1.5/100)
            tp3 = price * (1 + tp_pct*2.5/100)
        elif direction == "SELL":
            entry = price
            sl = price * (1 + sl_pct/100)
            tp1 = price * (1 - tp_pct/100)
            tp2 = price * (1 - tp_pct*1.5/100)
            tp3 = price * (1 - tp_pct*2.5/100)
        else:
            entry = sl = tp1 = tp2 = tp3 = price
        
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "sl_pct": round(sl_pct, 2),
            "tp_pct": round(tp_pct, 2),
            "evidence": evidence,
            "strategy": self.name,
            "timeframe": self.tf,
        }


class DayTradingStrategy1h(TimeframeStrategy):
    """
    1-Hour DAY TRADING Strategy
    ============================
    Goal: Catch $200-800 moves on BTC within 4-12 hours
    Approach: Trend following + breakout + structure
    
    Key indicators:
    - EMA 20/50 for trend
    - MACD for momentum confirmation
    - Support/Resistance for entries
    - Volume Profile for conviction
    - ATR for sizing
    
    SL: 0.5-1.0% of price
    TP: 1.0-2.0% of price
    Hold time: 4-12 hours
    """
    
    def __init__(self):
        super().__init__("DayTrading1h", "1h")
    
    def analyze(self, df, symbol="BTCUSDT"):
        if df is None or df.empty or len(df) < 30:
            return {"direction": "NO_TRADE", "confidence": 0, "evidence": ["Insufficient data"]}
        
        evidence = []
        score = 0
        price = float(df.iloc[-1]["close"])
        
        # 1. EMA Trend — 20/50
        if 'ema20' in df.columns and 'ema50' in df.columns:
            ema20 = float(df.iloc[-1]["ema20"])
            ema50 = float(df.iloc[-1]["ema50"])
            price_ema20_dist = (price - ema20) / ema20 * 100
            
            if ema20 > ema50 and price > ema20:
                score += 0.3
                evidence.append(f"🟢 Strong uptrend: EMA20 > EMA50, price above both")
            elif ema20 < ema50 and price < ema20:
                score -= 0.3
                evidence.append(f"🔴 Strong downtrend: EMA20 < EMA50, price below both")
            elif ema20 > ema50:
                score += 0.15
                evidence.append(f"🟡 Bullish trend but price below EMA20 — pullback entry?")
            elif ema20 < ema50:
                score -= 0.15
                evidence.append(f"🟡 Bearish trend but price above EMA20 — bounce entry?")
        
        # 2. MACD — Momentum
        if 'macd_hist' in df.columns:
            macd_hist = float(df.iloc[-1]["macd_hist"])
            macd_hist_prev = float(df.iloc[-2]["macd_hist"])
            
            if macd_hist > 0 and macd_hist_prev <= 0:
                score += 0.25
                evidence.append(f"🟢 MACD histogram turned positive — momentum shift")
            elif macd_hist < 0 and macd_hist_prev >= 0:
                score -= 0.25
                evidence.append(f"🔴 MACD histogram turned negative — momentum shift")
            elif macd_hist > 0:
                score += 0.1
            elif macd_hist < 0:
                score -= 0.1
        
        # 3. ADX — Trend Strength
        if 'adx' in df.columns:
            adx = float(df.iloc[-1]["adx"])
            if adx > 25:
                evidence.append(f"📊 ADX={adx:.0f} — strong trend (good for trend following)")
                score *= 1.15
            elif adx < 15:
                evidence.append(f"📊 ADX={adx:.0f} — weak trend (avoid trend trades)")
                score *= 0.8
        
        # 4. Volume Profile — Conviction
        if 'vwap' in df.columns:
            vwap = float(df.iloc[-1]["vwap"])
            if price > vwap:
                score += 0.1
                evidence.append(f"🟢 Price above VWAP — bullish control")
            else:
                score -= 0.1
                evidence.append(f"🔴 Price below VWAP — bearish control")
        
        # 5. ATR-based SL/TP
        if 'atr' in df.columns:
            atr = float(df.iloc[-1]["atr"])
            atr_pct = atr / price * 100
            sl_pct = max(0.5, min(1.0, atr_pct * 0.8))
            tp_pct = max(1.0, min(2.0, atr_pct * 1.5))
        else:
            sl_pct = 0.7
            tp_pct = 1.3
        
        direction = "BUY" if score > 0.2 else "SELL" if score < -0.2 else "NO_TRADE"
        confidence = min(100, abs(score) * 120 + 35)
        
        if direction == "BUY":
            entry = price
            sl = price * (1 - sl_pct/100)
            tp1 = price * (1 + tp_pct/100)
            tp2 = price * (1 + tp_pct*1.5/100)
            tp3 = price * (1 + tp_pct*2.5/100)
        elif direction == "SELL":
            entry = price
            sl = price * (1 + sl_pct/100)
            tp1 = price * (1 - tp_pct/100)
            tp2 = price * (1 - tp_pct*1.5/100)
            tp3 = price * (1 - tp_pct*2.5/100)
        else:
            entry = sl = tp1 = tp2 = tp3 = price
        
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "sl_pct": round(sl_pct, 2),
            "tp_pct": round(tp_pct, 2),
            "evidence": evidence,
            "strategy": self.name,
            "timeframe": self.tf,
        }


class SwingStrategy4h(TimeframeStrategy):
    """
    4-Hour SWING Strategy
    ======================
    Goal: Catch $500-2000 moves on BTC within 1-5 days
    Approach: Structure + supply/demand + Wyckoff
    
    SL: 1.0-2.0% of price
    TP: 2.0-5.0% of price
    Hold time: 1-5 days
    """
    
    def __init__(self):
        super().__init__("Swing4h", "4h")
    
    def analyze(self, df, symbol="BTCUSDT"):
        if df is None or df.empty or len(df) < 30:
            return {"direction": "NO_TRADE", "confidence": 0, "evidence": ["Insufficient data"]}
        
        evidence = []
        score = 0
        price = float(df.iloc[-1]["close"])
        
        # 1. Multi-EMA Stack
        emas = {}
        for period in [8, 21, 50]:
            col = f"ema{period}"
            if col in df.columns:
                emas[period] = float(df.iloc[-1][col])
        
        if len(emas) == 3:
            if emas[8] > emas[21] > emas[50]:
                score += 0.35
                evidence.append(f"🟢 Perfect bullish EMA stack: 8>{emas[8]:.0f} > 21>{emas[21]:.0f} > 50>{emas[50]:.0f}")
            elif emas[8] < emas[21] < emas[50]:
                score -= 0.35
                evidence.append(f"🔴 Perfect bearish EMA stack: 8<{emas[8]:.0f} < 21<{emas[21]:.0f} < 50<{emas[50]:.0f}")
            elif emas[8] > emas[21]:
                score += 0.1
                evidence.append(f"🟡 Short-term bullish (8 > 21)")
            elif emas[8] < emas[21]:
                score -= 0.1
                evidence.append(f"🔴 Short-term bearish (8 < 21)")
        
        # 2. RSI with trend context
        if 'rsi' in df.columns:
            rsi = float(df.iloc[-1]["rsi"])
            if rsi < 35 and score > 0:
                score += 0.2
                evidence.append(f"🟢 RSI oversold ({rsi:.0f}) in uptrend — strong buy")
            elif rsi > 65 and score < 0:
                score -= 0.2
                evidence.append(f"🔴 RSI overbought ({rsi:.0f}) in downtrend — strong sell")
            elif rsi < 30:
                score += 0.15
                evidence.append(f"🟢 RSI deeply oversold ({rsi:.0f})")
            elif rsi > 70:
                score -= 0.15
                evidence.append(f"🔴 RSI deeply overbought ({rsi:.0f})")
        
        # 3. SuperTrend
        if 'supertrend' in df.columns:
            st = float(df.iloc[-1]["supertrend"])
            if price > st:
                score += 0.15
                evidence.append(f"🟢 Price above SuperTrend (${st:,.0f}) — bullish")
            else:
                score -= 0.15
                evidence.append(f"🔴 Price below SuperTrend (${st:,.0f}) — bearish")
        
        # 4. ATR-based SL/TP
        if 'atr' in df.columns:
            atr = float(df.iloc[-1]["atr"])
            atr_pct = atr / price * 100
            sl_pct = max(1.0, min(2.0, atr_pct * 1.0))
            tp_pct = max(2.0, min(5.0, atr_pct * 2.5))
        else:
            sl_pct = 1.5
            tp_pct = 3.0
        
        direction = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
        confidence = min(100, abs(score) * 100 + 40)
        
        if direction == "BUY":
            entry = price
            sl = price * (1 - sl_pct/100)
            tp1 = price * (1 + tp_pct/100)
            tp2 = price * (1 + tp_pct*1.5/100)
            tp3 = price * (1 + tp_pct*2.5/100)
        elif direction == "SELL":
            entry = price
            sl = price * (1 + sl_pct/100)
            tp1 = price * (1 - tp_pct/100)
            tp2 = price * (1 - tp_pct*1.5/100)
            tp3 = price * (1 - tp_pct*2.5/100)
        else:
            entry = sl = tp1 = tp2 = tp3 = price
        
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "sl_pct": round(sl_pct, 2),
            "tp_pct": round(tp_pct, 2),
            "evidence": evidence,
            "strategy": self.name,
            "timeframe": self.tf,
        }


class PositionStrategy1d(TimeframeStrategy):
    """
    1-Day POSITION Strategy
    ========================
    Goal: Catch $2000-10000 moves on BTC within 1-4 weeks
    Approach: Macro trend + fundamentals + market structure
    
    SL: 2.0-5.0% of price
    TP: 5.0-15.0% of price
    Hold time: 1-4 weeks
    """
    
    def __init__(self):
        super().__init__("Position1d", "1d")
    
    def analyze(self, df, symbol="BTCUSDT"):
        if df is None or df.empty or len(df) < 30:
            return {"direction": "NO_TRADE", "confidence": 0, "evidence": ["Insufficient data"]}
        
        evidence = []
        score = 0
        price = float(df.iloc[-1]["close"])
        
        # 1. Weekly trend
        if 'ema50' in df.columns and 'ema200' in df.columns:
            ema50 = float(df.iloc[-1]["ema50"])
            ema200 = float(df.iloc[-1]["ema200"])
            
            if ema50 > ema200 and price > ema50:
                score += 0.35
                evidence.append(f"🟢 Macro uptrend: EMA50 > EMA200, price above both")
            elif ema50 < ema200 and price < ema50:
                score -= 0.35
                evidence.append(f"🔴 Macro downtrend: EMA50 < EMA200, price below both")
            elif ema50 > ema200:
                score += 0.15
                evidence.append(f"🟡 Long-term bullish but short-term pullback")
            elif ema50 < ema200:
                score -= 0.15
                evidence.append(f"🔴 Long-term bearish but short-term bounce")
        
        # 2. Monthly RSI
        if 'rsi' in df.columns:
            rsi = float(df.iloc[-1]["rsi"])
            if rsi < 40:
                score += 0.2
                evidence.append(f"🟢 Daily RSI oversold ({rsi:.0f}) — accumulation zone")
            elif rsi > 60:
                score -= 0.2
                evidence.append(f"🔴 Daily RSI overbought ({rsi:.0f}) — distribution zone")
        
        # 3. ATR-based SL/TP
        if 'atr' in df.columns:
            atr = float(df.iloc[-1]["atr"])
            atr_pct = atr / price * 100
            sl_pct = max(2.0, min(5.0, atr_pct * 1.2))
            tp_pct = max(5.0, min(15.0, atr_pct * 4.0))
        else:
            sl_pct = 3.0
            tp_pct = 8.0
        
        direction = "BUY" if score > 0.2 else "SELL" if score < -0.2 else "NO_TRADE"
        confidence = min(100, abs(score) * 80 + 45)
        
        if direction == "BUY":
            entry = price
            sl = price * (1 - sl_pct/100)
            tp1 = price * (1 + tp_pct/100)
            tp2 = price * (1 + tp_pct*1.5/100)
            tp3 = price * (1 + tp_pct*2.5/100)
        elif direction == "SELL":
            entry = price
            sl = price * (1 + sl_pct/100)
            tp1 = price * (1 - tp_pct/100)
            tp2 = price * (1 - tp_pct*1.5/100)
            tp3 = price * (1 - tp_pct*2.5/100)
        else:
            entry = sl = tp1 = tp2 = tp3 = price
        
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "sl_pct": round(sl_pct, 2),
            "tp_pct": round(tp_pct, 2),
            "evidence": evidence,
            "strategy": self.name,
            "timeframe": self.tf,
        }


def get_all_strategies():
    """Return all timeframe strategies."""
    return {
        "15m": ScalpingStrategy15m(),
        "1h": DayTradingStrategy1h(),
        "4h": SwingStrategy4h(),
        "1d": PositionStrategy1d(),
    }
