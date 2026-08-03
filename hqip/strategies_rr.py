"""
HQIP RISK-REWARD Optimized Strategies
======================================
Core rule: $1 risk → $2-4 reward (R:R = 1:2 to 1:4)
With 20x leverage:
  SL = 0.5% price move = $1 loss on $10
  TP1 = 1.0% price move = $2 profit on $10
  TP2 = 1.5% price move = $3 profit on $10
  TP3 = 2.0% price move = $4 profit on $10
"""
import numpy as np


def analyze_15m(df, symbol="BTCUSDT"):
    """15m Scalping — Catch $200-800 moves. R:R = 1:2 minimum."""
    if df is None or df.empty or len(df) < 30:
        return {"direction": "NO_TRADE", "confidence": 0, "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0}
    
    evidence = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # ── 1. RSI — Mean Reversion ──
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 30:
            score += 0.35
            evidence.append(f"🟢 RSI={rsi:.0f} خیلی پایین — اشباع فروش")
        elif rsi > 70:
            score -= 0.35
            evidence.append(f"🔴 RSI={rsi:.0f} خیلی بالا — اشباع خرید")
        elif rsi < 40:
            score += 0.15
            evidence.append(f"🟡 RSI={rsi:.0f} نزدیک کف")
        elif rsi > 60:
            score -= 0.15
            evidence.append(f"🟡 RSI={rsi:.0f} نزدیک سقف")
    
    # ── 2. Bollinger Bands — Mean Reversion ──
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        bb_lower = float(df.iloc[-1]["bb_lower"])
        bb_upper = float(df.iloc[-1]["bb_upper"])
        
        if price <= bb_lower * 1.003:
            score += 0.30
            evidence.append(f"🟢 قیمت روی باند پایین — برگشت محتمل")
        elif price >= bb_upper * 0.997:
            score -= 0.30
            evidence.append(f"🔴 قیمت روی باند بالا — اصلاح محتمل")
    
    # ── 3. EMA 9/21 Crossover ──
    if 'ema9' in df.columns and 'ema21' in df.columns:
        ema9 = float(df.iloc[-1]["ema9"])
        ema21 = float(df.iloc[-1]["ema21"])
        ema9_prev = float(df.iloc[-2]["ema9"])
        ema21_prev = float(df.iloc[-2]["ema21"])
        
        if ema9 > ema21 and ema9_prev <= ema21_prev:
            score += 0.30
            evidence.append(f"🟢 کراس صعودی EMA 9/21")
        elif ema9 < ema21 and ema9_prev >= ema21_prev:
            score -= 0.30
            evidence.append(f"🔴 کراس نزولی EMA 9/21")
        elif ema9 > ema21:
            score += 0.10
        elif ema9 < ema21:
            score -= 0.10
    
    # ── 4. Volume Confirmation ──
    if 'volume' in df.columns:
        vol = df["volume"].values
        avg_vol = np.mean(vol[-20:])
        if vol[-1] > avg_vol * 1.5:
            evidence.append(f"📊 حجم بالا ({vol[-1]/avg_vol:.1f}x) — تأیید حرکت")
            score *= 1.15
    
    # ── 5. MACD Histogram ──
    if 'macd_hist' in df.columns:
        mh = float(df.iloc[-1]["macd_hist"])
        mh_prev = float(df.iloc[-2]["macd_hist"])
        if mh > 0 and mh_prev <= 0:
            score += 0.20
            evidence.append(f"🟢 MACD مثبت شد")
        elif mh < 0 and mh_prev >= 0:
            score -= 0.20
            evidence.append(f"🔴 MACD منفی شد")
    
    # ── Direction & Confidence ──
    direction = "BUY" if score > 0.20 else "SELL" if score < -0.20 else "NO_TRADE"
    confidence = min(100, abs(score) * 120 + 30)
    
    # ── Fixed R:R = 1:2 to 1:4 ──
    # SL = 0.5% of price = $1 loss on $10 with 20x
    sl_pct = 0.50
    tp1_pct = 1.00  # $2 profit
    tp2_pct = 1.50  # $3 profit
    tp3_pct = 2.00  # $4 profit
    
    if direction == "BUY":
        entry = price
        sl = price * (1 - sl_pct / 100)
        tp1 = price * (1 + tp1_pct / 100)
        tp2 = price * (1 + tp2_pct / 100)
        tp3 = price * (1 + tp3_pct / 100)
    elif direction == "SELL":
        entry = price
        sl = price * (1 + sl_pct / 100)
        tp1 = price * (1 - tp1_pct / 100)
        tp2 = price * (1 - tp2_pct / 100)
        tp3 = price * (1 - tp3_pct / 100)
    else:
        entry = sl = tp1 = tp2 = tp3 = price
    
    return {
        "direction": direction, "confidence": round(confidence, 1),
        "entry": round(entry, 2), "sl": round(sl, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "sl_pct": sl_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct, "tp3_pct": tp3_pct,
        "evidence": evidence, "strategy": "Scalping15m", "timeframe": "15m",
    }


def analyze_1h(df, symbol="BTCUSDT"):
    """1h Day Trading — Catch $500-2000 moves. R:R = 1:3."""
    if df is None or df.empty or len(df) < 30:
        return {"direction": "NO_TRADE", "confidence": 0, "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0}
    
    evidence = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # ── 1. EMA Trend 20/50 ──
    if 'ema20' in df.columns and 'ema50' in df.columns:
        ema20 = float(df.iloc[-1]["ema20"])
        ema50 = float(df.iloc[-1]["ema50"])
        
        if ema20 > ema50 and price > ema20:
            score += 0.35
            evidence.append(f"🟢 روند صعودی قوی: EMA20 > EMA50")
        elif ema20 < ema50 and price < ema20:
            score -= 0.35
            evidence.append(f"🔴 روند نزولی قوی: EMA20 < EMA50")
        elif ema20 > ema50:
            score += 0.10
            evidence.append(f"🟡 صعودی ولی قیمت زیر EMA20")
        elif ema20 < ema50:
            score -= 0.10
            evidence.append(f"🔴 نزولی ولی قیمت بالای EMA20")
    
    # ── 2. MACD ──
    if 'macd_hist' in df.columns:
        mh = float(df.iloc[-1]["macd_hist"])
        mh_prev = float(df.iloc[-2]["macd_hist"])
        if mh > 0 and mh_prev <= 0:
            score += 0.25
            evidence.append(f"🟢 MACD کراس صعودی")
        elif mh < 0 and mh_prev >= 0:
            score -= 0.25
            evidence.append(f"🔴 MACD کراس نزولی")
        elif mh > 0:
            score += 0.10
        elif mh < 0:
            score -= 0.10
    
    # ── 3. ADX — Trend Strength ──
    if 'adx' in df.columns:
        adx = float(df.iloc[-1]["adx"])
        if adx > 25:
            evidence.append(f"📊 ADX={adx:.0f} — روند قوی")
            score *= 1.15
        elif adx < 15:
            evidence.append(f"📊 ADX={adx:.0f} — بازار رِنج")
            score *= 0.7
    
    # ── 4. RSI ──
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 35:
            score += 0.20
            evidence.append(f"🟢 RSI={rsi:.0f} اشباع فروش")
        elif rsi > 65:
            score -= 0.20
            evidence.append(f"🔴 RSI={rsi:.0f} اشباع خرید")
    
    # ── 5. VWAP ──
    if 'vwap' in df.columns:
        vwap = float(df.iloc[-1]["vwap"])
        if price > vwap:
            score += 0.10
            evidence.append(f"🟢 بالای VWAP")
        else:
            score -= 0.10
            evidence.append(f"🔴 زیر VWAP")
    
    direction = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    confidence = min(100, abs(score) * 100 + 40)
    
    # R:R = 1:3 → SL=0.5%, TP=1.5%
    sl_pct = 0.50
    tp1_pct = 1.50
    tp2_pct = 2.25
    tp3_pct = 3.00
    
    if direction == "BUY":
        entry = price
        sl = price * (1 - sl_pct / 100)
        tp1 = price * (1 + tp1_pct / 100)
        tp2 = price * (1 + tp2_pct / 100)
        tp3 = price * (1 + tp3_pct / 100)
    elif direction == "SELL":
        entry = price
        sl = price * (1 + sl_pct / 100)
        tp1 = price * (1 - tp1_pct / 100)
        tp2 = price * (1 - tp2_pct / 100)
        tp3 = price * (1 - tp3_pct / 100)
    else:
        entry = sl = tp1 = tp2 = tp3 = price
    
    return {
        "direction": direction, "confidence": round(confidence, 1),
        "entry": round(entry, 2), "sl": round(sl, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "sl_pct": sl_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct, "tp3_pct": tp3_pct,
        "evidence": evidence, "strategy": "DayTrading1h", "timeframe": "1h",
    }


def analyze_4h(df, symbol="BTCUSDT"):
    """4h Swing — Catch $1000-5000 moves. R:R = 1:4."""
    if df is None or df.empty or len(df) < 30:
        return {"direction": "NO_TRADE", "confidence": 0, "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0}
    
    evidence = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # ── EMA Stack ──
    emas = {}
    for p in [8, 21, 50]:
        col = f"ema{p}"
        if col in df.columns:
            emas[p] = float(df.iloc[-1][col])
    
    if len(emas) == 3:
        if emas[8] > emas[21] > emas[50]:
            score += 0.35
            evidence.append(f"🟢 EMA Stack صعودی کامل")
        elif emas[8] < emas[21] < emas[50]:
            score -= 0.35
            evidence.append(f"🔴 EMA Stack نزولی کامل")
        elif emas[8] > emas[21]:
            score += 0.10
        elif emas[8] < emas[21]:
            score -= 0.10
    
    # ── RSI + SuperTrend ──
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 35:
            score += 0.20
            evidence.append(f"🟢 RSI={rsi:.0f}")
        elif rsi > 65:
            score -= 0.20
            evidence.append(f"🔴 RSI={rsi:.0f}")
    
    if 'supertrend' in df.columns:
        st = float(df.iloc[-1]["supertrend"])
        if price > st:
            score += 0.15
            evidence.append(f"🟢 بالای SuperTrend")
        else:
            score -= 0.15
            evidence.append(f"🔴 زیر SuperTrend")
    
    direction = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    confidence = min(100, abs(score) * 100 + 45)
    
    # R:R = 1:4 → SL=0.5%, TP=2.0%
    sl_pct = 0.50
    tp1_pct = 2.00
    tp2_pct = 3.00
    tp3_pct = 4.00
    
    if direction == "BUY":
        entry = price
        sl = price * (1 - sl_pct / 100)
        tp1 = price * (1 + tp1_pct / 100)
        tp2 = price * (1 + tp2_pct / 100)
        tp3 = price * (1 + tp3_pct / 100)
    elif direction == "SELL":
        entry = price
        sl = price * (1 + sl_pct / 100)
        tp1 = price * (1 - tp1_pct / 100)
        tp2 = price * (1 - tp2_pct / 100)
        tp3 = price * (1 - tp3_pct / 100)
    else:
        entry = sl = tp1 = tp2 = tp3 = price
    
    return {
        "direction": direction, "confidence": round(confidence, 1),
        "entry": round(entry, 2), "sl": round(sl, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "sl_pct": sl_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct, "tp3_pct": tp3_pct,
        "evidence": evidence, "strategy": "Swing4h", "timeframe": "4h",
    }


def analyze_1d(df, symbol="BTCUSDT"):
    """1d Position — Catch $2000+ moves. R:R = 1:5."""
    if df is None or df.empty or len(df) < 30:
        return {"direction": "NO_TRADE", "confidence": 0, "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0}
    
    evidence = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    if 'ema50' in df.columns and 'ema200' in df.columns:
        ema50 = float(df.iloc[-1]["ema50"])
        ema200 = float(df.iloc[-1]["ema200"])
        if ema50 > ema200 and price > ema50:
            score += 0.40
            evidence.append(f"🟢 روند بلندمدت صعودی")
        elif ema50 < ema200 and price < ema50:
            score -= 0.40
            evidence.append(f"🔴 روند بلندمدت نزولی")
    
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 40:
            score += 0.20
            evidence.append(f"🟢 RSI روزانه={rsi:.0f}")
        elif rsi > 60:
            score -= 0.20
            evidence.append(f"🔴 RSI روزانه={rsi:.0f}")
    
    direction = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    confidence = min(100, abs(score) * 80 + 50)
    
    sl_pct = 0.50
    tp1_pct = 2.50
    tp2_pct = 3.75
    tp3_pct = 5.00
    
    if direction == "BUY":
        entry = price
        sl = price * (1 - sl_pct / 100)
        tp1 = price * (1 + tp1_pct / 100)
        tp2 = price * (1 + tp2_pct / 100)
        tp3 = price * (1 + tp3_pct / 100)
    elif direction == "SELL":
        entry = price
        sl = price * (1 + sl_pct / 100)
        tp1 = price * (1 - tp1_pct / 100)
        tp2 = price * (1 - tp2_pct / 100)
        tp3 = price * (1 - tp3_pct / 100)
    else:
        entry = sl = tp1 = tp2 = tp3 = price
    
    return {
        "direction": direction, "confidence": round(confidence, 1),
        "entry": round(entry, 2), "sl": round(sl, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "sl_pct": sl_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct, "tp3_pct": tp3_pct,
        "evidence": evidence, "strategy": "Position1d", "timeframe": "1d",
    }
