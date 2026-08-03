"""
HQIP v2 — Market Hunter Engine
================================
Multi-timeframe signals with:
- Per-TF strategies optimized for R:R 1:2 to 1:4
- Trading session awareness (Asia/Europe/America)
- Full fundamental + technical analysis
- 6 trades/day max, catch $500+ moves on BTC
"""
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Trading Sessions (UTC) ────────────────────────────────
SESSIONS = {
    "asia":     {"start": 0,  "end": 8,   "name": "آسیا 🇯🇵", "volatility": "low"},
    "europe":   {"start": 7,  "end": 16,  "name": "اروپا 🇪🇺", "volatility": "medium"},
    "america":  {"start": 13, "end": 22,  "name": "آمریکا 🇺🇸", "volatility": "high"},
    "overlap":  {"start": 13, "end": 16,  "name": "همپوشانی 🇪🇺🇺🇸", "volatility": "very_high"},
}

def get_current_session():
    """Which trading session are we in?"""
    utc_hour = datetime.now(timezone.utc).hour
    
    sessions_active = []
    for key, s in SESSIONS.items():
        if key == "overlap":
            if s["start"] <= utc_hour < s["end"]:
                sessions_active.append(s["name"])
        else:
            if s["start"] <= utc_hour < s["end"]:
                sessions_active.append(s["name"])
    
    if not sessions_active:
        return "خارج از ساعات 🌙", "low"
    
    # Check overlap first (highest volatility)
    if "همپوشانی 🇪🇺🇺🇸" in sessions_active:
        return "همپوشانی اروپا/آمریکا 🇪🇺🇺🇸", "very_high"
    
    return " + ".join(sessions_active), "medium" if len(sessions_active) > 1 else "low"


def get_session_adjustment(session_name, volatility):
    """Adjust confidence based on session."""
    if "very_high" in volatility:
        return 1.10, "بهترین زمان — نقدینگی بالا"
    elif "high" in volatility:
        return 1.05, "سشن آمریکا — نقدینگی خوب"
    elif "medium" in volatility:
        return 1.00, "سشن اروپا — نقدینگی متوسط"
    else:
        return 0.85, "سشن آسیا — نقدینگی پایین (احتیاط)"


# ── Multi-Timeframe Analyzer ───────────────────────────────
def analyze_all_timeframes(df_15m, df_1h=None, df_4h=None, df_1d=None, symbol="BTCUSDT"):
    """
    Run analysis on ALL timeframes and return independent signals.
    Each TF has its own optimized strategy.
    """
    results = {}
    
    # ── 15m SCALPING ──
    results["15m"] = _analyze_15m(df_15m, symbol)
    
    # ── 1h DAY TRADING ──
    if df_1h is not None and len(df_1h) >= 30:
        results["1h"] = _analyze_1h(df_1h, symbol)
    
    # ── 4h SWING ──
    if df_4h is not None and len(df_4h) >= 30:
        results["4h"] = _analyze_4h(df_4h, symbol)
    
    # ── 1d POSITION ──
    if df_1d is not None and len(df_1d) >= 30:
        results["1d"] = _analyze_1d(df_1d, symbol)
    
    return results


def _analyze_15m(df, symbol):
    """15m Scalping — Catch $200-800 moves. R:R=1:2"""
    if df is None or df.empty or len(df) < 30:
        return _no_trade("داده کافی نیست")
    
    ev = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # RSI
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 30:
            score += 0.35; ev.append(f"🟢 RSI={rsi:.0f} اشباع فروش")
        elif rsi > 70:
            score -= 0.35; ev.append(f"🔴 RSI={rsi:.0f} اشباع خرید")
        elif rsi < 40:
            score += 0.15; ev.append(f"🟡 RSI={rsi:.0f} نزدیک کف")
        elif rsi > 60:
            score -= 0.15; ev.append(f"🟡 RSI={rsi:.0f} نزدیک سقف")
    
    # Bollinger
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        bb_l = float(df.iloc[-1]["bb_lower"])
        bb_u = float(df.iloc[-1]["bb_upper"])
        if price <= bb_l * 1.003:
            score += 0.30; ev.append(f"🟢 روی باند پایین — برگشت")
        elif price >= bb_u * 0.997:
            score -= 0.30; ev.append(f"🔴 روی باند بالا — اصلاح")
    
    # EMA 9/21 Crossover
    if 'ema9' in df.columns and 'ema21' in df.columns:
        e9 = float(df.iloc[-1]["ema9"])
        e21 = float(df.iloc[-1]["ema21"])
        e9p = float(df.iloc[-2]["ema9"])
        e21p = float(df.iloc[-2]["ema21"])
        if e9 > e21 and e9p <= e21p:
            score += 0.30; ev.append(f"🟢 کراس صعودی EMA 9/21")
        elif e9 < e21 and e9p >= e21p:
            score -= 0.30; ev.append(f"🔴 کراس نزولی EMA 9/21")
        elif e9 > e21: score += 0.10
        elif e9 < e21: score -= 0.10
    
    # MACD
    if 'macd_hist' in df.columns:
        mh = float(df.iloc[-1]["macd_hist"])
        mhp = float(df.iloc[-2]["macd_hist"])
        if mh > 0 and mhp <= 0:
            score += 0.20; ev.append(f"🟢 MACD مثبت شد")
        elif mh < 0 and mhp >= 0:
            score -= 0.20; ev.append(f"🔴 MACD منفی شد")
    
    # Volume
    if 'volume' in df.columns:
        vol = df["volume"].values
        avg = np.mean(vol[-20:])
        if vol[-1] > avg * 1.5:
            ev.append(f"📊 حجم بالا ({vol[-1]/avg:.1f}x)")
            score *= 1.15
    
    d = "BUY" if score > 0.20 else "SELL" if score < -0.20 else "NO_TRADE"
    c = min(100, abs(score) * 120 + 30)
    
    return _make_signal(d, c, price, 0.010, 0.020, 0.030, 0.040, ev, "Scalping15m", "15m")


def _analyze_1h(df, symbol):
    """1h Day Trading — Catch $500-1500 moves. R:R=1:3"""
    if df is None or df.empty or len(df) < 30:
        return _no_trade("داده کافی نیست")
    
    ev = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # EMA 20/50 Trend
    if 'ema20' in df.columns and 'ema50' in df.columns:
        e20 = float(df.iloc[-1]["ema20"])
        e50 = float(df.iloc[-1]["ema50"])
        if e20 > e50 and price > e20:
            score += 0.35; ev.append(f"🟢 روند صعودی قوی")
        elif e20 < e50 and price < e20:
            score -= 0.35; ev.append(f"🔴 روند نزولی قوی")
        elif e20 > e50:
            score += 0.10; ev.append(f"🟡 صعودی — پولبک؟")
        elif e20 < e50:
            score -= 0.10; ev.append(f"🔴 نزولی — بونس؟")
    
    # MACD
    if 'macd_hist' in df.columns:
        mh = float(df.iloc[-1]["macd_hist"])
        mhp = float(df.iloc[-2]["macd_hist"])
        if mh > 0 and mhp <= 0:
            score += 0.25; ev.append(f"🟢 MACD کراس صعودی")
        elif mh < 0 and mhp >= 0:
            score -= 0.25; ev.append(f"🔴 MACD کراس نزولی")
        elif mh > 0: score += 0.10
        elif mh < 0: score -= 0.10
    
    # ADX
    if 'adx' in df.columns:
        adx = float(df.iloc[-1]["adx"])
        if adx > 25:
            ev.append(f"📊 ADX={adx:.0f} روند قوی"); score *= 1.15
        elif adx < 15:
            ev.append(f"📊 ADX={adx:.0f} بازار رِنج"); score *= 0.7
    
    # RSI
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 35: score += 0.20; ev.append(f"🟢 RSI={rsi:.0f}")
        elif rsi > 65: score -= 0.20; ev.append(f"🔴 RSI={rsi:.0f}")
    
    # VWAP
    if 'vwap' in df.columns:
        vwap = float(df.iloc[-1]["vwap"])
        if price > vwap: score += 0.10; ev.append(f"🟢 بالای VWAP")
        else: score -= 0.10; ev.append(f"🔴 زیر VWAP")
    
    d = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    c = min(100, abs(score) * 100 + 40)
    
    return _make_signal(d, c, price, 0.010, 0.020, 0.030, 0.040, ev, "DayTrading1h", "1h")


def _analyze_4h(df, symbol):
    """4h Swing — Catch $1000-3000 moves. R:R=1:4"""
    if df is None or df.empty or len(df) < 30:
        return _no_trade("داده کافی نیست")
    
    ev = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    # EMA Stack 8/21/50
    emas = {}
    for p in [8, 21, 50]:
        col = f"ema{p}"
        if col in df.columns:
            emas[p] = float(df.iloc[-1][col])
    
    if len(emas) == 3:
        if emas[8] > emas[21] > emas[50]:
            score += 0.35; ev.append(f"🟢 EMA Stack صعودی کامل")
        elif emas[8] < emas[21] < emas[50]:
            score -= 0.35; ev.append(f"🔴 EMA Stack نزولی کامل")
        elif emas[8] > emas[21]: score += 0.10
        elif emas[8] < emas[21]: score -= 0.10
    
    # RSI
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 35: score += 0.20; ev.append(f"🟢 RSI={rsi:.0f}")
        elif rsi > 65: score -= 0.20; ev.append(f"🔴 RSI={rsi:.0f}")
    
    # SuperTrend
    if 'supertrend' in df.columns:
        st = float(df.iloc[-1]["supertrend"])
        if price > st: score += 0.15; ev.append(f"🟢 بالای SuperTrend")
        else: score -= 0.15; ev.append(f"🔴 زیر SuperTrend")
    
    d = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    c = min(100, abs(score) * 100 + 45)
    
    return _make_signal(d, c, price, 0.010, 0.020, 0.030, 0.040, ev, "Swing4h", "4h")


def _analyze_1d(df, symbol):
    """1d Position — Catch $2000+ moves. R:R=1:5"""
    if df is None or df.empty or len(df) < 30:
        return _no_trade("داده کافی نیست")
    
    ev = []
    score = 0
    price = float(df.iloc[-1]["close"])
    
    if 'ema50' in df.columns and 'ema200' in df.columns:
        e50 = float(df.iloc[-1]["ema50"])
        e200 = float(df.iloc[-1]["ema200"])
        if e50 > e200 and price > e50:
            score += 0.40; ev.append(f"🟢 روند بلندمدت صعودی")
        elif e50 < e200 and price < e50:
            score -= 0.40; ev.append(f"🔴 روند بلندمدت نزولی")
    
    if 'rsi' in df.columns:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 40: score += 0.20; ev.append(f"🟢 RSI روزانه={rsi:.0f}")
        elif rsi > 60: score -= 0.20; ev.append(f"🔴 RSI روزانه={rsi:.0f}")
    
    d = "BUY" if score > 0.25 else "SELL" if score < -0.25 else "NO_TRADE"
    c = min(100, abs(score) * 80 + 50)
    
    return _make_signal(d, c, price, 0.010, 0.025, 0.0375, 0.050, ev, "Position1d", "1d")


def _make_signal(direction, confidence, price, sl_pct, tp1_pct, tp2_pct, tp3_pct, evidence, strategy, tf):
    """Create a formatted signal with R:R 1:2+."""
    if direction == "BUY":
        entry = price
        sl = price * (1 - sl_pct)
        tp1 = price * (1 + tp1_pct)
        tp2 = price * (1 + tp2_pct)
        tp3 = price * (1 + tp3_pct)
    elif direction == "SELL":
        entry = price
        sl = price * (1 + sl_pct)
        tp1 = price * (1 - tp1_pct)
        tp2 = price * (1 - tp2_pct)
        tp3 = price * (1 - tp3_pct)
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
        "sl_pct": round(sl_pct * 100, 2),
        "tp1_pct": round(tp1_pct * 100, 2),
        "tp2_pct": round(tp2_pct * 100, 2),
        "tp3_pct": round(tp3_pct * 100, 2),
        "evidence": evidence,
        "strategy": strategy,
        "timeframe": tf,
    }


def _no_trade(reason):
    return {
        "direction": "NO_TRADE", "confidence": 0,
        "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0,
        "sl_pct": 0, "tp1_pct": 0, "tp2_pct": 0, "tp3_pct": 0,
        "evidence": [reason], "strategy": "None", "timeframe": "?",
    }
