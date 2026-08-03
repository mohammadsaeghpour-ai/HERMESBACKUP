#!/usr/bin/env python3
"""
HQIP Absolute Zero Scanner — Learn From Scratch
"""
import sys
sys.path.insert(0, "/data/workspace")
from hqip.absolute_zero import AbsoluteZeroEngine, get_session
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators
from datetime import datetime, timezone

dp = DataPlatform()
engine = AbsoluteZeroEngine()

CAPITAL = 20
MAX_LOSS = 4
LEVERAGE = 20

now_utc = datetime.now(timezone.utc)
session_key, session_label = get_session(now_utc.hour)

print("=" * 55)
print("  🔬 HQIP ABSOLUTE ZERO — یادگیری از صفر")
print(f"  💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ ${MAX_LOSS} ضرر")
print(f"  🕐 سشن: {session_label}")
print("=" * 55)

for symbol in ["BTCUSDT", "ETHUSDT"]:
    print(f"\n{'━' * 55}")
    print(f"  🔍 {symbol}")
    print(f"{'━' * 55}")
    
    observations = {}
    raw_data = {}
    
    for tf in ["15m", "1h", "4h", "1d"]:
        try:
            df = dp.fetch_ohlcv(symbol, tf, limit=300)
            if df is not None and not df.empty and len(df) >= 20:
                df = calculate_all_indicators(df)
                raw_data[tf] = df
                obs = engine.observe(df, tf)
                observations[tf] = obs
        except:
            pass
    
    signals = engine.decide(observations, session_key)
    
    # Cross-TF agreement
    buy_tfs = [tf for tf, s in signals.items() if s["direction"] == "BUY"]
    sell_tfs = [tf for tf, s in signals.items() if s["direction"] == "SELL"]
    
    for tf in ["15m", "1h", "4h", "1d"]:
        if tf not in signals:
            continue
        s = signals[tf]
        d = s["direction"]
        c = s["confidence"]
        
        if d == "NO_TRADE":
            print(f"\n  ⏸️ [{tf}] بدون سیگنال ({c:.0f}%)")
            for r in s["reasons"][:2]:
                print(f"     {r}")
            continue
        
        emoji = "🟢" if d == "BUY" else "🔴"
        d_fa = "خرید" if d == "BUY" else "فروش"
        
        # Position sizing for $4 max loss
        sl_dist = abs(s["sl"] - s["entry"]) / s["entry"] if s["entry"] > 0 else 0
        pos_val = MAX_LOSS / sl_dist if sl_dist > 0.001 else 0
        margin = pos_val / LEVERAGE
        tp1_usd = pos_val * abs(s["tp1"] - s["entry"]) / s["entry"]
        tp2_usd = pos_val * abs(s["tp2"] - s["entry"]) / s["entry"]
        tp3_usd = pos_val * abs(s["tp3"] - s["entry"]) / s["entry"]
        
        print(f"\n  {emoji} [{tf}] {d_fa} | اطمینان {c:.0f}% | {s['structure']}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  📍 ورود:     ${s['entry']:,.2f}")
        print(f"  🛑 استاپ:    ${s['sl']:,.2f} ({s['sl_pct']:.2f}%) → ضرر: -${MAX_LOSS:.2f}")
        print(f"  🎯 تارگت ۱: ${s['tp1']:,.2f} ({s['tp1_pct']:.2f}%) → سود: +${tp1_usd:.2f} | R:R 1:{tp1_usd / MAX_LOSS:.1f}")
        print(f"  🎯 تارگت ۲: ${s['tp2']:,.2f} ({s['tp2_pct']:.2f}%) → سود: +${tp2_usd:.2f} | R:R 1:{tp2_usd / MAX_LOSS:.1f}")
        print(f"  🎯 تارگت ۳: ${s['tp3']:,.2f} ({s['tp3_pct']:.2f}%) → سود: +${tp3_usd:.2f} | R:R 1:{tp3_usd / MAX_LOSS:.1f}")
        print(f"  📐 حجم: ${pos_val:,.0f} | مارجین: ${margin:,.1f}")
        print(f"  📊 شتاب: {s['velocity']:+.3f}% | حجم: {s['vol_intensity']:.1f}x | موقعیت: {s['position_in_range']:.0f}%")
        for r in s["reasons"]:
            print(f"     {r}")
    
    # Summary
    print(f"\n  {'─' * 45}")
    if buy_tfs and not sell_tfs:
        print(f"  ✅ همه TF ها خرید — سیگنال قوی 🟢")
    elif sell_tfs and not buy_tfs:
        print(f"  ✅ همه TF ها فروش — سیگنال قوی 🔴")
    elif buy_tfs and sell_tfs:
        print(f"  ⚠️ تناقض! خرید: {buy_tfs} | فروش: {sell_tfs} — احتیاط!")
    else:
        print(f"  ⏸️ بدون سیگنال — منتظر بمان")

print(f"\n{'═' * 55}")
print(f"  ⚠️ فقط توصیه — نه ربات خودکار")
print(f"{'═' * 55}")
