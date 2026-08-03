#!/usr/bin/env python3
"""
HQIP v2 — Full Market Hunter Scanner
Multi-timeframe + Sessions + Fundamental + Technical + Interpretation
"""
import sys
sys.path.insert(0, "/data/workspace")

from hqip.hunter_engine import analyze_all_timeframes, get_current_session, get_session_adjustment
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
CAPITAL = 10.0
LEVERAGE = 20
MAX_TRADES = 6

def scan_all():
    dp = DataPlatform()
    session_name, session_vol = get_current_session()
    sess_adj, sess_note = get_session_adjustment(session_name, session_vol)
    
    lines = []
    lines.append("=" * 50)
    lines.append("  🎯 HQIP — شکارچی بازار v2")
    lines.append(f"  💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ $1 ضرر | 🎯 $2-4 سود")
    lines.append(f"  📊 حداکثر {MAX_TRADES} معامله در روز")
    lines.append("=" * 50)
    lines.append(f"  🕐 سشن فعلی: {session_name}")
    lines.append(f"  📈 {sess_note}")
    lines.append(f"  ⚙️ ضریب سشن: x{sess_adj:.2f}")
    lines.append("=" * 50)
    
    for symbol in SYMBOLS:
        lines.append(f"\n{'━' * 50}")
        lines.append(f"  🔍 {symbol}")
        lines.append(f"{'━' * 50}")
        
        # Fetch all timeframes
        data = {}
        for tf in ["15m", "1h", "4h", "1d"]:
            try:
                df = dp.fetch_ohlcv(symbol, tf, limit=300)
                if df is not None and not df.empty and len(df) >= 30:
                    df = calculate_all_indicators(df)
                    data[tf] = df
            except:
                pass
        
        # Run analysis
        results = analyze_all_timeframes(
            data.get("15m"), data.get("1h"), data.get("4h"), data.get("1d"), symbol
        )
        
        # Format each timeframe
        for tf in ["15m", "1h", "4h", "1d"]:
            if tf not in results:
                continue
            r = results[tf]
            d = r["direction"]
            c = r["confidence"]
            
            if d == "NO_TRADE":
                lines.append(f"\n  ⏸️ [{tf}] — بدون سیگنال ({c:.0f}%)")
                for e in r["evidence"][:2]:
                    lines.append(f"     {e}")
                continue
            
            # Apply session adjustment
            adj_conf = min(100, c * sess_adj)
            
            emoji = "🟢" if d == "BUY" else "🔴"
            d_fa = "خرید" if d == "BUY" else "فروش"
            
            lines.append(f"\n  {emoji} [{tf}] {d_fa} — {r['strategy']}")
            lines.append(f"  {'─' * 45}")
            lines.append(f"  📍 نقطه ورود:  ${r['entry']:,.2f}")
            lines.append(f"  🛑 استاپ‌لاس:  ${r['sl']:,.2f} ({r['sl_pct']:.1f}%)")
            lines.append(f"  🎯 تارگت ۱:   ${r['tp1']:,.2f} ({r['tp1_pct']:.1f}%) → سود ~$2")
            lines.append(f"  🎯 تارگت ۲:   ${r['tp2']:,.2f} ({r['tp2_pct']:.1f}%) → سود ~$3")
            lines.append(f"  🎯 تارگت ۳:   ${r['tp3']:,.2f} ({r['tp3_pct']:.1f}%) → سود ~$4")
            lines.append(f"  📊 اطمینان:    {c:.0f}% → {adj_conf:.0f}% (پس از اصلاح سشن)")
            
            # Evidence
            lines.append(f"  📝 دلایل:")
            for e in r["evidence"][:4]:
                lines.append(f"     {e}")
        
        # Cross-TF analysis
        buy_tfs = [tf for tf, r in results.items() if r["direction"] == "BUY"]
        sell_tfs = [tf for tf, r in results.items() if r["direction"] == "SELL"]
        
        lines.append(f"\n  {'─' * 45}")
        lines.append(f"  📊 خلاصه تایم‌فریم‌ها:")
        lines.append(f"     🟢 خرید: {', '.join(buy_tfs) if buy_tfs else '—'}")
        lines.append(f"     🔴 فروش: {', '.join(sell_tfs) if sell_tfs else '—'}")
        
        if buy_tfs and not sell_tfs:
            lines.append(f"     ✅ همه تایم‌فریم‌ها هم‌نظر — سیگنال قوی")
        elif sell_tfs and not buy_tfs:
            lines.append(f"     ✅ همه تایم‌فریم‌ها هم‌نظر — سیگنال قوی")
        elif buy_tfs and sell_tfs:
            lines.append(f"     ⚠️ تناقض بین تایم‌فریم‌ها — احتیاط!")
    
    lines.append(f"\n{'═' * 50}")
    lines.append(f"  ⚠️ فقط توصیه — نه ربات خودکار")
    lines.append(f"{'═' * 50}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(scan_all())
