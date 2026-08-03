#!/usr/bin/env python3
"""HQIP Quick Scanner — Used by cron job every 15 minutes"""
import sys
sys.path.insert(0, "/data/workspace")

from hqip.agents.manager_agent import ManagerAgent
from hqip.config import SYMBOLS

CAPITAL = 20
LEVERAGE = 30
MAX_LOSS = 5

def scan_all():
    m = ManagerAgent()
    lines = []
    lines.append("🎯 HQIP — Smart Money Signals (15min)")
    lines.append(f"💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ ${MAX_LOSS} max")
    lines.append("─" * 35)

    signals = []

    for sym in SYMBOLS:
        try:
            r = m.scan(sym, capital=CAPITAL, max_loss=MAX_LOSS, leverage=LEVERAGE)
        except Exception as e:
            lines.append(f"❌ {sym}: Error")
            continue

        d = r.get("direction", "?")
        grade = r.get("grade", "?")
        conf = r.get("confidence", 0)
        entry = r.get("entry")
        sl = r.get("sl")
        tp1 = r.get("tp1")
        tp2 = r.get("tp2")
        tp3 = r.get("tp3")
        pos_val = r.get("position_value", 0)

        if d == "NO_TRADE":
            lines.append(f"")
            lines.append(f"⚪ {sym} — NO TRADE ({conf:.0f}%)")
            lines.append(f"  ایجنت‌ها اجماع نکردن — ورود نکن")
            continue

        # Calculate exact $ P&L
        if entry and sl and d in ("BUY", "SELL"):
            if d == "BUY":
                sl_loss = (sl - entry) / entry * pos_val * LEVERAGE
                tp1_p = (tp1 - entry) / entry * pos_val * LEVERAGE
                tp2_p = (tp2 - entry) / entry * pos_val * LEVERAGE
                tp3_p = (tp3 - entry) / entry * pos_val * LEVERAGE
            else:
                sl_loss = (entry - sl) / entry * pos_val * LEVERAGE * -1
                tp1_p = (entry - tp1) / entry * pos_val * LEVERAGE
                tp2_p = (entry - tp2) / entry * pos_val * LEVERAGE
                tp3_p = (entry - tp3) / entry * pos_val * LEVERAGE
        else:
            sl_loss = tp1_p = tp2_p = tp3_p = 0

        emoji = "🟢" if d == "BUY" else "🔴"
        tp1_prob = min(95, conf * 0.85)

        lines.append(f"")
        lines.append(f"{emoji} *{sym}* — {d} — {grade} ({conf:.0f}%)")
        lines.append(f"📍 Entry: ${entry:,.2f}" if entry else "")
        lines.append(f"🛑 SL: ${sl:,.2f} → ضرر: ${abs(sl_loss):.2f}" if sl else "")
        lines.append(f"🎯 TP1: ${tp1:,.2f} → سود: +${tp1_p:.2f} (~{tp1_prob:.0f}%)" if tp1 else "")
        lines.append(f"🎯 TP2: ${tp2:,.2f} → سود: +${tp2_p:.2f}" if tp2 else "")
        lines.append(f"🎯 TP3: ${tp3:,.2f} → سود: +${tp3_p:.2f}" if tp3 else "")

        # Top 3 reasons
        reasons = r.get("explanation", [])
        smart_money_reasons = [str(l) for l in reasons[1:4] if any(k in str(l) for k in ["Whale", "SMC", "Liquidity", "Wyckoff", "SmartAction", "Supply", "STOP HUNT", "SWEEP", "SPRING", "UPTRAP", "ABS"])]
        if not smart_money_reasons:
            smart_money_reasons = [str(l) for l in reasons[1:4]]

        for reason in smart_money_reasons[:3]:
            lines.append(f"  {reason[:70]}")

    lines.append("")
    lines.append("─" * 35)
    lines.append("⚠️ فقط توصیه — نه ربات خودکار")

    return "\n".join(lines)


if __name__ == "__main__":
    result = scan_all()
    print(result)
