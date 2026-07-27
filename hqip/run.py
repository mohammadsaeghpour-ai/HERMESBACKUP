"""
HQIP CLI Runner — Interactive Hunter Mode
==========================================
Asks ALL questions upfront, shows exact $ profit/loss per target/SL.
"""
import sys
from hqip.agents.manager_agent import ManagerAgent
from hqip.config import SYMBOLS

def ask_all():
    print("""
╔══════════════════════════════════════════════════════════╗
║         🧠 HQIP — Smart Money Signal Hunter             ║
║         21 Expert Agents | Real-Time Analysis           ║
╚══════════════════════════════════════════════════════════╝
""")
    print("📋 لطفاً اطلاعات زیر را وارد کنید:")
    print("─" * 50)

    try:
        capital = input("  💰 سرمایه (USD) [10]: ").strip()
        capital = float(capital) if capital else 10.0

        leverage = input("  ⚡ اهرم [20]: ").strip()
        leverage = int(leverage) if leverage else 20

        max_loss = input(f"  🛡️ حداکثر ضرر هر معامله (USD) [{capital*0.5:.0f}]: ").strip()
        max_loss = float(max_loss) if max_loss else capital * 0.5

        symbols_input = input(f"  🪙 ارزها (با کاما) [{','.join(SYMBOLS)}]: ").strip()
        symbols = [s.strip().upper() for s in symbols_input.split(",")] if symbols_input else SYMBOLS

        print(f"\n  ✅ سرمایه: ${capital:.2f}")
        print(f"  ⚡ اهرم: {leverage}x → قدرت خرید: ${capital * leverage:.2f}")
        print(f"  🛡️ حداکثر ضرر: ${max_loss:.2f}/معامله")
        print(f"  🪙 ارزها: {', '.join(symbols)}")
        print("─" * 50)

        return capital, max_loss, leverage, symbols
    except ValueError:
        return 10.0, 5.0, 20, SYMBOLS


def print_signal(r, capital, leverage, max_loss):
    d = r['direction']
    grade = r.get('grade', '?')
    conf = r.get('confidence', 0)
    entry = r.get('entry') or 0
    sl = r.get('sl') or 0
    tp1 = r.get('tp1') or 0
    tp2 = r.get('tp2') or 0
    tp3 = r.get('tp3') or 0
    pos = r.get('position_size') or 0
    pos_val = r.get('position_value') or 0

    # Calculate exact $ P&L at each level (with leverage)
    if entry and sl and d in ("BUY", "SELL"):
        if d == "BUY":
            sl_loss = (sl - entry) / entry * pos_val * leverage
            tp1_profit = (tp1 - entry) / entry * pos_val * leverage
            tp2_profit = (tp2 - entry) / entry * pos_val * leverage
            tp3_profit = (tp3 - entry) / entry * pos_val * leverage
        else:  # SELL
            sl_loss = (entry - sl) / entry * pos_val * leverage * -1
            tp1_profit = (entry - tp1) / entry * pos_val * leverage
            tp2_profit = (entry - tp2) / entry * pos_val * leverage
            tp3_profit = (entry - tp3) / entry * pos_val * leverage

        tp1_prob = min(95, conf * 0.85)
        tp2_prob = min(75, conf * 0.60)
        tp3_prob = min(50, conf * 0.35)
    else:
        sl_loss = tp1_profit = tp2_profit = tp3_profit = 0
        tp1_prob = tp2_prob = tp3_prob = 0

    emoji = '🟢' if d == 'BUY' else '🔴' if d == 'SELL' else '⚪'
    grade_e = {'A+':'🏆','A':'🥇','B+':'🥈','B':'🥉','C':'📋'}.get(grade, '📋')

    print()
    print('╔═══════════════════════════════════════════════════════════════════════════════╗')
    print('║                       🎯 HQIP SIGNAL — SMART MONEY HUNTER                   ║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')
    print(f'║  💰 ${capital:.2f} | ⚡ {leverage}x | 🛡️ ${max_loss:.2f} max loss | 🤖 21 agents       ║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')

    if d == 'NO_TRADE':
        print(f'║                                                                               ║')
        print(f'║  ⚪ {r["symbol"]} — ❌ NO TRADE                                              ║')
        print(f'║  Confidence: {conf:.0f}% — ایجنت‌ها اجماع نکردن                              ║')
        print(f'║  ✅ این تصمیم هوشمندانه‌ایه — ورود نکردن بهتر از ورود بده                   ║')
    else:
        print(f'║                                                                               ║')
        print(f'║  {emoji} {r["symbol"]} — {d} — {grade_e} درجه: {grade} — اطمینان: {conf:.0f}%          ║')
        print(f'║  ─────────────────────────────────────────────────────────────────────────── ║')
        print(f'║                                                                               ║')
        print(f'║  📍 نقطه ورود:  ${entry:<14,.2f}  حجم: ${pos_val:.2f} ({pos:.6f})          ║')
        print(f'║                                                                               ║')
        print(f'║  🛑 استاپ لاس:  ${sl:<14,.2f}  💀 ضرر: ${sl_loss:.2f}                        ║')
        print(f'║                     فاصله: {abs(entry-sl)/entry*100:.2f}%                                        ║')
        print(f'║  ─────────────────────────────────────────────────────────────────────────── ║')
        print(f'║                                                                               ║')
        print(f'║  🎯 تارگت ۱:   ${tp1:<14,.2f}  💚 سود: +${tp1_profit:.2f}                    ║')
        print(f'║                     فاصله: {abs(tp1-entry)/entry*100:.2f}% | احتمال: ~{tp1_prob:.0f}%                 ║')
        print(f'║                                                                               ║')
        print(f'║  🎯 تارگت ۲:   ${tp2:<14,.2f}  💚 سود: +${tp2_profit:.2f}                    ║')
        print(f'║                     فاصله: {abs(tp2-entry)/entry*100:.2f}% | احتمال: ~{tp2_prob:.0f}%                 ║')
        print(f'║                                                                               ║')
        print(f'║  🎯 تارگت ۳:   ${tp3:<14,.2f}  💚 سود: +${tp3_profit:.2f}                    ║')
        print(f'║                     فاصله: {abs(tp3-entry)/entry*100:.2f}% | احتمال: ~{tp3_prob:.0f}%                 ║')
        print(f'║                                                                               ║')
        print(f'║  ─────────────────────────────────────────────────────────────────────────── ║')
        rr = r.get('risk_reward', 0) or 0
        risk_pct = abs(sl_loss) / capital * 100 if capital else 0
        print(f'║  📐 RR: 1:{rr:.1f}  |  ⚠️ ریسک: ${abs(sl_loss):.2f} ({risk_pct:.1f}% سرمایه)           ║')

        # Risk check
        if abs(sl_loss) > max_loss:
            print(f'║  🚨 هشدار: ضرر استاپ (${abs(sl_loss):.2f}) بیشتر از حد مجاز (${max_loss:.2f}) است!     ║')

    print(f'║                                                                               ║')
    print('║  📋 دلایل ایجنت‌ها:                                                           ║')
    print('║  ─────────────────────────────────────────────────────────────────────────── ║')
    reasons = r.get('explanation', [])
    for line in reasons[1:10]:
        line = str(line)[:82]
        print(f'║    {line:<80}║')

    print(f'║                                                                               ║')
    print('╚═══════════════════════════════════════════════════════════════════════════════╝')
    print(f'  ⚠️ این فقط توصیه است — نه ربات خودکار | 21 ایجنت هوش مصنوعی')


def print_summary(results, capital, leverage):
    print()
    print('╔═══════════════════════════════════════════════════════════════════════════════╗')
    print('║                           📋 SUMMARY — ALL SIGNALS                          ║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')
    print(f'║  💰 ${capital:.2f} | ⚡ {leverage}x | 🤖 21 agents per signal                   ║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')
    print(f'║  ارز         │ سیگنال     │ درجه  │ اطمینان │ ورود          │ SL            ║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')

    for r in results:
        d = r.get('direction', '?')
        sym = r.get('symbol', '?')
        grade = r.get('grade', '?')
        conf = r.get('confidence', 0)
        entry = r.get('entry')
        sl = r.get('sl')
        pos_val = r.get('position_value', 0)

        emoji = '🟢' if d == 'BUY' else '🔴' if d == 'SELL' else '⚪'
        entry_s = f'${entry:.2f}' if entry else '—'
        sl_s = f'${sl:.2f}' if sl else '—'

        print(f'║  {sym:<11} │ {emoji} {d:<8} │ {grade:<5} │ {conf:>5.0f}% │ {entry_s:<14} │ {sl_s:<13} ║')

    print('╚═══════════════════════════════════════════════════════════════════════════════╝')
    print()


def main():
    capital, max_loss, leverage, symbols = ask_all()

    m = ManagerAgent()
    results = []

    for sym in symbols:
        try:
            r = m.scan(sym, capital=capital, max_loss=max_loss, leverage=leverage)
            results.append(r)
            print_signal(r, capital, leverage, max_loss)
        except Exception as e:
            print(f"Error {sym}: {e}")
            results.append({"symbol": sym, "direction": "ERROR"})

    print_summary(results, capital, leverage)


if __name__ == "__main__":
    main()
