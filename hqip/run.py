"""
HQIP CLI Runner - Interactive Mode
===================================
Asks user for capital, leverage, max loss, then shows results in table.
"""
import sys, json, os
from hqip.orchestrator import Orchestrator
from hqip.config import SYMBOLS

def clear():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🧠 HQIP — Trading Intelligence Platform            ║
║     Multi-Agent Analysis | 16 Expert Agents | Real-Time     ║
╚══════════════════════════════════════════════════════════════╝""")

def ask_config():
    print("\n📋 لطفاً اطلاعات حساب خود را وارد کنید:")
    print("─" * 50)

    try:
        capital = input("  💰 سرمایه (USD) [10000]: ").strip()
        capital = float(capital) if capital else 10000

        max_loss = input("  🛡️ حداکثر ضرر هر معامله (USD) [100]: ").strip()
        max_loss = float(max_loss) if max_loss else 100

        leverage = input("  ⚡ اهرم [5]: ").strip()
        leverage = int(leverage) if leverage else 5

        symbols_input = input(f"  🪙 ارزها (با کاما جدا کنید) [{','.join(SYMBOLS)}]: ").strip()
        symbols = [s.strip().upper() for s in symbols_input.split(",")] if symbols_input else SYMBOLS

        print(f"\n  ✅ سرمایه: ${capital:,.0f} | اهرم: {leverage}x | حداکثر ضرر: ${max_loss:.0f}/معامله")
        print(f"  🪙 ارزها: {', '.join(symbols)}")
        print("─" * 50)

        return capital, max_loss, leverage, symbols
    except ValueError:
        print("  ⚠️ مقدار نامعتبر، از مقادیر پیش‌فرض استفاده می‌شود")
        return 10000, 100, 5, SYMBOLS

def print_signal_table(results):
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                              📊 نتایج تحلیل — SIGNAL TABLE                                ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")

    for r in results:
        if r.get("direction") == "ERROR":
            print(f"║  ❌ {r['symbol']}: Error                                                                      ║")
            continue

        d = r["direction"]
        symbol = r["symbol"]
        grade = r.get("grade", "?")
        conf = r.get("confidence", 0)
        entry = r.get("entry")
        sl = r.get("sl")
        tp1 = r.get("tp1")
        tp2 = r.get("tp2")
        tp3 = r.get("tp3")
        pos = r.get("position_size", 0)
        pos_val = r.get("position_value", 0)
        rr1 = r.get("risk_reward", 0)

        emoji = "🟢" if d == "BUY" else "🔴" if d == "SELL" else "⚪"
        grade_emoji = {"A+": "🏆", "A": "🥇", "B+": "🥈", "B": "🥉", "C": "📋"}.get(grade, "📋")

        # Calculate hit probabilities (simple estimate based on RR and confidence)
        tp1_prob = min(95, conf * 0.85) if entry else 0
        tp2_prob = min(75, conf * 0.60) if entry else 0
        tp3_prob = min(50, conf * 0.35) if entry else 0

        if d == "NO_TRADE":
            print(f"║                                                                                              ║")
            print(f"║  ⚪ {symbol:<10} │ ❌ NO TRADE                                                            ║")
            print(f"║     درجه: {grade} │ اطمینان: {conf:.0f}%                                                          ║")
            print(f"║     دلیل: اجماع کافی بین ایجنت‌ها وجود ندارد                                                  ║")
            print(f"║                                                                                              ║")
            print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
            continue

        print(f"║                                                                                              ║")
        print(f"║  {emoji} {symbol:<10} │ {d:<6} │ {grade_emoji} درجه: {grade} │ اطمینان: {conf:.0f}%                           ║")
        print(f"║  {'─'*89} ║")

        if entry:
            sl_dist = abs(entry - sl) / entry * 100 if sl else 0
            tp1_dist = abs(tp1 - entry) / entry * 100 if tp1 else 0
            tp2_dist = abs(tp2 - entry) / entry * 100 if tp2 else 0
            tp3_dist = abs(tp3 - entry) / entry * 100 if tp3 else 0

            print(f"║                                                                                              ║")
            print(f"║  📍 نقطه ورود (Entry):   {entry:<15.2f}  │  حجم پوزیشن: {pos:.6f} (${pos_val:,.0f})             ║")
            print(f"║  🛑 استاپ لاس (SL):      {sl:<15.2f}  │  فاصله SL: {sl_dist:.2f}%                                    ║")
            print(f"║  {'─'*89} ║")
            print(f"║  🎯 تارگت ۱ (TP1):      {tp1:<15.2f}  │  فاصله: {tp1_dist:.2f}%  │  احتمال تارگت: ~{tp1_prob:.0f}%           ║")
            print(f"║  🎯 تارگت ۲ (TP2):      {tp2:<15.2f}  │  فاصله: {tp2_dist:.2f}%  │  احتمال تارگت: ~{tp2_prob:.0f}%           ║")
            print(f"║  🎯 تارگت ۳ (TP3):      {tp3:<15.2f}  │  فاصله: {tp3_dist:.2f}%  │  احتمال تارگت: ~{tp3_prob:.0f}%           ║")
            print(f"║  {'─'*89} ║")
            print(f"║  📐 نسبت ریسک/ریوارد:   1:{rr1:.1f}          │  RR مناسب: {'✅' if rr1 >= 1.5 else '⚠️'}                                       ║")

        # Top reasons
        reasons = r.get("explanation", [])
        if reasons:
            print(f"║  {'─'*89} ║")
            print(f"║  📋 دلایل اصلی:                                                                              ║")
            for line in reasons[1:6]:  # Skip header
                line = line[:85]
                print(f"║     {line:<86}║")

        print(f"║                                                                                              ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")

    print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝")

def print_summary_table(results, capital, leverage, max_loss):
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                              📋 خلاصه — SUMMARY TABLE                                      ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  💰 سرمایه: ${capital:,.0f}  │  ⚡ اهرم: {leverage}x  │  🛡️ حداکثر ضرر: ${max_loss:.0f}/معامله        ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'ارز':<12} │ {'سیگنال':<8} │ {'درجه':<6} │ {'اطمینان':<8} │ {'ورود':<14} │ {'SL':<14} │ {'TP1':<14}  ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════╣")

    for r in results:
        d = r.get("direction", "?")
        symbol = r.get("symbol", "?")
        grade = r.get("grade", "?")
        conf = r.get("confidence", 0)
        entry = r.get("entry")
        sl = r.get("sl")
        tp1 = r.get("tp1")

        emoji = "🟢" if d == "BUY" else "🔴" if d == "SELL" else "⚪"

        entry_s = f"{entry:.2f}" if entry else "—"
        sl_s = f"{sl:.2f}" if sl else "—"
        tp1_s = f"{tp1:.2f}" if tp1 else "—"

        print(f"║  {symbol:<12} │ {emoji} {d:<5} │ {grade:<6} │ {conf:>5.0f}%  │ {entry_s:<14} │ {sl_s:<14} │ {tp1_s:<14}  ║")

    print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝")

    # Warnings
    buy_signals = [r for r in results if r.get("direction") == "BUY"]
    sell_signals = [r for r in results if r.get("direction") == "SELL"]

    if buy_signals:
        total_buy_risk = sum(r.get("position_value", 0) for r in buy_signals) / leverage
        print(f"\n  ⚠️ ریسک کل خرید: ${total_buy_risk:,.0f} ({total_buy_risk/capital*100:.1f}% سرمایه)")
    if sell_signals:
        total_sell_risk = sum(r.get("position_value", 0) for r in sell_signals) / leverage
        print(f"  ⚠️ ریسک کل فروش: ${total_sell_risk:,.0f} ({total_sell_risk/capital*100:.1f}% سرمایه)")

    print(f"\n  ⚠️ هشدار: این سیستم فقط توصیه معاملاتی است، نه ربات خودکار!")
    print(f"  ⚠️ همیشه مدیریت سرمایه را رعایت کنید.\n")

def main():
    print_header()
    capital, max_loss, leverage, symbols = ask_config()

    print(f"\n⏳ در حال تحلیل {len(symbols)} ارز...")
    print("─" * 50)

    orch = Orchestrator(capital=capital, max_loss=max_loss, leverage=leverage)
    results = []

    for symbol in symbols:
        try:
            r = orch.scan_symbol(symbol)
            results.append(r)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            results.append({"symbol": symbol, "direction": "ERROR", "error": str(e)})

    # Print results
    print_signal_table(results)
    print_summary_table(results, capital, leverage, max_loss)

if __name__ == "__main__":
    main()
