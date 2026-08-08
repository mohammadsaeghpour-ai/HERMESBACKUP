"""
HERMES Signal Monitor — DAG Pipeline
نظارت زنده روی سیگنال‌ها + یادآور استاپ
"""
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import os

# ═══════════════════════════════════════════════════
# سیگنال‌های فعال
# ═══════════════════════════════════════════════════
ACTIVE_SIGNALS = [
    {
        'id': 1,
        'symbol': 'BTC-USDT-SWAP',
        'name': 'BTC',
        'direction': 'BUY',
        'entry': 64565.30,
        'stop': 64198.00,
        'target1': 64932.00,
        'target2': 65299.00,
        'target3': 65666.00,
        'time': '15:00',
        'date': '2026-08-07',
        'status': 'ACTIVE'
    },
    {
        'id': 2,
        'symbol': 'ETH-USDT-SWAP',
        'name': 'ETH',
        'direction': 'BUY',
        'entry': 1906.92,
        'stop': 1891.91,
        'target1': 1921.93,
        'target2': 1936.94,
        'target3': 1951.95,
        'time': '15:00',
        'date': '2026-08-07',
        'status': 'ACTIVE'
    },
    {
        'id': 3,
        'symbol': 'SOL-USDT-SWAP',
        'name': 'SOL',
        'direction': 'BUY',
        'entry': 73.24,
        'stop': 72.78,
        'target1': 73.70,
        'target2': 74.16,
        'target3': 74.62,
        'time': '15:00',
        'date': '2026-08-07',
        'status': 'ACTIVE'
    }
]


def get_current_price(symbol):
    """دریافت قیمت لایو"""
    try:
        resp = requests.get(
            'https://www.okx.com/api/v5/market/ticker',
            params={'instId': symbol}, timeout=10
        )
        return float(resp.json()['data'][0]['last'])
    except:
        return None


def check_signal(signal):
    """بررسی وضعیت سیگنال"""
    price = get_current_price(signal['symbol'])
    if not price:
        return None
    
    entry = signal['entry']
    stop = signal['stop']
    t1 = signal['target1']
    t2 = signal['target2']
    t3 = signal['target3']
    
    if signal['direction'] == 'BUY':
        pnl_pct = ((price - entry) / entry) * 100
        pnl_20x = pnl_pct * 20
        
        hit_stop = price <= stop
        hit_t1 = price >= t1
        hit_t2 = price >= t2
        hit_t3 = price >= t3
    else:
        pnl_pct = ((entry - price) / entry) * 100
        pnl_20x = pnl_pct * 20
        
        hit_stop = price >= stop
        hit_t1 = price <= t1
        hit_t2 = price <= t2
        hit_t3 = price <= t3
    
    # تعیین وضعیت
    if hit_stop:
        status = 'STOP_HIT'
        alert = '🛑 STOP LOSS ZADE SHOD!'
    elif hit_t3:
        status = 'TARGET3_HIT'
        alert = '🎯🎯🎯 TARGET 3 REACHED!'
    elif hit_t2:
        status = 'TARGET2_HIT'
        alert = '🎯🎯 TARGET 2 REACHED!'
    elif hit_t1:
        status = 'TARGET1_HIT'
        alert = '🎯 TARGET 1 REACHED!'
    elif pnl_pct > 0:
        status = 'PROFIT'
        alert = f'✅ SOD: {pnl_pct:+.2f}%'
    else:
        status = 'LOSS'
        alert = f'❌ ZER: {pnl_pct:+.2f}%'
    
    return {
        'signal': signal,
        'current_price': price,
        'pnl_pct': pnl_pct,
        'pnl_20x': pnl_20x,
        'hit_stop': hit_stop,
        'hit_t1': hit_t1,
        'hit_t2': hit_t2,
        'hit_t3': hit_t3,
        'status': status,
        'alert': alert
    }


def format_monitor(results):
    """فرمت نظارت"""
    tehran_tz = timezone(timedelta(hours=3, minutes=30))
    now_tehran = datetime.now(tehran_tz)
    
    output = f"""
{'═'*50}
📡 HERMES Signal Monitor — DAG
{'═'*50}
📅 {now_tehran.strftime('%Y-%m-%d %H:%M')} Tehran Time
{'─'*50}"""
    
    for r in results:
        s = r['signal']
        emoji = "🟢" if s['direction'] == 'BUY' else "🔴"
        status_emoji = "✅" if r['pnl_pct'] > 0 else "❌"
        
        output += f"""
{emoji} {s['name']} — {s['direction']}
  ورود: ${s['entry']:,.2f}
  الان: ${r['current_price']:,.2f}
  سود: {r['pnl_pct']:+.2f}% (20x: {r['pnl_20x']:+.2f}%)
  استاپ: ${s['stop']:,.2f} {'🛑 ZADE SHOD' if r['hit_stop'] else '✅ AMEN'}
  تارگت۱: ${s['target1']:,.2f} {'✅' if r['hit_t1'] else '⏳'}
  تارگت۲: ${s['target2']:,.2f} {'✅' if r['hit_t2'] else '⏳'}
  تارگت۳: ${s['target3']:,.2f} {'✅' if r['hit_t3'] else '⏳'}
  وضعیت: {r['alert']}"""
    
    # خلاصه
    total_pnl = sum(r['pnl_20x'] for r in results)
    wins = sum(1 for r in results if r['pnl_pct'] > 0)
    losses = sum(1 for r in results if r['pnl_pct'] <= 0)
    
    output += f"""
{'─'*50}
📊 خلاصه:
  ✅ برد: {wins} | ❌ باخت: {losses}
  💰 کل سود (20x): {total_pnl:+.2f}%
{'═'*50}"""
    
    return output


def run_monitor():
    """اجرای نظارت"""
    results = []
    
    for signal in ACTIVE_SIGNALS:
        if signal['status'] == 'ACTIVE':
            result = check_signal(signal)
            if result:
                results.append(result)
    
    if results:
        print(format_monitor(results))
        
        # بررسی هشدارها
        for r in results:
            if r['hit_stop']:
                print(f"\n🛑⚠️⚠️⚠️ STOP LOSS HIT — {r['signal']['name']} ⚠️⚠️⚠️")
            elif r['hit_t3']:
                print(f"\n🎯🎉🎉 TARGET 3 HIT — {r['signal']['name']} 🎉🎉")
            elif r['hit_t2']:
                print(f"\n🎯🎯 TARGET 2 HIT — {r['signal']['name']}")
            elif r['hit_t1']:
                print(f"\n🎯 TARGET 1 HIT — {r['signal']['name']}")
    
    return results


if __name__ == '__main__':
    run_monitor()
