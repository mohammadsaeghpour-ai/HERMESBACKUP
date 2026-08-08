"""
HERMES Signal Monitor — Stop Reminder + DAG Pipeline
نظارت زنده + یادآور استاپ + گزارش خودکار
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
        'id': 1, 'symbol': 'BTC-USDT-SWAP', 'name': 'BTC',
        'direction': 'BUY', 'entry': 64970.80,
        'stop': 64536.00, 'target1': 65255.00, 'target2': 65540.00, 'target3': 65824.00,
        'time': '12:45', 'date': '2026-08-08', 'status': 'ACTIVE'
    },
    {
        'id': 2, 'symbol': 'ETH-USDT-SWAP', 'name': 'ETH',
        'direction': 'BUY', 'entry': 1916.10,
        'stop': 1901.00, 'target1': 1927.00, 'target2': 1938.00, 'target3': 1949.00,
        'time': '12:45', 'date': '2026-08-08', 'status': 'ACTIVE'
    },
    {
        'id': 3, 'symbol': 'SOL-USDT-SWAP', 'name': 'SOL',
        'direction': 'BUY', 'entry': 74.80,
        'stop': 74.32, 'target1': 75.28, 'target2': 75.76, 'target3': 76.24,
        'time': '12:45', 'date': '2026-08-08', 'status': 'ACTIVE'
    }
]

SIGNAL_FILE = os.path.join(os.path.dirname(__file__), 'active_signals.json')
MONITOR_LOG = os.path.join(os.path.dirname(__file__), 'monitor_log.json')


def get_price(symbol):
    try:
        resp = requests.get('https://www.okx.com/api/v5/market/ticker', params={'instId': symbol}, timeout=10)
        return float(resp.json()['data'][0]['last'])
    except:
        return None


def check_signal(sig):
    price = get_price(sig['symbol'])
    if not price:
        return None
    
    entry = sig['entry']
    stop = sig['stop']
    t1, t2, t3 = sig['target1'], sig['target2'], sig['target3']
    
    if sig['direction'] == 'BUY':
        pnl_pct = ((price - entry) / entry) * 100
        pnl_20x = pnl_pct * 20
        hit_stop = price <= stop
        hit_t1, hit_t2, hit_t3 = price >= t1, price >= t2, price >= t3
    else:
        pnl_pct = ((entry - price) / entry) * 100
        pnl_20x = pnl_pct * 20
        hit_stop = price >= stop
        hit_t1, hit_t2, hit_t3 = price <= t1, price <= t2, price <= t3
    
    # فاصله تا استاپ
    stop_distance = abs(price - stop) / price * 100
    
    # وضعیت
    if hit_stop:
        status, alert = 'STOP_HIT', '🛑 STOP LOSS HIT!'
    elif hit_t3:
        status, alert = 'TARGET3', '🎯🎯🎯 TARGET 3!'
    elif hit_t2:
        status, alert = 'TARGET2', '🎯🎯 TARGET 2!'
    elif hit_t1:
        status, alert = 'TARGET1', '🎯 TARGET 1!'
    elif pnl_pct > 0:
        status, alert = 'PROFIT', f'✅ +{pnl_pct:.2f}%'
    else:
        status, alert = 'LOSS', f'❌ {pnl_pct:.2f}%'
    
    # هشدار استاپ
    stop_warning = ''
    if stop_distance < 0.3:
        stop_warning = f'⚠️ فاصله تا استاپ فقط {stop_distance:.2f}%!'
    elif stop_distance < 0.5:
        stop_warning = f'⚡ فاصله تا استاپ: {stop_distance:.2f}%'
    
    return {
        'signal': sig,
        'price': price,
        'pnl_pct': pnl_pct,
        'pnl_20x': pnl_20x,
        'hit_stop': hit_stop,
        'hit_t1': hit_t1, 'hit_t2': hit_t2, 'hit_t3': hit_t3,
        'status': status,
        'alert': alert,
        'stop_distance': stop_distance,
        'stop_warning': stop_warning
    }


def format_output(results):
    tehran_tz = timezone(timedelta(hours=3, minutes=30))
    now = datetime.now(tehran_tz)
    
    out = f"""
{'═'*50}
📡 HERMES Signal Monitor
{'═'*50}
📅 {now.strftime('%Y-%m-%d %H:%M')} Tehran
{'─'*50}"""
    
    for r in results:
        s = r['signal']
        emoji = "🟢" if s['direction'] == 'BUY' else "🔴"
        
        out += f"""
{emoji} {s['name']} — {s['direction']} @ {s['time']}
  ورود: ${s['entry']:,.2f}
  الان: ${r['price']:,.2f}
  سود: {r['pnl_pct']:+.2f}% (20x: {r['pnl_20x']:+.2f}%)
  ─────────────────────
  🛑 استاپ: ${s['stop']:,.2f} ({r['stop_distance']:.2f}% فاصله)
  {'🚨 نزدیک استاپ!' if r['stop_warning'] else '✅ امن'}
  🎯 تارگت۱: ${s['target1']:,.2f} {'✅' if r['hit_t1'] else '⏳'}
  🎯 تارگت۲: ${s['target2']:,.2f} {'✅' if r['hit_t2'] else '⏳'}
  🎯 تارگت۳: ${s['target3']:,.2f} {'✅' if r['hit_t3'] else '⏳'}
  📊 وضعیت: {r['alert']}"""
        
        if r['stop_warning']:
            out += f"\n  {r['stop_warning']}"
    
    # خلاصه
    total_pnl = sum(r['pnl_20x'] for r in results)
    wins = sum(1 for r in results if r['pnl_pct'] > 0)
    losses = sum(1 for r in results if r['pnl_pct'] <= 0)
    stops = sum(1 for r in results if r['hit_stop'])
    
    out += f"""
{'─'*50}
📊 خلاصه:
  ✅ برد: {wins} | ❌ باخت: {losses} | 🛑 استاپ: {stops}
  💰 کل سود (20x): {total_pnl:+.2f}%
{'═'*50}"""
    
    return out


def run_monitor():
    results = []
    for sig in ACTIVE_SIGNALS:
        if sig['status'] == 'ACTIVE':
            r = check_signal(sig)
            if r:
                results.append(r)
    
    if results:
        print(format_output(results))
        
        # لاگ
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'results': [{
                'name': r['signal']['name'],
                'price': r['price'],
                'pnl_pct': r['pnl_pct'],
                'status': r['status'],
                'stop_distance': r['stop_distance']
            } for r in results]
        }
        
        try:
            logs = []
            if os.path.exists(MONITOR_LOG):
                with open(MONITOR_LOG, 'r') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            logs = logs[-100:]  # نگه‌داشتن ۱۰۰ لاگ آخر
            with open(MONITOR_LOG, 'w') as f:
                json.dump(logs, f, indent=2)
        except:
            pass
    
    return results


if __name__ == '__main__':
    run_monitor()
