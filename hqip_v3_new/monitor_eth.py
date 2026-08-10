"""
ETH Trade Monitor — هر ۱۵ دقیقه
"""
import requests
import sys

ENTRY = 1908.0
STOP = 1889.0
T1 = 1927.0
T2 = 1942.0
T3 = 1957.0

resp = requests.get('https://www.okx.com/api/v5/market/ticker', params={'instId': 'ETH-USDT-SWAP'}, timeout=10)
price = float(resp.json()['data'][0]['last'])

pnl = ((price - ENTRY) / ENTRY) * 100
pnl_20x = pnl * 20

stop_dist = ((price - STOP) / price) * 100
t3_dist = ((T3 - price) / price) * 100

hit_t1 = price >= T1
hit_t2 = price >= T2
hit_t3 = price >= T3
hit_stop = price <= STOP

print(f"═══════════════════════════════════════")
print(f"📡 ETH Trade Monitor")
print(f"═══════════════════════════════════════")
print(f"💰 قیمت: ${price:,.2f}")
print(f"📊 سود: {pnl:+.2f}% (20x: {pnl_20x:+.2f}%)")
print(f"─────────────────────────────────────")
print(f"📍 ورود: ${ENTRY:,.2f}")
print(f"🛑 استاپ: ${STOP:,.2f} ({stop_dist:.2f}% فاصله)")
print(f"🎯 تارگت۱: ${T1:,.2f} {'✅' if hit_t1 else '⏳'}")
print(f"🎯 تارگت۲: ${T2:,.2f} {'✅' if hit_t2 else '⏳'}")
print(f"🎯 تارگت۳: ${T3:,.2f} {'✅' if hit_t3 else '⏳'} ({t3_dist:.2f}% باقی)")
print(f"─────────────────────────────────────")

if hit_stop:
    print(f"🛑 STOP LOSS HIT!")
elif hit_t3:
    print(f"🎯🎯🎯 TARGET 3 HIT! +51.6%")
elif hit_t2:
    print(f"🎯🎯 TARGET 2 HIT! +36%")
elif hit_t1:
    print(f"🎯 TARGET 1 HIT! +20%")
elif pnl > 0:
    print(f"✅ سودده — ادامه بده")
else:
    print(f"⚠️ در ضرر — مراقب باش")

print(f"═══════════════════════════════════════")
