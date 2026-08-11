"""
HERMES Market Monitor — بهترین فرصت معاملاتی
هر ۳۰ دقیقه بازار رو چک می‌کنه
"""
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

tehran_tz = timezone(timedelta(hours=3, minutes=30))

def get_data(symbol, tf='1H', limit=50):
    resp = requests.get('https://www.okx.com/api/v5/market/candles', params={'instId': symbol, 'bar': tf, 'limit': str(limit)}, timeout=10)
    candles = resp.json()['data']
    return {
        'close': np.array([float(c[4]) for c in candles]),
        'high': np.array([float(c[2]) for c in candles]),
        'low': np.array([float(c[3]) for c in candles]),
        'volume': np.array([float(c[5]) for c in candles]),
    }

def get_price(symbol):
    resp = requests.get('https://www.okx.com/api/v5/market/ticker', params={'instId': symbol}, timeout=10)
    return float(resp.json()['data'][0]['last'])

def analyze(symbol):
    data = get_data(symbol)
    close = data['close']
    volume = data['volume']
    price = get_price(symbol)
    
    # RSI
    deltas = np.diff(close)
    gains = np.mean(deltas[deltas > 0][-14:]) if np.any(deltas > 0) else 0
    losses = -np.mean(deltas[deltas < 0][-14:]) if np.any(deltas < 0) else 1e-8
    rsi = 100 - (100 / (1 + gains / max(losses, 1e-8)))
    
    # MACD
    ema12 = np.mean(close[-12:])
    ema26 = np.mean(close[-26:])
    macd = ema12 - ema26
    signal_line = macd * 0.5
    
    # EMA
    ema8 = np.mean(close[-8:])
    ema20 = np.mean(close[-20:])
    
    # Volume
    vol_avg = np.mean(volume[-20:])
    vol_ratio = volume[-1] / max(vol_avg, 1)
    
    # ATR
    tr = np.maximum(data['high'][1:] - data['low'][1:], np.maximum(np.abs(data['high'][1:] - close[:-1]), np.abs(data['low'][1:] - close[:-1])))
    atr = np.mean(tr[-14:])
    
    # امتیاز
    buy_score = 0
    sell_score = 0
    
    if rsi < 35: buy_score += 25
    elif rsi < 40: buy_score += 15
    if rsi > 65: sell_score += 25
    elif rsi > 60: sell_score += 15
    if macd > signal_line: buy_score += 20
    if macd < signal_line: sell_score += 20
    if ema8 > ema20: buy_score += 15
    if ema8 < ema20: sell_score += 15
    if vol_ratio > 1.5: buy_score += 15
    if vol_ratio > 1.5: sell_score += 15
    
    return {
        'price': price, 'rsi': rsi, 'macd': macd, 'signal': signal_line,
        'ema8': ema8, 'ema20': ema20, 'vol_ratio': vol_ratio, 'atr': atr,
        'buy_score': buy_score, 'sell_score': sell_score
    }

def run_monitor():
    now = datetime.now(tehran_tz)
    
    symbols = ['ETH-USDT-SWAP', 'BTC-USDT-SWAP', 'SOL-USDT-SWAP']
    
    print(f"{'═'*55}")
    print(f"  📡 HERMES Market Monitor")
    print(f"  📅 {now.strftime('%Y-%m-%d %H:%M')} Tehran")
    print(f"{'═'*55}")
    
    best_signal = None
    best_score = 0
    
    for symbol in symbols:
        try:
            result = analyze(symbol)
            name = symbol.replace('-USDT-SWAP', '')
            
            if result['buy_score'] > result['sell_score'] and result['buy_score'] > best_score:
                best_score = result['buy_score']
                best_signal = ('BUY', name, result)
            elif result['sell_score'] > result['buy_score'] and result['sell_score'] > best_score:
                best_score = result['sell_score']
                best_signal = ('SELL', name, result)
            
            print(f"\n  {name}: ${result['price']:,.2f}")
            print(f"    RSI: {result['rsi']:.1f} | MACD: {result['macd']:.2f} {'✅' if result['macd'] > result['signal'] else '❌'}")
            print(f"    EMA8: {result['ema8']:.2f} > EMA20: {result['ema20']:.2f} {'✅' if result['ema8'] > result['ema20'] else '❌'}")
            print(f"    Volume: {result['vol_ratio']:.2f}x")
            print(f"    Buy: {result['buy_score']} | Sell: {result['sell_score']}")
        except:
            pass
    
    print(f"\n{'═'*55}")
    
    if best_signal and best_score >= 50:
        direction, name, result = best_signal
        print(f"  🎯 بهترین فرصت: {name} — {direction}")
        print(f"  📍 قیمت: ${result['price']:,.2f}")
        print(f"  📊 امتیاز: {best_score}/100")
        
        if direction == 'BUY':
            stop = result['price'] - result['atr'] * 1.5
            t1 = result['price'] + result['atr'] * 1.5
        else:
            stop = result['price'] + result['atr'] * 1.5
            t1 = result['price'] - result['atr'] * 1.5
        
        print(f"  🛑 استاپ: ${stop:,.2f}")
        print(f"  🎯 تارگت: ${t1:,.2f}")
    else:
        print(f"  ⏸ هنوز فرصت مناسبی نیست — صبر کن")
    
    print(f"{'═'*55}")

if __name__ == '__main__':
    run_monitor()
