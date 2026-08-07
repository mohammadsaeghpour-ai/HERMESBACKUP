"""
HERMES Signal Engine — خودکار سیگنال‌دهی
هر ساعت چرخه کامل تحلیل + سیگنال
"""
import requests
import numpy as np
from datetime import datetime
import json
import os

# ۱۰ ارز برتر بازار
TOP_10 = [
    'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
    'XRP-USDT-SWAP', 'ADA-USDT-SWAP', 'DOGE-USDT-SWAP', 'AVAX-USDT-SWAP',
    'DOT-USDT-SWAP', 'LINK-USDT-SWAP'
]

# ارزهای اصلی (تمرکز بیشتر)
PRIMARY = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']


class SignalEngine:
    """موتور سیگنال‌دهی خودکار"""
    
    def __init__(self):
        self.signals = []
        self.file_path = os.path.join(os.path.dirname(__file__), 'signals.json')
    
    def get_candles(self, symbol, tf='1H', limit=50):
        """دریافت کندل‌ها"""
        try:
            resp = requests.get(
                'https://www.okx.com/api/v5/market/candles',
                params={'instId': symbol, 'bar': tf, 'limit': str(limit)},
                timeout=10
            )
            data = resp.json()['data']
            closes = np.array([float(c[4]) for c in data])
            highs = np.array([float(c[2]) for c in data])
            lows = np.array([float(c[3]) for c in data])
            volumes = np.array([float(c[5]) for c in data])
            return closes, highs, lows, volumes
        except:
            return None, None, None, None
    
    def get_price(self, symbol):
        """دریافت قیمت لایو"""
        try:
            resp = requests.get(
                'https://www.okx.com/api/v5/market/ticker',
                params={'instId': symbol},
                timeout=10
            )
            data = resp.json()['data'][0]
            return {
                'price': float(data['last']),
                'high24': float(data['high24h']),
                'low24': float(data['low24h']),
                'open24': float(data['open24h']),
                'vol24': float(data['volCcy24h'])
            }
        except:
            return None
    
    def calc_rsi(self, closes, period=14):
        """محاسبه RSI"""
        deltas = np.diff(closes)
        gains = np.mean(deltas[deltas > 0][-period:]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0][-period:]) if np.any(deltas < 0) else 1e-8
        return 100 - (100 / (1 + gains / max(losses, 1e-8)))
    
    def calc_macd(self, closes):
        """محاسبه MACD"""
        if len(closes) < 26:
            return 0, 0
        ema12 = np.mean(closes[-12:])
        ema26 = np.mean(closes[-26:])
        macd = ema12 - ema26
        signal = macd * 0.5
        return macd, signal
    
    def calc_bb(self, closes, period=20):
        """محاسبه Bollinger Bands"""
        if len(closes) < period:
            return closes[-1] * 1.02, closes[-1], closes[-1] * 0.98
        middle = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        return middle + 2 * std, middle, middle - 2 * std
    
    def calc_atr(self, highs, lows, closes, period=14):
        """محاسبه ATR"""
        if len(closes) < 2:
            return 0
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                 np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:])
    
    def calc_adx(self, closes, period=14):
        """محاسبه ADX"""
        rsi = self.calc_rsi(closes, period)
        return min(100, abs(rsi - 50) * 2)
    
    def detect_regime(self, closes, adx, atr):
        """تشخیص رژیم بازار"""
        atr_avg = np.mean(np.abs(np.diff(closes[-20:]))) if len(closes) > 20 else atr
        
        if adx > 25:
            return "TRENDING"
        elif atr > atr_avg * 1.5:
            return "VOLATILE"
        else:
            return "RANGING"
    
    def analyze(self, symbol):
        """تحلیل کامل یک ارز"""
        # دریافت داده
        price_data = self.get_price(symbol)
        if not price_data:
            return None
        
        closes_1h, highs_1h, lows_1h, vols_1h = self.get_candles(symbol, '1H', 50)
        closes_4h, highs_4h, lows_4h, vols_4h = self.get_candles(symbol, '4H', 50)
        
        if closes_1h is None:
            return None
        
        # اندیکاتورها
        rsi = self.calc_rsi(closes_1h)
        macd, signal = self.calc_macd(closes_1h)
        bb_upper, bb_middle, bb_lower = self.calc_bb(closes_1h)
        atr = self.calc_atr(highs_1h, lows_1h, closes_1h)
        adx = self.calc_adx(closes_1h)
        
        # EMA
        ema8 = np.mean(closes_1h[-8:])
        ema20 = np.mean(closes_1h[-20:])
        ema50 = np.mean(closes_1h[-50:]) if len(closes_1h) >= 50 else np.mean(closes_1h)
        
        # رژیم بازار
        regime = self.detect_regime(closes_1h, adx, atr)
        
        # حجم
        vol_avg = np.mean(vols_1h[-20:]) if len(vols_1h) >= 20 else np.mean(vols_1h)
        vol_ratio = vols_1h[-1] / max(vol_avg, 1)
        
        # قیمت
        price = price_data['price']
        change24 = ((price - price_data['low24']) / price_data['low24']) * 100
        
        # Support/Resistance
        support = min(lows_1h[-20:])
        resistance = max(highs_1h[-20:])
        
        # تعیین جهت سیگنال
        direction = None
        confidence = 0
        
        if rsi < 35 and macd > signal and ema8 > ema20:
            direction = "BUY"
            confidence = min(90, 60 + (40 - rsi) * 2 + vol_ratio * 10)
        elif rsi > 65 and macd < signal and ema8 < ema20:
            direction = "SELL"
            confidence = min(90, 60 + (rsi - 60) * 2 + vol_ratio * 10)
        elif rsi < 40 and price < bb_lower * 1.01:
            direction = "BUY"
            confidence = min(85, 55 + (40 - rsi) * 1.5)
        elif rsi > 60 and price > bb_upper * 0.99:
            direction = "SELL"
            confidence = min(85, 55 + (rsi - 60) * 1.5)
        
        if direction is None:
            return None
        
        # محاسبه stop loss و targets
        if direction == "BUY":
            stop_loss = price - 1.5 * atr
            target1 = price + 1 * atr
            target2 = price + 2 * atr
            target3 = price + 3 * atr
        else:
            stop_loss = price + 1.5 * atr
            target1 = price - 1 * atr
            target2 = price - 2 * atr
            target3 = price - 3 * atr
        
        # R:R
        risk = abs(price - stop_loss)
        rr1 = abs(target1 - price) / max(risk, 0.01)
        rr2 = abs(target2 - price) / max(risk, 0.01)
        rr3 = abs(target3 - price) / max(risk, 0.01)
        
        return {
            'symbol': symbol.replace('-USDT-SWAP', ''),
            'direction': direction,
            'confidence': confidence,
            'entry': price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'target3': target3,
            'rr1': rr1,
            'rr2': rr2,
            'rr3': rr3,
            'rsi': rsi,
            'macd': "BULLISH" if macd > signal else "BEARISH",
            'adx': adx,
            'atr': atr,
            'regime': regime,
            'volume': vol_ratio,
            'support': support,
            'resistance': resistance,
            'change24': change24,
            'timestamp': datetime.now().isoformat()
        }
    
    def format_signal(self, signal):
        """فرمت سیگنال استاندارد"""
        if not signal:
            return ""
        
        direction_emoji = "🟢" if signal['direction'] == "BUY" else "🔴"
        
        return f"""
═══════════════════════════════════════
🚨 HERMES SIGNAL — {signal['symbol']}USDT — 1H
═══════════════════════════════════════
📅 Time: {signal['timestamp'][:16]}
🎯 Direction: {direction_emoji} {signal['direction']}
💰 Entry Zone: ${signal['entry']:,.2f}
🛡️ Stop Loss: ${signal['stop_loss']:,.2f} ({abs(signal['entry'] - signal['stop_loss']) / signal['entry'] * 100:.1f}%)
🎯 Target 1: ${signal['target1']:,.2f} (+{abs(signal['target1'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr1']:.1f})
🎯 Target 2: ${signal['target2']:,.2f} (+{abs(signal['target2'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr2']:.1f})
🎯 Target 3: ${signal['target3']:,.2f} (+{abs(signal['target3'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr3']:.1f})
📊 Confidence: {signal['confidence']:.0f}%
⚡ Signal Strength: {'STRONG' if signal['confidence'] > 75 else 'MODERATE' if signal['confidence'] > 60 else 'WEAK'}
🌊 Market Regime: {signal['regime']}
💹 Volume Confirmation: {'YES' if signal['volume'] > 1 else 'NO'}
─────────────────────────────────────
📈 Technical Summary:
  RSI(14): {signal['rsi']:.1f} | MACD: {signal['macd']} | ADX: {signal['adx']:.1f}
  ATR: ${signal['atr']:.2f} | Support: ${signal['support']:,.2f} | Resistance: ${signal['resistance']:,.2f}
═══════════════════════════════════════"""
    
    def run_cycle(self):
        """اجرای یک چرخه کامل"""
        print(f"\n{'='*60}")
        print(f"  🤖 HERMES Signal Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        signals = []
        
        # تحلیل ارزهای اصلی
        for symbol in PRIMARY:
            result = self.analyze(symbol)
            if result:
                signals.append(result)
                print(self.format_signal(result))
        
        # تحلیل ۱۰ ارز برتر
        for symbol in TOP_10:
            if symbol not in PRIMARY:
                result = self.analyze(symbol)
                if result and result['confidence'] > 70:
                    signals.append(result)
                    print(self.format_signal(result))
        
        if not signals:
            print("\n  ⏳ سیگنالی صادر نشد — بازار خنثی")
        
        # ذخیره
        self.signals = signals
        self.save()
        
        return signals
    
    def save(self):
        """ذخیره سیگنال‌ها"""
        data = {
            'signals': self.signals,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """بارگذاری سیگنال‌ها"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return None


if __name__ == '__main__':
    engine = SignalEngine()
    signals = engine.run_cycle()
    
    print(f"\n{'='*60}")
    print(f"  📊 خلاصه چرخه")
    print(f"{'='*60}")
    print(f"  سیگنال‌های صادر شده: {len(signals)}")
    for s in signals:
        print(f"    {s['symbol']}: {s['direction']} (Confidence: {s['confidence']:.0f}%)")
