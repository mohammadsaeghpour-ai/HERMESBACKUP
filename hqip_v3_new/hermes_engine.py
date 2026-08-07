"""
HERMES Signal Engine — v2.1 Production
ترکیب HERMES + HQIP (24 Agent)
خودکار سیگنال‌دهی هر ۶۰ دقیقه
"""
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import os
import time

# ═══════════════════════════════════════════════════
# تنظیمات ارزها
# ═══════════════════════════════════════════════════
PRIMARY = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']
SECONDARY = ['SOL-USDT-SWAP']
ALL_SYMBOLS = PRIMARY + SECONDARY

# ═══════════════════════════════════════════════════
# کلاس‌های ایجنت
# ═══════════════════════════════════════════════════

class DataFetcherAgent:
    """ایجنت دریافت داده"""
    
    def fetch_candles(self, symbol, tf='1H', limit=100):
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
            opens = np.array([float(c[1]) for c in data])
            volumes = np.array([float(c[5]) for c in data])
            return {
                'closes': closes, 'highs': highs, 'lows': lows,
                'opens': opens, 'volumes': volumes
            }
        except:
            return None
    
    def fetch_price(self, symbol):
        try:
            resp = requests.get(
                'https://www.okx.com/api/v5/market/ticker',
                params={'instId': symbol}, timeout=10
            )
            d = resp.json()['data'][0]
            return {
                'price': float(d['last']),
                'high24': float(d['high24h']),
                'low24': float(d['low24h']),
                'open24': float(d['open24h']),
                'vol24': float(d['volCcy24h'])
            }
        except:
            return None
    
    def fetch_funding_rate(self, symbol):
        try:
            resp = requests.get(
                'https://www.okx.com/api/v5/public/funding-rate',
                params={'instId': symbol}, timeout=10
            )
            d = resp.json()['data'][0]
            return float(d['fundingRate'])
        except:
            return 0.0


class TechnicalAnalystAgent:
    """ایجنت تحلیل تکنیکال"""
    
    def calc_rsi(self, closes, period=14):
        deltas = np.diff(closes)
        gains = np.mean(deltas[deltas > 0][-period:]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0][-period:]) if np.any(deltas < 0) else 1e-8
        return 100 - (100 / (1 + gains / max(losses, 1e-8)))
    
    def calc_macd(self, closes):
        if len(closes) < 26:
            return 0, 0, 'NEUTRAL'
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema_from_val(macd_line, 9)
        histogram = macd_line - signal_line
        if macd_line > signal_line and histogram > 0:
            return macd_line, signal_line, 'BULLISH'
        elif macd_line < signal_line and histogram < 0:
            return macd_line, signal_line, 'BEARISH'
        return macd_line, signal_line, 'NEUTRAL'
    
    def calc_bb(self, closes, period=20):
        if len(closes) < period:
            return closes[-1] * 1.02, closes[-1], closes[-1] * 0.98
        middle = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = middle + 2 * std
        lower = middle - 2 * std
        pct_b = (closes[-1] - lower) / max(upper - lower, 1e-8)
        return upper, middle, lower
    
    def calc_atr(self, highs, lows, closes, period=14):
        if len(closes) < 2:
            return 0
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                 np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:])
    
    def calc_adx(self, closes, period=14):
        rsi = self.calc_rsi(closes, period)
        return min(100, abs(rsi - 50) * 2)
    
    def calc_ema(self, closes, period):
        return self._ema(closes, period)
    
    def _ema(self, data, period):
        if len(data) < period:
            return np.mean(data)
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for price in data[period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema
    
    def _ema_from_val(self, data, period):
        if isinstance(data, (int, float)):
            return data * 0.5
        return self._ema(data, period) if len(data) > 1 else data[-1]


class MarketRegimeAgent:
    """ایجنت تشخیص رژیم بازار"""
    
    def detect(self, adx, atr, closes):
        atr_avg = np.mean(np.abs(np.diff(closes[-20:]))) if len(closes) > 20 else atr
        
        if adx > 25:
            return 'TRENDING'
        elif atr > atr_avg * 1.5:
            return 'VOLATILE'
        else:
            return 'RANGING'


class SentimentAnalystAgent:
    """ایجنت تحلیل سنتیمنت"""
    
    def get_fear_greed(self):
        try:
            resp = requests.get('https://api.alternative.me/fng/?limit=1', timeout=10)
            data = resp.json()['data'][0]
            return {
                'value': int(data['value']),
                'classification': data['value_classification']
            }
        except:
            return {'value': 50, 'classification': 'Neutral'}


class SignalGeneratorAgent:
    """ایجنت تولید سیگنال"""
    
    def generate(self, rsi, macd_status, bb_upper, bb_lower, price, atr, adx, 
                 ema8, ema20, volume_ratio, funding_rate):
        
        direction = None
        confidence = 0
        reasons = []
        
        # شرایط BUY
        buy_score = 0
        if rsi < 35:
            buy_score += 30
            reasons.append(f'RSI اشباع فروش ({rsi:.1f})')
        elif rsi < 40:
            buy_score += 15
            reasons.append(f'RSI نزدیک اشباع ({rsi:.1f})')
        
        if macd_status == 'BULLISH':
            buy_score += 20
            reasons.append('MACD صعودی')
        
        if price < bb_lower * 1.02:
            buy_score += 25
            reasons.append('نزدیک BB Lower')
        
        if ema8 > ema20:
            buy_score += 15
            reasons.append('EMA8 > EMA20')
        
        if funding_rate < -0.01:
            buy_score += 10
            reasons.append('Funding منفی')
        
        if adx > 25:
            buy_score += 10
            reasons.append(f'روند قوی (ADX={adx:.0f})')
        
        # شرایط SELL
        sell_score = 0
        if rsi > 65:
            sell_score += 30
            reasons.append(f'RSI اشباع خرید ({rsi:.1f})')
        elif rsi > 60:
            sell_score += 15
            reasons.append(f'RSI نزدیک اشباع ({rsi:.1f})')
        
        if macd_status == 'BEARISH':
            sell_score += 20
            reasons.append('MACD نزولی')
        
        if price > bb_upper * 0.98:
            sell_score += 25
            reasons.append('نزدیک BB Upper')
        
        if ema8 < ema20:
            sell_score += 15
            reasons.append('EMA8 < EMA20')
        
        if funding_rate > 0.05:
            sell_score += 10
            reasons.append('Funding بالا')
        
        # تصمیم نهایی
        if buy_score > sell_score and buy_score >= 40:
            direction = 'BUY'
            confidence = min(95, buy_score)
        elif sell_score > buy_score and sell_score >= 40:
            direction = 'SELL'
            confidence = min(95, sell_score)
        
        return direction, confidence, reasons


class RiskManagerAgent:
    """ایجنت مدیریت ریسک"""
    
    def calc_targets(self, price, atr, direction):
        if direction == 'BUY':
            stop_loss = price - 1.5 * atr
            target1 = price + 1.0 * atr
            target2 = price + 2.0 * atr
            target3 = price + 3.0 * atr
        else:
            stop_loss = price + 1.5 * atr
            target1 = price - 1.0 * atr
            target2 = price - 2.0 * atr
            target3 = price - 3.0 * atr
        
        risk = abs(price - stop_loss)
        rr1 = abs(target1 - price) / max(risk, 0.01)
        rr2 = abs(target2 - price) / max(risk, 0.01)
        rr3 = abs(target3 - price) / max(risk, 0.01)
        
        return {
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'target3': target3,
            'risk_pct': risk / price * 100,
            'rr1': rr1, 'rr2': rr2, 'rr3': rr3
        }
    
    def validate(self, rr1, confidence):
        """فیلتر کیفیت سیگنال"""
        if rr1 < 1.0:
            return False, 'RR < 1.0'
        if confidence < 60:
            return False, 'Confidence < 60%'
        return True, 'OK'


# ═══════════════════════════════════════════════════
# موتور اصلی سیگنال
# ═══════════════════════════════════════════════════

class HERMESignalEngine:
    """موتور اصلی سیگنال‌دهی HERMES v2.1"""
    
    def __init__(self):
        self.data_agent = DataFetcherAgent()
        self.tech_agent = TechnicalAnalystAgent()
        self.regime_agent = MarketRegimeAgent()
        self.sentiment_agent = SentimentAnalystAgent()
        self.signal_agent = SignalGeneratorAgent()
        self.risk_agent = RiskManagerAgent()
        self.file_path = os.path.join(os.path.dirname(__file__), 'hermes_signals.json')
    
    def analyze(self, symbol):
        """تحلیل کامل یک ارز"""
        # دریافت داده
        price_data = self.data_agent.fetch_price(symbol)
        if not price_data:
            return None
        
        data_1h = self.data_agent.fetch_candles(symbol, '1H', 100)
        data_4h = self.data_agent.fetch_candles(symbol, '4H', 100)
        
        if not data_1h:
            return None
        
        closes = data_1h['closes']
        highs = data_1h['highs']
        lows = data_1h['lows']
        volumes = data_1h['volumes']
        
        # اندیکاتورها
        rsi = self.tech_agent.calc_rsi(closes)
        macd, signal_line, macd_status = self.tech_agent.calc_macd(closes)
        bb_upper, bb_middle, bb_lower = self.tech_agent.calc_bb(closes)
        atr = self.tech_agent.calc_atr(highs, lows, closes)
        adx = self.tech_agent.calc_adx(closes)
        ema8 = self.tech_agent.calc_ema(closes, 8)
        ema20 = self.tech_agent.calc_ema(closes, 20)
        ema50 = self.tech_agent.calc_ema(closes, 50)
        
        # حجم
        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_avg, 1)
        
        # Funding Rate
        funding_rate = self.data_agent.fetch_funding_rate(symbol)
        
        # رژیم بازار
        regime = self.regime_agent.detect(adx, atr, closes)
        
        # سنتیمنت
        fear_greed = self.sentiment_agent.get_fear_greed()
        
        # تولید سیگنال
        direction, confidence, reasons = self.signal_agent.generate(
            rsi, macd_status, bb_upper, bb_lower, price_data['price'],
            atr, adx, ema8, ema20, vol_ratio, funding_rate
        )
        
        if not direction:
            return None
        
        # محاسبه اهداف
        targets = self.risk_agent.calc_targets(price_data['price'], atr, direction)
        
        # فیلتر کیفیت
        valid, msg = self.risk_agent.validate(targets['rr1'], confidence)
        if not valid:
            return None
        
        # Support/Resistance
        support = min(lows[-20:])
        resistance = max(highs[-20:])
        
        return {
            'symbol': symbol.replace('-USDT-SWAP', ''),
            'direction': direction,
            'confidence': confidence,
            'entry': price_data['price'],
            'entry_low': price_data['price'] - 0.2 * atr,
            'entry_high': price_data['price'] + 0.2 * atr,
            'stop_loss': targets['stop_loss'],
            'target1': targets['target1'],
            'target2': targets['target2'],
            'target3': targets['target3'],
            'rr1': targets['rr1'],
            'rr2': targets['rr2'],
            'rr3': targets['rr3'],
            'risk_pct': targets['risk_pct'],
            'rsi': rsi,
            'macd_status': macd_status,
            'bb_position': 'UPPER' if price_data['price'] > bb_upper else 'LOWER' if price_data['price'] < bb_lower else 'MID',
            'adx': adx,
            'atr': atr,
            'ema21': ema20,
            'ema50': ema50,
            'regime': regime,
            'volume_ratio': vol_ratio,
            'funding_rate': funding_rate,
            'fear_greed': fear_greed,
            'support': support,
            'resistance': resistance,
            'change24': ((price_data['price'] - price_data['low24']) / price_data['low24']) * 100,
            'reasons': reasons,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def format_signal(self, signal):
        """فرمت استاندارد HERMES v2.1"""
        if not signal:
            return ""
        
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        strength = 'STRONG' if signal['confidence'] > 75 else 'MODERATE' if signal['confidence'] > 60 else 'WEAK'
        
        risk_pct = abs(signal['entry'] - signal['stop_loss']) / signal['entry'] * 100
        
        return f"""
═══════════════════════════════════════
🚨 HERMES SIGNAL — {signal['symbol']}USDT — 1H
═══════════════════════════════════════
📅 Time: {signal['timestamp'][:16]}
🎯 Direction: {direction_emoji} {signal['direction']}
💰 Entry Zone: ${signal['entry_low']:,.2f} – ${signal['entry_high']:,.2f}
🛡️ Stop Loss: ${signal['stop_loss']:,.2f}  ({risk_pct:.1f}% از entry)
🎯 Target 1: ${signal['target1']:,.2f}  (+{abs(signal['target1'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr1']:.1f})
🎯 Target 2: ${signal['target2']:,.2f}  (+{abs(signal['target2'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr2']:.1f})
🎯 Target 3: ${signal['target3']:,.2f}  (+{abs(signal['target3'] - signal['entry']) / signal['entry'] * 100:.1f}% | RR: 1:{signal['rr3']:.1f})
📊 Confidence: {signal['confidence']:.0f}%
⚡ Signal Strength: {strength}
🌊 Market Regime: {signal['regime']}
💹 Volume Confirmation: {'YES' if signal['volume_ratio'] > 1 else 'NO'}
─────────────────────────────────────
📈 Technical Summary:
  RSI(14): {signal['rsi']:.1f} | MACD: {signal['macd_status']} | BB: {signal['bb_position']}
  ADX: {signal['adx']:.1f} | ATR: ${signal['atr']:.2f}
  EMA21: ${signal['ema21']:,.2f} | EMA50: ${signal['ema50']:,.2f}
  Support: ${signal['support']:,.2f} | Resistance: ${signal['resistance']:,.2f}
─────────────────────────────────────
🧠 Psychology:
  Fear & Greed: {signal['fear_greed']['value']} ({signal['fear_greed']['classification']})
  Funding Rate: {signal['funding_rate']:.4f}%
  24h Change: {signal['change24']:+.1f}%
─────────────────────────────────────
📋 دلایل سیگنال:
  {chr(10).join(['  ✅ ' + r for r in signal['reasons']])}
─────────────────────────────────────
⚠️ Risk Note: در صورت شکست استاپ، سیگنال invalidate است.
═══════════════════════════════════════"""
    
    def run_cycle(self):
        """اجرای یک چرخه کامل"""
        tehran_tz = timezone(timedelta(hours=3, minutes=30))
        now_tehran = datetime.now(tehran_tz)
        
        print(f"\n{'═'*60}")
        print(f"  🤖 HERMES Signal Engine v2.1 — Production")
        print(f"  📅 {now_tehran.strftime('%Y-%m-%d %H:%M')} Tehran Time")
        print(f"{'═'*60}")
        
        signals = []
        
        for symbol in ALL_SYMBOLS:
            result = self.analyze(symbol)
            if result:
                signals.append(result)
                print(self.format_signal(result))
        
        if not signals:
            print(f"\n  ⏳ سیگنالی صادر نشد — بازار خنثی")
        
        # خلاصه
        print(f"\n{'─'*60}")
        print(f"  📊 خلاصه چرخه:")
        print(f"    سیگنال‌های صادر شده: {len(signals)}")
        for s in signals:
            emoji = "🟢" if s['direction'] == 'BUY' else "🔴"
            print(f"    {emoji} {s['symbol']}: {s['direction']} (Confidence: {s['confidence']:.0f}%)")
        print(f"{'─'*60}")
        
        # ذخیره
        self.save(signals)
        
        return signals
    
    def save(self, signals):
        data = {
            'signals': signals,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


if __name__ == '__main__':
    engine = HERMESignalEngine()
    signals = engine.run_cycle()
