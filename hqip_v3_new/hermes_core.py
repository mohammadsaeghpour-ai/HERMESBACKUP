"""
HERMES Core Engine — بازسازی کامل
🎯 هدف: 65%+ Win Rate
📐 7 اندیکاتور + مدیریت ریسک + روانشناسی + پارサ
"""
import numpy as np


class HermesIndicators:
    """محاسبه 7 اندیکاتور اصلی"""

    @staticmethod
    def rsi(closes, period=14):
        deltas = np.diff(closes)
        gains = np.mean(deltas[deltas > 0][-period:]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0][-period:]) if np.any(deltas < 0) else 1e-8
        return 100 - (100 / (1 + gains / max(losses, 1e-8)))

    @staticmethod
    def macd(closes, fast=12, slow=26, signal=9):
        ema_fast = closes[-fast:].mean()
        ema_slow = closes[-slow:].mean()
        macd_val = ema_fast - ema_slow
        signal_val = macd_val * 0.5
        return macd_val, signal_val

    @staticmethod
    def bollinger_bands(closes, period=20, std_dev=2):
        mid = closes[-period:].mean()
        std = closes[-period:].std()
        return mid - std_dev * std, mid, mid + std_dev * std

    @staticmethod
    def atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:])

    @staticmethod
    def adx(highs, lows, closes, period=14):
        plus_dm = np.maximum(np.diff(highs), 0)
        minus_dm = np.maximum(-np.diff(lows), 0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        atr_arr = tr[-period:]
        plus_di = 100 * np.mean(plus_dm[-period:]) / max(np.mean(atr_arr), 1e-8)
        minus_di = 100 * np.mean(minus_dm[-period:]) / max(np.mean(atr_arr), 1e-8)
        dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-8)
        return dx

    @staticmethod
    def ema(closes, period):
        return closes[-period:].mean()

    @staticmethod
    def stochastic(highs, lows, closes, k_period=14, d_period=3):
        lowest = np.min(lows[-k_period:])
        highest = np.max(highs[-k_period:])
        k = 100 * (closes[-1] - lowest) / max(highest - lowest, 1e-8)
        return k, k  #简化版

    @staticmethod
    def volume_ratio(volumes, period=20):
        avg = np.mean(volumes[-period:])
        return volumes[-1] / max(avg, 1)


class HermesRiskManager:
    """مدیریت ریسک هوشمند"""

    @staticmethod
    def atr_stop(entry, atr, direction='long'):
        """استاپ ATR-Based"""
        stop_distance = atr * 1.5
        if direction == 'long':
            return entry - stop_distance
        else:
            return entry + stop_distance

    @staticmethod
    def atr_targets(entry, atr, direction='long'):
        """تارگت‌ها بر اساس ATR"""
        if direction == 'long':
            t1 = entry + atr * 1.5
            t2 = entry + atr * 2.5
            t3 = entry + atr * 3.5
        else:
            t1 = entry - atr * 1.5
            t2 = entry - atr * 2.5
            t3 = entry - atr * 3.5
        return t1, t2, t3

    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """Kelly Criterion for Position Sizing"""
        if avg_loss == 0:
            return 0
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly = (b * win_rate - q) / b
        return max(0, min(kelly, 0.25))  # حداکثر 25%

    @staticmethod
    def position_size(capital, risk_pct, entry, stop):
        """محاسبه سایز پوزیشن"""
        risk_amount = capital * risk_pct
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            return 0
        return risk_amount / stop_distance

    @staticmethod
    def trailing_stop(entry, current_price, atr, direction='long'):
        """Trailing Stop"""
        if direction == 'long':
            return current_price - atr * 1.0
        else:
            return current_price + atr * 1.0

    @staticmethod
    def drawdown_check(trades, max_consecutive=3):
        """بررسی Drawdown"""
        if len(trades) < max_consecutive:
            return True
        consecutive_losses = 0
        for t in trades[-max_consecutive:]:
            if t < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
        return consecutive_losses < max_consecutive


class HermesWyckoff:
    """تحلیل Wyckoff"""

    @staticmethod
    def detect_phase(closes, volumes, rsi):
        """تشخیص فاز Wyckoff"""
        recent_trend = closes[-1] - closes[-10]
        vol_trend = np.mean(volumes[-5:]) - np.mean(volumes[-10:])

        if recent_trend < 0 and rsi < 35 and vol_trend < 0:
            return "ACCUMULATION", "تجمیع — نهنگ‌ها می‌خرن"
        elif recent_trend > 0 and rsi > 65 and vol_trend > 0:
            return "DISTRIBUTION", "توزیع — نهنگ‌ها می‌فروشن"
        elif recent_trend > 0 and vol_trend > 0:
            return "MARKUP", "صعودی — روند صعودی فعال"
        elif recent_trend < 0 and vol_trend > 0:
            return "MARKDOWN", "نزولی — روند نزولی فعال"
        else:
            return "ACCUMULATION", "تجمیع — بازار خنثی"


class HermesVolumeProfile:
    """Volume Profile"""

    @staticmethod
    def poc(highs, lows, closes, volumes):
        """Point of Control"""
        price_range = np.linspace(np.min(lows), np.max(highs), 50)
        vol_at_price = np.zeros(50)
        for i in range(len(closes)):
            idx = np.argmin(np.abs(price_range - closes[i]))
            vol_at_price[idx] += volumes[i]
        return price_range[np.argmax(vol_at_price)]

    @staticmethod
    def value_area(highs, lows, closes, volumes, pct=0.70):
        """Value Area High/Low"""
        poc = HermesVolumeProfile.poc(highs, lows, closes, volumes)
        price_range = np.linspace(np.min(lows), np.max(highs), 50)
        vol_at_price = np.zeros(50)
        for i in range(len(closes)):
            idx = np.argmin(np.abs(price_range - closes[i]))
            vol_at_price[idx] += volumes[i]

        total_vol = np.sum(vol_at_price)
        poc_idx = np.argmax(vol_at_price)
        cumulative = vol_at_price[poc_idx]
        va_high_idx = poc_idx
        va_low_idx = poc_idx

        while cumulative < total_vol * pct:
            expand_up = vol_at_price[va_high_idx + 1] if va_high_idx + 1 < 50 else 0
            expand_down = vol_at_price[va_low_idx - 1] if va_low_idx - 1 >= 0 else 0
            if expand_up >= expand_down:
                va_high_idx += 1
                cumulative += expand_up
            else:
                va_low_idx -= 1
                cumulative += expand_down

        return price_range[va_low_idx], price_range[va_high_idx]


class HermesParsa:
    """تأثیرگذاری پارسا — 5 لایه"""

    @staticmethod
    def frequency_zero(rsi, bb_lower, bb_upper, stoch_k):
        """فرکانس صفر — اشباع شدید"""
        score = 0
        reasons = []

        if rsi < 25 or rsi > 75:
            score += 20
            reasons.append(f"RSI اشباع شدید ({rsi:.1f})")
        elif rsi < 30 or rsi > 70:
            score += 10
            reasons.append(f"RSI نزدیک اشباع ({rsi:.1f})")

        if stoch_k < 20 or stoch_k > 80:
            score += 10
            reasons.append(f"Stochastic اشباع ({stoch_k:.1f})")

        return score, reasons

    @staticmethod
    def shock_aligned(macd, signal_line, closes, ema21, volume_ratio):
        """شوک همراستا"""
        score = 0
        reasons = []

        if macd > signal_line and closes[-1] > ema21:
            score += 15
            reasons.append("شوک همراستا صعودی")
        elif macd < signal_line and closes[-1] < ema21:
            score += 15
            reasons.append("شوک همراستا نزولی")
        else:
            score -= 10
            reasons.append("شوک ناهمراستا")

        if volume_ratio > 1.5:
            score += 5
            reasons.append("Volume تأیید شوک")

        return score, reasons

    @staticmethod
    def negentropy(volume_ratio, adx, atr, price):
        """نگنتروپی — مقاومت در برابر آنتروپی"""
        score = 0
        reasons = []

        if volume_ratio > 1.5:
            score += 10
            reasons.append(f"Volume بالا ({volume_ratio:.2f}x)")

        if adx > 25:
            score += 5
            reasons.append(f"روند قوی (ADX={adx:.1f})")

        atr_pct = (atr / price) * 100
        if atr_pct < 1.0:
            score += 5
            reasons.append(f"نوسان کم ({atr_pct:.2f}%)")

        return score, reasons

    @staticmethod
    def conscious_refusal(rsi, macd, adx, closes, ema21, ema50):
        """امتناع آگاهانه — تشخیص بن‌بست"""
        score = 0
        reasons = []

        # بن‌بست: RSI خنثی + MACD خنثی + ADX ضعیف
        rsi_neutral = 40 < rsi < 60
        macd_neutral = abs(macd) < 5
        adx_weak = adx < 20

        if rsi_neutral and macd_neutral and adx_weak:
            score -= 20
            reasons.append("بن‌بست واقعی — امتناع آگاهانه")
        elif rsi_neutral and adx_weak:
            score -= 10
            reasons.append("بن‌بست جزئی")

        return score, reasons

    @staticmethod
    def comprehensive_impact(confirmations):
        """تاثیرگذاری جامع — تأیید چند لایه‌ای"""
        score = 0
        reasons = []

        if confirmations >= 5:
            score += 20
            reasons.append(f"تأثیرگذاری جامع قوی ({confirmations} تأیید)")
        elif confirmations >= 4:
            score += 15
            reasons.append(f"تأثیرگذاری جامع ({confirmations} تأیید)")
        elif confirmations >= 3:
            score += 10
            reasons.append(f"تأثیرگذاری جامع ({confirmations} تأیید)")

        return score, reasons
