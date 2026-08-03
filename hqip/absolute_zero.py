"""
HQIP Absolute Zero — Learn Market From Scratch
================================================
No preconceptions. Just price action + volume + time.
Learns from small moves ($100) to big moves ($1000+).
Session-aware: Asia / Europe / America.
"""
import sys
sys.path.insert(0, "/data/workspace")
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# ── SESSION DEFINITIONS ───────────────────────────────────
SESSIONS = {
    "asia":    {"start": 0,  "end": 8,  "label": "آسیا", "typical_range_pct": 0.3},
    "europe":  {"start": 7,  "end": 16, "label": "اروپا", "typical_range_pct": 0.6},
    "america": {"start": 13, "end": 22, "label": "آمریکا", "typical_range_pct": 0.9},
    "overlap": {"start": 13, "end": 16, "label": "همپوشانی", "typical_range_pct": 1.2},
}

def get_session(hour_utc):
    if 13 <= hour_utc < 16:
        return "overlap", "همپوشانی اروپا/آمریکا 🇪🇺🇺🇸"
    for key, s in SESSIONS.items():
        if s["start"] <= hour_utc < s["end"]:
            return key, s["label"]
    return "off", "خارج از ساعات"


class AbsoluteZeroEngine:
    """
    Starts from zero knowledge.
    Learns by observing:
    1. Price velocity (how fast price moves)
    2. Volume intensity (how much volume confirms)
    3. Range expansion (small range → big range = breakout)
    4. Session patterns (what happens in each session)
    5. Multi-timeframe agreement (alignment = conviction)
    """
    
    def __init__(self):
        self.observations = []
    
    def observe(self, df, tf):
        """Observe raw market data — no assumptions."""
        if df is None or df.empty or len(df) < 20:
            return None
        
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        
        p = closes[-1]
        
        # ── VELOCITY: How fast is price moving? ──
        # Compare last 5 candles vs previous 5
        recent_move = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) > 5 else 0
        older_move = (closes[-6] - closes[-11]) / closes[-11] * 100 if len(closes) > 10 else 0
        velocity = recent_move - older_move  # positive = accelerating up
        
        # ── VOLUME INTENSITY ──
        avg_vol = np.mean(volumes[-20:])
        recent_vol = np.mean(volumes[-3:])
        vol_intensity = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        # ── RANGE: How big are candles? ──
        ranges = (highs - lows) / closes * 100
        avg_range = np.mean(ranges[-20:])
        recent_range = np.mean(ranges[-3:])
        range_expansion = recent_range / avg_range if avg_range > 0 else 1.0
        
        # ── ABSORPTION: High volume + small range = absorption ──
        absorption = vol_intensity / range_expansion if range_expansion > 0 else 1.0
        
        # ── DIRECTION BIAS ──
        green = sum(1 for i in range(-5, 0) if closes[i] > closes[i-1])
        red = 5 - green
        direction_bias = (green - red) / 5  # +1 = all green, -1 = all red
        
        # ── SWING STRUCTURE ──
        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        # Structure: are we making higher highs + higher lows?
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_highs = swing_highs[-1] > swing_highs[-2]
            higher_lows = swing_lows[-1] > swing_lows[-2]
            lower_highs = swing_highs[-1] < swing_highs[-2]
            lower_lows = swing_lows[-1] < swing_lows[-2]
            
            if higher_highs and higher_lows:
                structure = "UP"
            elif lower_highs and lower_lows:
                structure = "DOWN"
            else:
                structure = "RANGE"
        else:
            structure = "UNKNOWN"
        
        # ── KEY LEVELS ──
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        mid = (recent_high + recent_low) / 2
        position_in_range = (p - recent_low) / (recent_high - recent_low) * 100 if recent_high != recent_low else 50
        
        return {
            "price": p,
            "velocity": velocity,
            "vol_intensity": vol_intensity,
            "range_expansion": range_expansion,
            "absorption": absorption,
            "direction_bias": direction_bias,
            "structure": structure,
            "position_in_range": position_in_range,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "swing_highs": swing_highs[-3:],
            "swing_lows": swing_lows[-3:],
            "avg_range_pct": avg_range,
            "tf": tf,
        }
    
    def decide(self, observations_by_tf, session_key):
        """
        Make decision from observations across all timeframes.
        Returns per-TF signals with proper entry/SL/TP.
        """
        # Session adjustment
        session_ranges = {
            "asia": 0.3, "europe": 0.6, "america": 0.9,
            "overlap": 1.2, "off": 0.2
        }
        expected_range = session_ranges.get(session_key, 0.5)
        
        signals = {}
        
        for tf, obs in observations_by_tf.items():
            if obs is None:
                signals[tf] = {"direction": "NO_TRADE", "confidence": 0, "reason": "insufficient data"}
                continue
            
            score = 0
            reasons = []
            
            # ── FACTOR 1: Structure (40% weight) ──
            if obs["structure"] == "UP":
                score += 0.40
                reasons.append(f"🟢 ساختار صعودی (HH+HL)")
            elif obs["structure"] == "DOWN":
                score -= 0.40
                reasons.append(f"🔴 ساختار نزولی (LH+LL)")
            else:
                reasons.append(f"🟡 رِنج")
            
            # ── FACTOR 2: Velocity (25% weight) ──
            vel = obs["velocity"]
            if vel > 0.3:
                score += 0.25
                reasons.append(f"🟢 شتاب صعودی ({vel:+.2f}%)")
            elif vel < -0.3:
                score -= 0.25
                reasons.append(f"🔴 شتاب نزولی ({vel:+.2f}%)")
            elif vel > 0.05:
                score += 0.05
                reasons.append(f"🟡 حرکت آهسته صعودی")
            elif vel < -0.05:
                score -= 0.05
                reasons.append(f"🟡 حرکت آهسته نزولی")
            
            # ── FACTOR 3: Volume confirms (20% weight) ──
            if obs["vol_intensity"] > 1.5:
                # High volume confirms the direction
                if obs["direction_bias"] > 0.2:
                    score += 0.20
                    reasons.append(f"🟢 حجم بالا + کندل‌های سبز ({obs['vol_intensity']:.1f}x)")
                elif obs["direction_bias"] < -0.2:
                    score -= 0.20
                    reasons.append(f"🔴 حجم بالا + کندل‌های قرمز ({obs['vol_intensity']:.1f}x)")
                else:
                    reasons.append(f"📊 حجم بالا ولی جهت نامشخص")
            elif obs["absorption"] > 2.0:
                reasons.append(f"🔵 جذب نهادی (حجم بالا + دامنه کم)")
            
            # ── FACTOR 4: Range position (15% weight) ──
            pos = obs["position_in_range"]
            if pos > 85:
                score -= 0.15
                reasons.append(f"🔴 نزدیک مقاومت ({pos:.0f}%)")
            elif pos < 15:
                score += 0.15
                reasons.append(f"🟢 نزدیک حمایت ({pos:.0f}%)")
            elif pos > 60:
                score -= 0.05
                reasons.append(f"🟡 بالای میانه ({pos:.0f}%)")
            elif pos < 40:
                score += 0.05
                reasons.append(f"🟢 زیر میانه ({pos:.0f}%)")
            
            # ── FACTOR 5: Session context (bonus) ──
            if session_key == "overlap":
                score *= 1.10
                reasons.append(f"⚡ سشن همپوشانی — نقدینگی بالا")
            elif session_key == "asia":
                score *= 0.85
                reasons.append(f"⚠️ سشن آسیا — نقدینگی پایین")
            
            # ── DECISION ──
            direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NO_TRADE"
            confidence = min(100, abs(score) * 150 + 20)
            
            # ── ENTRY / SL / TP ──
            price = obs["price"]
            atr_proxy = obs["avg_range_pct"]
            
            # Dynamic SL based on ATR of timeframe
            sl_pct = max(0.3, min(1.5, atr_proxy * 2.0))
            tp1_pct = sl_pct * 2.0   # R:R = 1:2
            tp2_pct = sl_pct * 3.0   # R:R = 1:3
            tp3_pct = sl_pct * 4.0   # R:R = 1:4
            
            # Adjust for session
            sl_pct *= (expected_range / 0.6)  # normalize to typical
            tp1_pct *= (expected_range / 0.6)
            tp2_pct *= (expected_range / 0.6)
            tp3_pct *= (expected_range / 0.6)
            
            # Use swing levels for SL when available
            if direction == "SELL" and obs["swing_highs"]:
                swing_sl = max(obs["swing_highs"]) * 1.001
                swing_sl_pct = (swing_sl - price) / price * 100
                sl_pct = max(sl_pct, min(swing_sl_pct, 2.0))
                tp1_pct = sl_pct * 2.0
                tp2_pct = sl_pct * 3.0
                tp3_pct = sl_pct * 4.0
            elif direction == "BUY" and obs["swing_lows"]:
                swing_sl = min(obs["swing_lows"]) * 0.999
                swing_sl_pct = (price - swing_sl) / price * 100
                sl_pct = max(sl_pct, min(swing_sl_pct, 2.0))
                tp1_pct = sl_pct * 2.0
                tp2_pct = sl_pct * 3.0
                tp3_pct = sl_pct * 4.0
            
            if direction == "BUY":
                entry = price
                sl = price * (1 - sl_pct / 100)
                tp1 = price * (1 + tp1_pct / 100)
                tp2 = price * (1 + tp2_pct / 100)
                tp3 = price * (1 + tp3_pct / 100)
            elif direction == "SELL":
                entry = price
                sl = price * (1 + sl_pct / 100)
                tp1 = price * (1 - tp1_pct / 100)
                tp2 = price * (1 - tp2_pct / 100)
                tp3 = price * (1 - tp3_pct / 100)
            else:
                entry = sl = tp1 = tp2 = tp3 = price
            
            signals[tf] = {
                "direction": direction,
                "confidence": round(confidence, 1),
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp1": round(tp1, 2),
                "tp2": round(tp2, 2),
                "tp3": round(tp3, 2),
                "sl_pct": round(sl_pct, 2),
                "tp1_pct": round(tp1_pct, 2),
                "tp2_pct": round(tp2_pct, 2),
                "tp3_pct": round(tp3_pct, 2),
                "score": round(score, 3),
                "reasons": reasons,
                "structure": obs["structure"],
                "velocity": round(obs["velocity"], 3),
                "vol_intensity": round(obs["vol_intensity"], 2),
                "position_in_range": round(obs["position_in_range"], 1),
            }
        
        return signals
