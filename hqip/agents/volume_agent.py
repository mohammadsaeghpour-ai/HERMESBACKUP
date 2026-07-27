"""
Volume Agent
============
Analyzes OBV, VWAP, Volume Profile, accumulation/distribution.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class VolumeAgent(BaseAgent):
    name = "Volume"
    weight = 1.2

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0

        vol_ratio = df["vol_ratio"].iloc[-1]
        obv_last = df["obv"].iloc[-1]
        obv_ema = df["obv_ema"].iloc[-1]
        close = df["close"].iloc[-1]
        vwap_val = df["vwap"].iloc[-1]
        vwap_dist = df["vwap_dist"].iloc[-1]

        # Volume Spike
        if vol_ratio > 2.0:
            evidence.append(f"🔴 MAJOR Volume Spike: {vol_ratio:.1f}x avg")
            # Direction depends on candle
            if df["bullish_candle"].iloc[-1]:
                score += 0.4
                evidence.append("Volume spike on bullish candle = accumulation")
            else:
                score -= 0.4
                evidence.append("Volume spike on bearish candle = distribution")
        elif vol_ratio > 1.5:
            evidence.append(f"Volume elevated: {vol_ratio:.1f}x avg")
            score += 0.1 if df["bullish_candle"].iloc[-1] else -0.1
        elif vol_ratio < 0.5:
            evidence.append(f"Low volume: {vol_ratio:.1f}x avg (indecision)")
        else:
            evidence.append(f"Normal volume: {vol_ratio:.1f}x avg")

        # OBV
        if obv_last > obv_ema:
            evidence.append("OBV above its EMA = bullish volume flow")
            score += 0.2
        else:
            evidence.append("OBV below its EMA = bearish volume flow")
            score -= 0.2

        # OBV Trend (last 10 bars)
        obv_trend = df["obv"].iloc[-10:].values
        if len(obv_trend) > 1:
            obv_slope = np.polyfit(range(10), obv_trend, 1)[0]
            if obv_slope > 0:
                evidence.append("OBV trending up = sustained buying")
                score += 0.1
            else:
                evidence.append("OBV trending down = sustained selling")
                score -= 0.1

        # VWAP
        if vwap_dist > 0.5:
            evidence.append(f"Price {vwap_dist:.2f}% above VWAP (premium)")
            score += 0.1  # Can mean continuation or exhaustion
        elif vwap_dist < -0.5:
            evidence.append(f"Price {abs(vwap_dist):.2f}% below VWAP (discount)")
            score -= 0.1  # Can mean continuation or bounce
        else:
            evidence.append(f"Price near VWAP ({vwap_dist:.2f}%)")

        # Volume Profile
        try:
            from hqip.indicators import volume_profile
            poc, vah, val = volume_profile(df)
            if close > vah:
                evidence.append(f"Price above VAH ({vah:.2f}) = strong bullish")
                score += 0.15
            elif close < val:
                evidence.append(f"Price below VAL ({val:.2f}) = strong bearish")
                score -= 0.15
            elif close > poc:
                evidence.append(f"Price in upper range (POC: {poc:.2f})")
                score += 0.05
            else:
                evidence.append(f"Price in lower range (POC: {poc:.2f})")
                score -= 0.05
        except:
            pass

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 90)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            reasoning=f"Volume {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} | OBV {'↑' if obv_last > obv_ema else '↓'} | VWAP dist={vwap_dist:.2f}%"
        )
