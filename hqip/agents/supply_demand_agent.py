"""
Supply/Demand Agent — Zone Detection
====================================
Detects supply (sell) and demand (buy) zones from base patterns.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class SupplyDemandAgent(BaseAgent):
    name = "SupplyDemand"
    weight = 1.3

    def _find_zones(self, df):
        zones = []
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        vols = df["volume"].values

        for i in range(3, len(df) - 3):
            # Demand Zone: base (narrow range) followed by strong rally
            base_range = max(highs[i-1:i+2]) - min(lows[i-1:i+2])
            avg_range = np.mean([highs[j] - lows[j] for j in range(max(0,i-20), i)])

            if avg_range == 0: continue

            # Base = small candle body
            base_size = base_range / avg_range if avg_range > 0 else 1

            # Rally after base
            rally = (closes[i+3] - closes[i]) / closes[i] * 100 if i + 3 < len(df) else 0

            if base_size < 0.7 and rally > 1.5:
                # Demand zone: base before rally
                zones.append({
                    "type": "demand", "top": max(highs[i-1:i+2]),
                    "bottom": min(lows[i-1:i+2]), "strength": rally,
                    "fresh": not any(closes[j] < max(highs[i-1:i+2]) and closes[j] > min(lows[i-1:i+2])
                                    for j in range(i+4, len(df)))
                })

            # Supply Zone: base followed by strong drop
            drop = -(closes[i+3] - closes[i]) / closes[i] * 100 if i + 3 < len(df) else 0

            if base_size < 0.7 and drop > 1.5:
                zones.append({
                    "type": "supply", "top": max(highs[i-1:i+2]),
                    "bottom": min(lows[i-1:i+2]), "strength": drop,
                    "fresh": not any(closes[j] < max(highs[i-1:i+2]) and closes[j] > min(lows[i-1:i+2])
                                    for j in range(i+4, len(df)))
                })

        return zones

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0
        close = df["close"].iloc[-1]

        zones = self._find_zones(df)
        demand_zones = [z for z in zones if z["type"] == "demand"]
        supply_zones = [z for z in zones if z["type"] == "supply"]

        evidence.append(f"Zones found: {len(demand_zones)} demand, {len(supply_zones)} supply")

        # Check if price is near a fresh demand zone
        for dz in demand_zones[-3:]:
            if dz["bottom"] <= close <= dz["top"] * 1.02:
                fresh = "FRESH (never tested)" if dz["fresh"] else "tested"
                evidence.append(f"🟢 Price IN Demand Zone ({dz['bottom']:.2f}-{dz['top']:.2f}) [{fresh}]")
                evidence.append("   🏦 Base before rally — institutions built positions here")
                if dz["fresh"]:
                    score += 0.35
                    evidence.append("   ⚡ Fresh zone = strongest probability of bounce")
                else:
                    score += 0.15
            elif close > dz["top"] and dz["top"] > close * 0.99:
                dist = (close - dz["top"]) / close * 100
                evidence.append(f"🟢 Price above demand zone ({dz['bottom']:.2f}-{dz['top']:.2f}), {dist:.2f}% above")

        # Check if price is near a fresh supply zone
        for sz in supply_zones[-3:]:
            if sz["bottom"] * 0.98 <= close <= sz["top"]:
                fresh = "FRESH" if sz["fresh"] else "tested"
                evidence.append(f"🔴 Price IN Supply Zone ({sz['bottom']:.2f}-{sz['top']:.2f}) [{fresh}]")
                evidence.append("   🏦 Base before drop — institutions distributed here")
                if sz["fresh"]:
                    score -= 0.35
                    evidence.append("   ⚡ Fresh supply zone = high probability of rejection")
                else:
                    score -= 0.15

        # Nearest zones
        nearest_demand = max([z for z in demand_zones if z["top"] < close], key=lambda z: z["top"], default=None)
        nearest_supply = min([z for z in supply_zones if z["bottom"] > close], key=lambda z: z["bottom"], default=None)

        if nearest_demand:
            dist = (close - nearest_demand["top"]) / close * 100
            evidence.append(f"📍 Nearest demand: {nearest_demand['top']:.2f} ({dist:.1f}% below)")
        if nearest_supply:
            dist = (nearest_supply["bottom"] - close) / close * 100
            evidence.append(f"📍 Nearest supply: {nearest_supply['bottom']:.2f} ({dist:.1f}% above)")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 20)

        return self._out(
            direction=direction, confidence=confidence,
            score=np.clip(score, -1, 1), evidence=evidence,
            data={"demand_zones": len(demand_zones), "supply_zones": len(supply_zones)},
            reasoning=f"S/D: {'demand' if score > 0 else 'supply' if score < 0 else 'neutral'} zone"
        )
