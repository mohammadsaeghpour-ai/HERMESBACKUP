"""
Liquidity Agent — Where Retail Stops Are Clustered
=================================================
Detects liquidity pools, voids, magnetic levels, stop hunt targets.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class LiquidityAgent(BaseAgent):
    name = "Liquidity"
    weight = 1.5

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0
        close = df["close"].iloc[-1]

        # Find swing highs/lows
        highs = df["high"].rolling(5, center=True).max()
        lows = df["low"].rolling(5, center=True).min()

        swing_highs = df.loc[highs == df["high"], "high"].dropna().values[-5:]
        swing_lows = df.loc[lows == df["low"], "low"].dropna().values[-5:]

        # ── Buy-Side Liquidity (BSL) — above swing highs ──
        # Retail traders put stops ABOVE swing highs
        bsl_levels = [h for h in swing_highs if h > close]
        bsl_nearest = min(bsl_levels) if bsl_levels else None
        if bsl_nearest:
            dist = (bsl_nearest - close) / close * 100
            evidence.append(f"BSL (buy stops): {bsl_nearest:.2f} ({dist:.1f}% above)")
            evidence.append("   🎯 Institutions will push price HERE to trigger buy stops, then reverse")

            # Check if recently swept
            recent_high = df["high"].iloc[-3:].max()
            if recent_high > bsl_nearest and close < bsl_nearest:
                evidence.append("🔴 BSL SWEPT — stops already hunted above!")
                evidence.append("   🏦 Retail longs trapped, institutions absorbed their liquidity")
                score -= 0.3

        # ── Sell-Side Liquidity (SSL) — below swing lows ──
        ssl_levels = [l for l in swing_lows if l < close]
        ssl_nearest = max(ssl_levels) if ssl_levels else None
        if ssl_nearest:
            dist = (close - ssl_nearest) / close * 100
            evidence.append(f"SSL (sell stops): {ssl_nearest:.2f} ({dist:.1f}% below)")
            evidence.append("   🎯 Institutions will push price HERE to trigger sell stops, then reverse")

            recent_low = df["low"].iloc[-3:].min()
            if recent_low < ssl_nearest and close > ssl_nearest:
                evidence.append("🟢 SSL SWEPT — stops already hunted below!")
                evidence.append("   🏦 Retail shorts trapped, institutions bought their stops")
                score += 0.3

        # ── Liquidity Voids ──
        # Areas where price moved fast with little volume = no support
        for i in range(len(df)-10, len(df)-1):
            if i < 0: continue
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            atr_val = df["atr"].iloc[i] if "atr" in df.columns else 1
            if atr_val > 0 and body > atr_val * 2:
                evidence.append(f"⚡ Liquidity void near {df['close'].iloc[i]:.2f} — fast move, no support")

        # ── Magnetic Levels ──
        # Round numbers act as magnets
        round_levels = [round(close / 100) * 100, round(close / 50) * 50]
        for rl in round_levels:
            if abs(rl - close) / close * 100 < 1:
                evidence.append(f"🧲 Magnetic round number: {rl} ({abs(rl-close)/close*100:.2f}% away)")

        # ── Stop Hunt Probability ──
        if bsl_nearest and ssl_nearest:
            bsl_dist = (bsl_nearest - close) / close * 100
            ssl_dist = (close - ssl_nearest) / close * 100
            evidence.append(f"📊 Stop hunt targets: BSL {bsl_dist:.1f}% up | SSL {ssl_dist:.1f}% down")
            if ssl_dist < bsl_dist:
                evidence.append("🎯 More likely to hunt SSL first (closer)")
                score += 0.1
            else:
                evidence.append("🎯 More likely to hunt BSL first (closer)")
                score -= 0.1

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 20)

        return self._out(
            direction=direction, confidence=confidence,
            score=np.clip(score, -1, 1), evidence=evidence,
            data={"bsl": bsl_nearest, "ssl": ssl_nearest},
            reasoning=f"Liquidity: {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'}"
        )
