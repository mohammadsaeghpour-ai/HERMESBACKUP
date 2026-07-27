"""
SMC Agent — Smart Money Concepts
================================
Detects institutional patterns: Order Blocks, FVG, Liquidity, BOS, CHoCH.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class SMCAgent(BaseAgent):
    name = "SMC"
    weight = 1.6

    def _find_swings(self, high, low, lookback=5):
        sh, sl = [], []
        for i in range(lookback, len(high) - lookback):
            if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
                sh.append((i, high.iloc[i]))
            if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
                sl.append((i, low.iloc[i]))
        return sh, sl

    def _find_order_blocks(self, df):
        """Find last opposing candle before strong move = institutional footprint"""
        obs = []
        for i in range(2, min(len(df), 50)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            avg_body = df["body"].abs().rolling(20).mean().iloc[i]
            if avg_body == 0: continue
            if body > avg_body * 2:
                prev = i - 1
                prev_body = df["close"].iloc[prev] - df["open"].iloc[prev]
                curr_dir = 1 if df["close"].iloc[i] > df["open"].iloc[i] else -1
                if (curr_dir == 1 and prev_body < 0) or (curr_dir == -1 and prev_body > 0):
                    obs.append({
                        "index": prev, "type": "bullish" if curr_dir == 1 else "bearish",
                        "high": df["high"].iloc[prev], "low": df["low"].iloc[prev],
                        "strength": body / avg_body
                    })
        return obs

    def _find_fvgs(self, df):
        """Fair Value Gaps: gap between candle 1 high and candle 3 low"""
        fvgs = []
        for i in range(2, min(len(df), 30)):
            # Bullish FVG
            if df["low"].iloc[i] > df["high"].iloc[i-2]:
                fvgs.append({"index": i-1, "type": "bullish",
                    "top": df["low"].iloc[i], "bottom": df["high"].iloc[i-2],
                    "mid": (df["low"].iloc[i] + df["high"].iloc[i-2]) / 2})
            # Bearish FVG
            if df["high"].iloc[i] < df["low"].iloc[i-2]:
                fvgs.append({"index": i-1, "type": "bearish",
                    "top": df["low"].iloc[i-2], "bottom": df["high"].iloc[i],
                    "mid": (df["low"].iloc[i-2] + df["high"].iloc[i-2]) / 2})
        return fvgs

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["Insufficient data"])

        evidence = []
        score = 0.0
        close = df["close"].iloc[-1]

        # ── Swing Detection ──
        sh, sl = self._find_swings(df["high"], df["low"], 5)
        evidence.append(f"Swing highs: {len(sh)} | Swing lows: {len(sl)}")

        # ── Liquidity Pools ──
        bsl = [h[1] for h in sh[-3:]] if sh else []  # Buy-side liquidity above highs
        ssl = [l[1] for l in sl[-3:]] if sl else []   # Sell-side liquidity below lows
        evidence.append(f"BSL (buy stops above): {[f'{x:.2f}' for x in bsl]}")
        evidence.append(f"SSL (sell stops below): {[f'{x:.2f}' for x in ssl]}")

        # ── Liquidity Sweep Detection ──
        if len(df) > 2 and sh:
            prev_high = sh[-1][1] if sh else 0
            if df["high"].iloc[-2] > prev_high and close < prev_high:
                evidence.append(f"🔴 LIQUIDITY SWEEP ABOVE: Price took {prev_high:.2f} then closed below — STOP HUNT")
                evidence.append("   🏦 Institutions absorbed buy-side liquidity, likely reversing down")
                score -= 0.4

        if len(df) > 2 and sl:
            prev_low = sl[-1][1] if sl else 999999
            if df["low"].iloc[-2] < prev_low and close > prev_low:
                evidence.append(f"🟢 LIQUIDITY SWEEP BELOW: Price took {prev_low:.2f} then closed above — STOP HUNT")
                evidence.append("   🏦 Institutions absorbed sell-side liquidity, likely reversing up")
                score += 0.4

        # ── Order Blocks ──
        obs = self._find_order_blocks(df)
        bullish_obs = [o for o in obs if o["type"] == "bullish" and o["low"] <= close <= o["high"] * 1.01]
        bearish_obs = [o for o in obs if o["type"] == "bearish" and o["low"] * 0.99 <= close <= o["high"]]

        if bullish_obs:
            evidence.append(f"🟢 Price INSIDE Bullish Order Block ({bullish_obs[0]['low']:.2f}-{bullish_obs[0]['high']:.2f})")
            evidence.append("   🏦 This is where institutions placed large buy orders — high probability bounce zone")
            score += 0.3
        if bearish_obs:
            evidence.append(f"🔴 Price INSIDE Bearish Order Block ({bearish_obs[0]['low']:.2f}-{bearish_obs[0]['high']:.2f})")
            evidence.append("   🏦 This is where institutions placed large sell orders — high probability rejection zone")
            score -= 0.3

        # ── Fair Value Gaps ──
        fvgs = self._find_fvgs(df)
        bullish_fvgs = [f for f in fvgs if f["type"] == "bullish" and f["bottom"] <= close <= f["top"] * 1.02]
        bearish_fvgs = [f for f in fvgs if f["type"] == "bearish" and f["bottom"] * 0.98 <= close <= f["top"]]

        if bullish_fvgs:
            evidence.append(f"🟢 Price in Bullish FVG zone ({bullish_fvgs[0]['bottom']:.2f}-{bullish_fvgs[0]['top']:.2f})")
            evidence.append("   🏦 Market imbalance — price likely to fill this gap upward")
            score += 0.2
        if bearish_fvgs:
            evidence.append(f"🔴 Price in Bearish FVG zone ({bearish_fvgs[0]['bottom']:.2f}-{bearish_fvgs[0]['top']:.2f})")
            evidence.append("   🏦 Market imbalance — price likely to fill this gap downward")
            score -= 0.2

        # ── Premium/Discount ──
        if sh and sl:
            range_high = max(h[1] for h in sh[-5:])
            range_low = min(l[1] for l in sl[-5:])
            range_size = range_high - range_low
            if range_size > 0:
                position = (close - range_low) / range_size
                if position < 0.5:
                    evidence.append(f"🟢 DISCOUNT zone ({position:.0%}) — below equilibrium")
                    evidence.append("   🏦 Institutions prefer buying below fair value")
                    score += 0.15
                else:
                    evidence.append(f"🔴 PREMIUM zone ({position:.0%}) — above equilibrium")
                    evidence.append("   🏦 Institutions prefer selling above fair value")
                    score -= 0.15

                # OTE zone
                ote_low = range_high - range_size * 0.786
                ote_high = range_high - range_size * 0.618
                if ote_low <= close <= ote_high:
                    evidence.append(f"🟢 OTE ZONE ({ote_low:.2f}-{ote_high:.2f}) — Optimal Trade Entry")
                    evidence.append("   🏦 The 62-78% retracement — where institutions scale in")

        # ── BOS / CHoCH ──
        if len(sh) >= 2:
            if close > sh[-2][1]:
                evidence.append(f"🟢 Break of Structure ABOVE {sh[-2][1]:.2f} — bullish continuation")
                score += 0.25
        if len(sl) >= 2:
            if close < sl[-2][1]:
                evidence.append(f"🔴 Break of Structure BELOW {sl[-2][1]:.2f} — bearish continuation")
                score -= 0.25

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 15)

        return self._out(
            direction=direction, confidence=confidence,
            score=np.clip(score, -1, 1), evidence=evidence,
            data={"swing_highs": len(sh), "swing_lows": len(sl), "fvgs": len(fvgs), "obs": len(obs)},
            reasoning=f"SMC: {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} structure"
        )
