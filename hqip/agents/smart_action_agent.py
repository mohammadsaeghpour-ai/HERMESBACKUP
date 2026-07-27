"""
Smart Action Agent
==================
Detects institutional smart money actions by analyzing price action patterns
that reveal what large players are doing behind the scenes.

Patterns detected:
- STOP HUNTS: Price wicks above/below key levels then reverses quickly
- MANIPULATION: False breakouts trapping retail traders
- ACCUMULATION: Narrow range + high volume = institutions buying
- DISTRIBUTION: Narrow range + high volume at highs = institutions selling
- EXHAUSTION: Strong move with declining volume = running out of steam
- INITIATION: Sudden volume spike + large body = new move beginning
"""
from hqip.agents.base import BaseAgent
import numpy as np


class SmartActionAgent(BaseAgent):
    name = "SmartAction"
    weight = 1.5

    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 20:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data for smart action detection"],
                reasoning="Need at least 20 candles"
            )

        evidence = []
        score = 0.0
        data = {}
        detected_actions = []

        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        v = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros(len(df))

        body = np.abs(c - o)
        full_range = h - l
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        mid_price = (h + l) / 2.0

        # Average volume over the lookback
        vol_lookback = min(20, len(v) - 1)
        avg_vol = np.mean(v[-vol_lookback - 1:-1]) if vol_lookback > 0 else 1.0
        avg_vol = max(avg_vol, 1e-10)

        # ---------------------------------------------------------------
        # 1. STOP HUNT DETECTION
        #    Price wicks far beyond key level but closes back inside.
        #    Institutions push price past obvious stops to fill their orders.
        # ---------------------------------------------------------------
        stop_hunt_count = 0
        for i in range(max(1, len(df) - 10), len(df)):
            rng = full_range[i]
            if rng == 0:
                continue

            # Upper stop hunt: long upper wick, close near low of candle
            if upper_wick[i] > rng * 0.65 and lower_wick[i] < rng * 0.20:
                close_near_low = (c[i] - l[i]) / max(rng, 1e-10) < 0.35
                if close_near_low:
                    stop_hunt_count += 1

            # Lower stop hunt: long lower wick, close near high of candle
            if lower_wick[i] > rng * 0.65 and upper_wick[i] < rng * 0.20:
                close_near_high = (h[i] - c[i]) / max(rng, 1e-10) < 0.35
                if close_near_high:
                    stop_hunt_count += 1

        if stop_hunt_count >= 2:
            evidence.append(f"🔴 STOP HUNTS detected ({stop_hunt_count}x in last 10 candles)")
            evidence.append("  Institutions are hunting stop losses by pushing price past key levels,")
            evidence.append("  then reversing to fill their real orders at better prices")
            score -= 0.25  # Bearish — traps weak hands, often precedes drop
            detected_actions.append("STOP_HUNT")

        elif stop_hunt_count == 1:
            evidence.append("⚠️ Single stop hunt pattern detected")
            score -= 0.10

        # ---------------------------------------------------------------
        # 2. MANIPULATION / FALSE BREAKOUT DETECTION
        #    Candle breaks a recent high/low but immediately reverses.
        #    Retail sees breakout and enters, institutions trap them.
        # ---------------------------------------------------------------
        recent_lookback = min(15, len(df) - 1)
        prev_high = np.max(h[-recent_lookback - 1:-1])
        prev_low = np.min(l[-recent_lookback - 1:-1])
        last_h, last_l, last_c, last_o = h[-1], l[-1], c[-1], o[-1]

        false_breakout_up = False
        false_breakout_down = False

        # False breakout UP: high exceeds recent range but closes below
        if last_h > prev_high and last_c < prev_high and last_c < last_o:
            false_breakout_up = True
            score -= 0.30
            evidence.append("🔴 FALSE BREAKOUT UP — Price broke above recent high but closed red and below the level")
            evidence.append("  Retail trapped in longs. Institutions distributed into breakout buyers.")
            detected_actions.append("MANIPULATION_BULL_TRAP")

        # False breakout DOWN: low exceeds recent range but closes above
        if last_l < prev_low and last_c > prev_low and last_c > last_o:
            false_breakout_down = True
            score += 0.30
            evidence.append("🟢 FALSE BREAKOUT DOWN — Price broke below recent low but closed green and above the level")
            evidence.append("  Retail trapped in shorts. Institutions accumulated from panic sellers.")
            detected_actions.append("MANIPULATION_BEAR_TRAP")

        # ---------------------------------------------------------------
        # 3. ACCUMULATION DETECTION
        #    Narrow range candles with above-average volume.
        #    Institutions quietly build positions without moving price.
        # ---------------------------------------------------------------
        acc_window = min(5, len(df) - 1)
        acc_ranges = full_range[-acc_window:]
        acc_bodies = body[-acc_window:]
        acc_vols = v[-acc_window:]

        avg_range = np.mean(full_range[-20:]) if len(full_range) >= 20 else np.mean(full_range)
        avg_range = max(avg_range, 1e-10)

        # Narrow range = candles with range < 60% of average
        narrow_count = np.sum(acc_ranges < avg_range * 0.6)
        high_vol_count = np.sum(acc_vols > avg_vol * 1.1)

        # Accumulation at lower prices (near recent lows)
        near_lows = last_l <= prev_low * 1.02  # within 2% of recent lows

        if narrow_count >= 3 and high_vol_count >= 2:
            if near_lows:
                evidence.append("🟢 ACCUMULATION detected — Narrow range + high volume near lows")
                evidence.append("  Institutions are quietly building long positions without moving price.")
                evidence.append("  They want to fill large orders without alerting the market.")
                score += 0.35
                detected_actions.append("ACCUMULATION")
            else:
                evidence.append("🟡 CONSOLIDATION with elevated volume — potential accumulation zone")
                score += 0.10
                detected_actions.append("CONSOLIDATION_ACCUM")

        # ---------------------------------------------------------------
        # 4. DISTRIBUTION DETECTION
        #    Narrow range candles with above-average volume at HIGHS.
        #    Institutions quietly sell into strength.
        # ---------------------------------------------------------------
        near_highs = last_h >= prev_high * 0.98  # within 2% of recent highs

        if narrow_count >= 3 and high_vol_count >= 2 and near_highs:
            evidence.append("🔴 DISTRIBUTION detected — Narrow range + high volume near highs")
            evidence.append("  Institutions are quietly selling into strength.")
            evidence.append("  They distribute their positions to buyers who think the rally will continue.")
            score -= 0.35
            detected_actions.append("DISTRIBUTION")

        # ---------------------------------------------------------------
        # 5. EXHAUSTION DETECTION
        #    Strong directional move with declining volume.
        #    The move is running out of steam — buyers/sellers exhausted.
        # ---------------------------------------------------------------
        recent_close = c[-1]
        lookback_close = c[-min(5, len(c))]
        move_pct = (recent_close - lookback_close) / max(abs(lookback_close), 1e-10)

        vol_trend = np.polyfit(range(acc_window), acc_vols, 1)[0] if acc_window >= 2 else 0
        vol_declining = vol_trend < -avg_vol * 0.05

        if abs(move_pct) > 0.02 and vol_declining:
            if move_pct > 0:
                evidence.append("🔴 EXHAUSTION — Strong rally with declining volume")
                evidence.append(f"  Price moved {move_pct*100:.1f}% but volume is fading.")
                evidence.append(f"  Buyers are running out of steam. Smart money may have already exited.")
                score -= 0.20
                detected_actions.append("EXHAUSTION_BULL")
            else:
                evidence.append("🟢 EXHAUSTION — Strong selloff with declining volume")
                evidence.append(f"  Price dropped {move_pct*100:.1f}% but volume is fading.")
                evidence.append(f"  Sellers are running out of steam. Smart money may be absorbing.")
                score += 0.20
                detected_actions.append("EXHAUSTION_BEAR")

        # ---------------------------------------------------------------
        # 6. INITIATION DETECTION
        #    Sudden volume spike + large body candle = new move beginning.
        #    Institutions are entering aggressively in one direction.
        # ---------------------------------------------------------------
        last_vol = v[-1]
        last_body = body[-1]
        avg_body = np.mean(body[-20:]) if len(body) >= 20 else np.mean(body)
        avg_body = max(avg_body, 1e-10)

        vol_spike_ratio = last_vol / max(avg_vol, 1e-10)
        body_spike_ratio = last_body / max(avg_body, 1e-10)

        if vol_spike_ratio > 2.0 and body_spike_ratio > 1.5:
            # Green initiation candle
            if last_c > last_o:
                evidence.append(f"🟢 INITIATION detected — Volume spike {vol_spike_ratio:.1f}x avg + large green body")
                evidence.append("  Institutions are aggressively entering long positions.")
                evidence.append("  This could be the start of a new directional move.")
                score += 0.40
                detected_actions.append("INITIATION_BULL")
            else:
                evidence.append(f"🔴 INITIATION detected — Volume spike {vol_spike_ratio:.1f}x avg + large red body")
                evidence.append("  Institutions are aggressively entering short positions.")
                evidence.append("  This could be the start of a new directional move down.")
                score -= 0.40
                detected_actions.append("INITIATION_BEAR")
        elif vol_spike_ratio > 1.5:
            evidence.append(f"⚠️ Elevated volume ({vol_spike_ratio:.1f}x avg) — watching for initiation")

        # ---------------------------------------------------------------
        # COMPOSITE SCORING & SUMMARY
        # ---------------------------------------------------------------
        data = {
            "stop_hunts": stop_hunt_count,
            "false_breakout_up": false_breakout_up,
            "false_breakout_down": false_breakout_down,
            "narrow_range_candles": int(narrow_count),
            "volume_spike_ratio": round(vol_spike_ratio, 2),
            "body_spike_ratio": round(body_spike_ratio, 2),
            "detected_actions": detected_actions,
            "move_pct": round(move_pct * 100, 2),
        }

        # Determine overall institutional posture
        if score > 0.2:
            posture = "INSTITUTIONAL ACCUMULATION / BUYING"
        elif score < -0.2:
            posture = "INSTITUTIONAL DISTRIBUTION / SELLING"
        else:
            posture = "NO CLEAR INSTITUTIONAL ACTION"

        # Confidence based on number of confirming signals
        signal_count = len(detected_actions)
        base_confidence = min(70, signal_count * 20 + 10)
        confidence = min(100, base_confidence + abs(score) * 30)

        # Build reasoning
        if detected_actions:
            actions_str = ", ".join(detected_actions)
            reasoning = (
                f"Smart money actions detected: [{actions_str}]. "
                f"Overall posture: {posture}. "
                f"Score {score:+.2f} based on {signal_count} confirming pattern(s)."
            )
        else:
            reasoning = "No significant institutional action patterns detected in recent candles."

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=float(np.clip(score, -1.0, 1.0)),
            evidence=evidence,
            data=data,
            reasoning=reasoning,
        )
