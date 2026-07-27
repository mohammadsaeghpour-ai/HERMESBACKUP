"""
Enhanced Market Structure Agent
================================
Multi-timeframe institutional market structure analysis.
Detects swing structure, internal vs external levels, market structure shifts,
ranges, institutional candles, and detailed BOS/CHoCH patterns.

Thinks like an institutional trader — explains what smart money is doing.
"""
from hqip.agents.base import BaseAgent
import numpy as np


class MarketStructureAgent(BaseAgent):
    name = "MarketStructure"
    weight = 1.4

    # ------------------------------------------------------------------
    # SWING DETECTION — multiple lookbacks for internal & external structure
    # ------------------------------------------------------------------
    def _find_swings(self, highs, lows, lookback=5):
        """Find swing highs and lows with given lookback."""
        n = len(highs)
        swing_highs = []  # list of (index, price)
        swing_lows = []

        for i in range(lookback, n - lookback):
            window_h = highs[i - lookback: i + lookback + 1]
            window_l = lows[i - lookback: i + lookback + 1]

            if highs[i] == np.max(window_h):
                swing_highs.append((i, highs[i]))
            if lows[i] == np.min(window_l):
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def _find_swings_multi(self, highs, lows):
        """Find swings at multiple lookback periods: 3 (internal), 5 (medium), 8 (external)."""
        results = {}
        for lb in [3, 5, 8]:
            sh, sl = self._find_swings(highs, lows, lookback=lb)
            results[lb] = (sh, sl)
        return results

    # ------------------------------------------------------------------
    # STRUCTURE ANALYSIS
    # ------------------------------------------------------------------
    def _analyze_structure(self, swing_highs, swing_lows, close):
        """Determine HH/HL or LH/LL from swing points."""
        result = {
            "pattern": "UNKNOWN",
            "hh_hl": False,
            "lh_ll": False,
            "hh_ll": False,
            "lh_hl": False,
            "last_high": None,
            "last_low": None,
        }

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result

        # Get last two swings
        h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
        l1, l2 = swing_lows[-2][1], swing_lows[-1][1]

        hh = h2 > h1
        hl = l2 > l1
        ll = l2 < l1
        lh = h2 < h1

        result["last_high"] = h2
        result["last_low"] = l2

        if hh and hl:
            result["hh_hl"] = True
            result["pattern"] = "HH_HL"
        elif ll and lh:
            result["lh_ll"] = True
            result["pattern"] = "LH_LL"
        elif hh and ll:
            result["hh_ll"] = True
            result["pattern"] = "HH_LL"  # expansion
        elif lh and hl:
            result["lh_hl"] = True
            result["pattern"] = "LH_HL"  # contraction / squeeze

        return result

    def _detect_structure_shift(self, swings_highs, swings_lows, close, lookback_name=""):
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
        signals = []

        if len(swings_highs) < 2 or len(swings_lows) < 2:
            return signals

        last_sh = swings_highs[-1][1]
        prev_sh = swings_highs[-2][1]
        last_sl = swings_lows[-1][1]
        prev_sl = swings_lows[-2][1]

        # Determine current structure
        bullish_structure = last_sh > prev_sh and last_sl > prev_sl
        bearish_structure = last_sh < prev_sh and last_sl < prev_sl

        # BOS = break in the direction of the trend
        if bullish_structure and close > last_sh:
            signals.append({
                "type": "BOS",
                "direction": "BULLISH",
                "level": last_sh,
                "label": f"[{lookback_name}] Break of Structure BULLISH — price closed above {last_sh:.2f}",
                "smart_money": "Institutions confirmed the uptrend by pushing price above the last swing high. "
                               "Smart money is adding to longs or defending the trend."
            })

        if bearish_structure and close < last_sl:
            signals.append({
                "type": "BOS",
                "direction": "BEARISH",
                "level": last_sl,
                "label": f"[{lookback_name}] Break of Structure BEARISH — price closed below {last_sl:.2f}",
                "smart_money": "Institutions confirmed the downtrend by pushing price below the last swing low. "
                               "Smart money is adding to shorts or defending the trend."
            })

        # CHoCH = reversal signal (break against the trend)
        if bullish_structure and close < last_sl:
            signals.append({
                "type": "CHoCH",
                "direction": "BEARISH",
                "level": last_sl,
                "label": f"[{lookback_name}] Change of Character BEARISH — price broke below {last_sl:.2f} in uptrend",
                "smart_money": "Institutions have shifted from buying to selling. The uptrend structure is broken. "
                               "Smart money is exiting longs and may be initiating shorts."
            })

        if bearish_structure and close > last_sh:
            signals.append({
                "type": "CHoCH",
                "direction": "BULLISH",
                "level": last_sh,
                "label": f"[{lookback_name}] Change of Character BULLISH — price broke above {last_sh:.2f} in downtrend",
                "smart_money": "Institutions have shifted from selling to buying. The downtrend structure is broken. "
                               "Smart money is exiting shorts and may be initiating longs."
            })

        return signals

    # ------------------------------------------------------------------
    # INSTITUTIONAL CANDLE DETECTION
    # ------------------------------------------------------------------
    def _detect_institutional_candles(self, o, h, l, c, v, avg_vol, avg_body):
        """Detect candles that show institutional footprint."""
        candles = []
        n = len(o)

        for i in range(max(0, n - 5), n):
            rng = h[i] - l[i]
            if rng == 0:
                continue

            body = abs(c[i] - o[i])
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            body_ratio = body / max(rng, 1e-10)
            vol_ratio = v[i] / max(avg_vol, 1e-10)
            body_ratio_vs_avg = body / max(avg_body, 1e-10)

            # Institutional candle: large body, small wicks, above-average volume
            is_large_body = body_ratio > 0.70 and body_ratio_vs_avg > 1.3
            has_small_wicks = (upper_wick + lower_wick) < rng * 0.30
            has_vol = vol_ratio > 1.2

            if is_large_body and has_small_wicks and has_vol:
                direction = "BULLISH" if c[i] > o[i] else "BEARISH"
                pct = (body / max(o[i], 1e-10)) * 100
                candles.append({
                    "index": i,
                    "direction": direction,
                    "body_pct": round(pct, 2),
                    "vol_ratio": round(vol_ratio, 1),
                    "close": round(c[i], 2),
                })

        return candles

    # ------------------------------------------------------------------
    # RANGE DETECTION
    # ------------------------------------------------------------------
    def _detect_range(self, h, l, c, lookback=20):
        """Detect if price is in a ranging market."""
        recent_h = h[-lookback:]
        recent_l = l[-lookback:]
        recent_c = c[-lookback:]

        range_high = np.max(recent_h)
        range_low = np.min(recent_l)
        range_size = range_high - range_low

        if range_size == 0:
            return {"in_range": False, "range_pct": 0}

        avg_range = np.mean(recent_h - recent_l)
        # Ranging if individual candle ranges are small relative to total range
        range_ratio = range_size / max(avg_range, 1e-10)
        in_range = range_ratio > 5  # total range is 5x+ average candle range

        # Where in the range is current price?
        position = (c[-1] - range_low) / max(range_size, 1e-10)

        return {
            "in_range": in_range,
            "range_high": round(range_high, 2),
            "range_low": round(range_low, 2),
            "range_pct": round(range_size / max(range_low, 1e-10) * 100, 2),
            "range_ratio": round(range_ratio, 1),
            "position_in_range": round(position, 2),  # 0=bottom, 1=top
        }

    # ------------------------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------------------------
    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data for market structure analysis"],
                reasoning="Need at least 30 candles"
            )

        evidence = []
        score = 0.0
        data = {}

        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        v = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(df))

        close = c[-1]
        avg_vol = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
        avg_vol = max(avg_vol, 1e-10)
        avg_body = np.mean(np.abs(c - o))
        avg_body = max(avg_body, 1e-10)

        # ---- Multi-timeframe swing detection ----
        swings_multi = self._find_swings_multi(h, l)

        # ---- Structure analysis per timeframe ----
        structure_results = {}
        for lb, (sh, sl) in swings_multi.items():
            name = {3: "Internal", 5: "Medium", 8: "External"}[lb]
            struct = self._analyze_structure(sh, sl, close)
            structure_results[name] = struct

            # Print structure
            if struct["pattern"] == "HH_HL":
                evidence.append(f"🟢 [{name}] Higher Highs + Higher Lows → bullish structure")
                score += 0.15
            elif struct["pattern"] == "LH_LL":
                evidence.append(f"🔴 [{name}] Lower Highs + Lower Lows → bearish structure")
                score -= 0.15
            elif struct["pattern"] == "HH_LL":
                evidence.append(f"⚠️ [{name}] Higher High + Lower Low → expanding range (volatility)")
            elif struct["pattern"] == "LH_HL":
                evidence.append(f"⚠️ [{name}] Lower High + Higher Low → contracting range (squeeze)")
                score += 0.05  # slight bullish bias — squeezes often break up

        # ---- Structure shift detection (BOS / CHoCH) ----
        all_shifts = []
        for lb, (sh, sl) in swings_multi.items():
            name = {3: "Internal", 5: "Medium", 8: "External"}[lb]
            shifts = self._detect_structure_shift(sh, sl, close, lookback_name=name)
            all_shifts.extend(shifts)

        for shift in all_shifts:
            label = shift["label"]
            sm = shift["smart_money"]
            evidence.append(f"📊 {label}")
            evidence.append(f"   💡 {sm}")

            if shift["type"] == "BOS":
                if shift["direction"] == "BULLISH":
                    score += 0.20
                else:
                    score -= 0.20
            elif shift["type"] == "CHoCH":
                # CHoCH is a reversal — stronger signal
                if shift["direction"] == "BULLISH":
                    score += 0.30
                else:
                    score -= 0.30

        data["structure_shifts"] = [
            {"type": s["type"], "direction": s["direction"], "level": round(s["level"], 2)}
            for s in all_shifts
        ]

        # ---- Key support / resistance from external swings ----
        ext_highs, ext_lows = swings_multi[8]
        if ext_highs:
            nearest_res = min([h[1] for h in ext_highs if h[1] > close], default=None)
            if nearest_res is not None:
                res_dist = (nearest_res - close) / close * 100
                evidence.append(f"📍 Key resistance (external): {nearest_res:.2f} ({res_dist:.2f}% above)")
                if res_dist < 0.5:
                    evidence.append("  ⚠️ At resistance — watch for rejection or decisive break")
                    score -= 0.10

        if ext_lows:
            nearest_sup = max([lo[1] for lo in ext_lows if lo[1] < close], default=None)
            if nearest_sup is not None:
                sup_dist = (close - nearest_sup) / close * 100
                evidence.append(f"📍 Key support (external): {nearest_sup:.2f} ({sup_dist:.2f}% below)")
                if sup_dist < 0.5:
                    evidence.append("  ⚠️ At support — watch for bounce or break")
                    score += 0.10

        # ---- Range detection ----
        range_info = self._detect_range(h, l, c, lookback=20)
        if range_info["in_range"]:
            pct = range_info["position_in_range"]
            evidence.append(
                f"📦 RANGING MARKET — {range_info['range_pct']:.1f}% range "
                f"({range_info['range_low']:.2f} - {range_info['range_high']:.2f})"
            )
            evidence.append(f"  Price at {pct*100:.0f}% of range")

            if pct > 0.75:
                evidence.append("  Near range top — institutions may distribute here")
                score -= 0.10
            elif pct < 0.25:
                evidence.append("  Near range bottom — institutions may accumulate here")
                score += 0.10
        else:
            evidence.append("📈 Trending market — no defined range")

        data["range"] = range_info

        # ---- Institutional candle detection ----
        inst_candles = self._detect_institutional_candles(o, h, l, c, v, avg_vol, avg_body)
        if inst_candles:
            bullish_inst = [ic for ic in inst_candles if ic["direction"] == "BULLISH"]
            bearish_inst = [ic for ic in inst_candles if ic["direction"] == "BEARISH"]

            evidence.append(
                f"🏛️ Institutional candles: {len(bullish_inst)} bullish, {len(bearish_inst)} bearish (last 5 bars)"
            )

            for ic in inst_candles[-3:]:  # show last 3
                emoji = "🟢" if ic["direction"] == "BULLISH" else "🔴"
                evidence.append(
                    f"  {emoji} {ic['direction']} — body {ic['body_pct']:.1f}%, "
                    f"vol {ic['vol_ratio']}x avg, close {ic['close']}"
                )

            if len(bullish_inst) > len(bearish_inst):
                evidence.append("  💡 More institutional buying candles — smart money accumulating")
                score += 0.15
            elif len(bearish_inst) > len(bullish_inst):
                evidence.append("  💡 More institutional selling candles — smart money distributing")
                score -= 0.15

        data["institutional_candles"] = inst_candles

        # ---- Summary: what is smart money doing? ----
        smart_money_actions = []
        if score > 0.3:
            smart_money_actions.append("Accumulating longs — pushing price up")
        elif score > 0.1:
            smart_money_actions.append("Defending support / adding to longs")
        elif score < -0.3:
            smart_money_actions.append("Distributing into strength — pushing price down")
        elif score < -0.1:
            smart_money_actions.append("Defending resistance / adding to shorts")
        else:
            smart_money_actions.append("No clear directional bias — re-evaluating positioning")

        # Count total swing points
        total_sh = sum(len(swings_multi[lb][0]) for lb in [3, 5, 8])
        total_sl = sum(len(swings_multi[lb][1]) for lb in [3, 5, 8])

        evidence.append(f"📊 Swing structure: {total_sh} swing highs, {total_sl} swing lows across 3 timeframes")
        evidence.append(f"🏛️ Smart money assessment: {smart_money_actions[0]}")

        # Direction / confidence
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 70 + len(all_shifts) * 5 + 15)

        reasoning = (
            f"Multi-timeframe structure analysis: "
            f"Internal={'bullish' if structure_results.get('Internal',{}).get('hh_hl') else 'bearish' if structure_results.get('Internal',{}).get('lh_ll') else 'mixed'}, "
            f"External={'bullish' if structure_results.get('External',{}).get('hh_hl') else 'bearish' if structure_results.get('External',{}).get('lh_ll') else 'mixed'}. "
            f"Structure shifts: {len([s for s in all_shifts if s['type']=='BOS'])} BOS, "
            f"{len([s for s in all_shifts if s['type']=='CHoCH'])} CHoCH. "
            f"Score {score:+.2f} → {direction}."
        )

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=float(np.clip(score, -1.0, 1.0)),
            evidence=evidence,
            data=data,
            reasoning=reasoning,
        )
