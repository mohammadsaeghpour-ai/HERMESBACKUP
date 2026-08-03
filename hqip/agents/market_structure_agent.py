"""
Market Structure Agent — Enhanced Structure Analysis
=====================================================
Multi-timeframe institutional market structure analysis:

- Swing highs/lows detection (fractal method, multiple lookbacks)
- Internal vs external structure
- Break of Structure (BOS) detection
- Change of Character (CHoCH) detection
- Market structure shift momentum (accelerating/decelerating)
- Range detection with boundaries
- Institutional candle detection (large body + volume)
- Multi-timeframe structure alignment

Weight: 1.4
"""
from hqip.agents.base import BaseAgent
import numpy as np


class MarketStructureAgent(BaseAgent):
    name = "MarketStructure"
    weight = 1.4

    # ── SWING DETECTION ────────────────────────────────────────
    def _find_swings(self, highs, lows, lookback=5):
        """Fractal swing highs/lows with given lookback."""
        n = len(highs)
        swing_highs = []
        swing_lows = []

        for i in range(lookback, n - lookback):
            window_h = highs[i - lookback: i + lookback + 1]
            window_l = lows[i - lookback: i + lookback + 1]

            if highs[i] == np.max(window_h):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == np.min(window_l):
                swing_lows.append((i, float(lows[i])))

        return swing_highs, swing_lows

    def _find_swings_multi(self, highs, lows):
        """Find swings at 3 lookback periods: 3 (internal), 5 (medium), 8 (external)."""
        return {lb: self._find_swings(highs, lows, lookback=lb) for lb in [3, 5, 8]}

    # ── STRUCTURE ANALYSIS ─────────────────────────────────────
    def _analyze_structure(self, swing_highs, swing_lows):
        """Determine HH/HL, LH/LL, expansion, or contraction from swings."""
        result = {"pattern": "UNKNOWN", "hh_hl": False, "lh_ll": False,
                   "hh_ll": False, "lh_hl": False,
                   "last_high": None, "last_low": None}

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result

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
            result["pattern"] = "HH_LL"
        elif lh and hl:
            result["lh_hl"] = True
            result["pattern"] = "LH_HL"

        return result

    # ── BOS / CHoCH DETECTION ──────────────────────────────────
    def _detect_structure_shifts(self, swings_highs, swings_lows, close, lookback_name=""):
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
        signals = []

        if len(swings_highs) < 2 or len(swings_lows) < 2:
            return signals

        last_sh = swings_highs[-1][1]
        prev_sh = swings_highs[-2][1]
        last_sl = swings_lows[-1][1]
        prev_sl = swings_lows[-2][1]

        bullish_structure = last_sh > prev_sh and last_sl > prev_sl
        bearish_structure = last_sh < prev_sh and last_sl < prev_sl

        # BOS — break in the direction of the trend
        if bullish_structure and close > last_sh:
            signals.append({
                "type": "BOS", "direction": "BULLISH", "level": last_sh,
                "label": f"[{lookback_name}] BOS BULLISH — price above {last_sh:.2f}",
                "smart_money": "Institutions confirmed uptrend by pushing above last swing high.",
            })
        if bearish_structure and close < last_sl:
            signals.append({
                "type": "BOS", "direction": "BEARISH", "level": last_sl,
                "label": f"[{lookback_name}] BOS BEARISH — price below {last_sl:.2f}",
                "smart_money": "Institutions confirmed downtrend by pushing below last swing low.",
            })

        # CHoCH — break against the trend (reversal)
        if bullish_structure and close < last_sl:
            signals.append({
                "type": "CHoCH", "direction": "BEARISH", "level": last_sl,
                "label": f"[{lookback_name}] CHoCH BEARISH — broke {last_sl:.2f} in uptrend",
                "smart_money": "Institutions shifted from buying to selling. Uptrend structure broken.",
            })
        if bearish_structure and close > last_sh:
            signals.append({
                "type": "CHoCH", "direction": "BULLISH", "level": last_sh,
                "label": f"[{lookback_name}] CHoCH BULLISH — broke {last_sh:.2f} in downtrend",
                "smart_money": "Institutions shifted from selling to buying. Downtrend structure broken.",
            })

        return signals

    # ── STRUCTURE MOMENTUM ─────────────────────────────────────
    def _compute_momentum(self, swings_highs, swings_lows):
        """
        Detect if market structure shift is accelerating or decelerating.
        Compare spacing and magnitude of recent swings.
        Returns: 'accelerating', 'decelerating', or 'stable'.
        """
        if len(swings_highs) < 3 or len(swings_lows) < 3:
            return "stable"

        # High swing spacing
        h_spacing = [
            swings_highs[i][0] - swings_highs[i - 1][0]
            for i in range(1, len(swings_highs))
        ]
        l_spacing = [
            swings_lows[i][0] - swings_lows[i - 1][0]
            for i in range(1, len(swings_lows))
        ]

        if len(h_spacing) < 2 or len(l_spacing) < 2:
            return "stable"

        # If recent swings are closer together = accelerating
        h_accel = h_spacing[-1] < np.mean(h_spacing[:-1]) * 0.7
        l_accel = l_spacing[-1] < np.mean(l_spacing[:-1]) * 0.7

        if h_accel and l_accel:
            return "accelerating"
        elif not h_accel and not l_accel:
            return "decelerating"
        return "stable"

    # ── INSTITUTIONAL CANDLE DETECTION ─────────────────────────
    def _detect_institutional_candles(self, o, h, l, c, v, avg_vol, avg_body):
        """Detect candles showing institutional footprint (large body, small wicks, high vol)."""
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
                    "vol_ratio": round(float(vol_ratio), 1),
                    "close": round(float(c[i]), 2),
                })

        return candles

    # ── RANGE DETECTION ────────────────────────────────────────
    def _detect_range(self, h, l, c, lookback=20):
        """Detect if price is in a ranging market with boundaries."""
        recent_h = h[-lookback:]
        recent_l = l[-lookback:]

        range_high = float(np.max(recent_h))
        range_low = float(np.min(recent_l))
        range_size = range_high - range_low

        if range_size <= 0:
            return {"in_range": False, "range_pct": 0}

        avg_candle_range = float(np.mean(recent_h - recent_l))
        range_ratio = range_size / max(avg_candle_range, 1e-10)
        in_range = range_ratio > 5

        position = (c[-1] - range_low) / max(range_size, 1e-10)

        return {
            "in_range": in_range,
            "range_high": round(range_high, 2),
            "range_low": round(range_low, 2),
            "range_pct": round(range_size / max(range_low, 1e-10) * 100, 2),
            "range_ratio": round(float(range_ratio), 1),
            "position_in_range": round(float(position), 2),
        }

    # ── MAIN ANALYSIS ──────────────────────────────────────────
    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data for market structure analysis (need 30+ candles)"],
                reasoning="Need at least 30 candles",
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
        avg_vol = max(float(np.mean(v[-20:])), 1e-10)
        avg_body = max(float(np.mean(np.abs(c - o))), 1e-10)

        # ── Multi-timeframe swing detection ──
        swings_multi = self._find_swings_multi(h, l)

        # ── Structure analysis per lookback ──
        structure_results = {}
        for lb, (sh, sl) in swings_multi.items():
            name = {3: "Internal", 5: "Medium", 8: "External"}[lb]
            struct = self._analyze_structure(sh, sl)
            structure_results[name] = struct

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
                score += 0.05  # slight bullish — squeezes often break up

        # ── Structure shifts (BOS / CHoCH) ──
        all_shifts = []
        for lb, (sh, sl) in swings_multi.items():
            name = {3: "Internal", 5: "Medium", 8: "External"}[lb]
            shifts = self._detect_structure_shifts(sh, sl, close, lookback_name=name)
            all_shifts.extend(shifts)

        for shift in all_shifts:
            evidence.append(f"📊 {shift['label']}")
            evidence.append(f"   💡 {shift['smart_money']}")

            if shift["type"] == "BOS":
                score += 0.20 if shift["direction"] == "BULLISH" else -0.20
            elif shift["type"] == "CHoCH":
                score += 0.30 if shift["direction"] == "BULLISH" else -0.30

        data["structure_shifts"] = [
            {"type": s["type"], "direction": s["direction"], "level": round(s["level"], 2)}
            for s in all_shifts
        ]

        # ── Structure Momentum ──
        momentum = {}
        for lb, (sh, sl) in swings_multi.items():
            name = {3: "Internal", 5: "Medium", 8: "External"}[lb]
            momentum[name] = self._compute_momentum(sh, sl)

        bull_momentum = sum(1 for v in momentum.values() if v == "accelerating")
        bear_momentum = sum(1 for v in momentum.values() if v == "decelerating")
        data["momentum"] = momentum

        if bull_momentum > bear_momentum:
            evidence.append("📈 Structure momentum: accelerating (trend strengthening)")
        elif bear_momentum > bull_momentum:
            evidence.append("📉 Structure momentum: decelerating (trend weakening)")

        # ── Key support / resistance from external swings ──
        ext_highs, ext_lows = swings_multi[8]

        if ext_highs:
            nearest_res = min(
                [sh[1] for sh in ext_highs if sh[1] > close], default=None
            )
            if nearest_res is not None:
                res_dist = (nearest_res - close) / close * 100
                evidence.append(
                    f"📍 Key resistance (external): {nearest_res:.2f} ({res_dist:.2f}% above)"
                )
                if res_dist < 0.5:
                    evidence.append("  ⚠️ At resistance — watch for rejection or break")
                    score -= 0.10

        if ext_lows:
            nearest_sup = max(
                [sl[1] for sl in ext_lows if sl[1] < close], default=None
            )
            if nearest_sup is not None:
                sup_dist = (close - nearest_sup) / close * 100
                evidence.append(
                    f"📍 Key support (external): {nearest_sup:.2f} ({sup_dist:.2f}% below)"
                )
                if sup_dist < 0.5:
                    evidence.append("  ⚠️ At support — watch for bounce or break")
                    score += 0.10

        # ── Range Detection ──
        range_info = self._detect_range(h, l, c, lookback=20)
        if range_info["in_range"]:
            pct = range_info["position_in_range"]
            evidence.append(
                f"📦 RANGING MARKET — {range_info['range_pct']:.1f}% range "
                f"({range_info['range_low']:.2f} - {range_info['range_high']:.2f})"
            )
            evidence.append(f"  Price at {pct * 100:.0f}% of range")

            if pct > 0.75:
                evidence.append("  Near range top — institutions may distribute")
                score -= 0.10
            elif pct < 0.25:
                evidence.append("  Near range bottom — institutions may accumulate")
                score += 0.10
        else:
            evidence.append("📈 Trending market — no defined range")

        data["range"] = range_info

        # ── Institutional Candle Detection ──
        inst_candles = self._detect_institutional_candles(o, h, l, c, v, avg_vol, avg_body)
        if inst_candles:
            bullish_inst = [ic for ic in inst_candles if ic["direction"] == "BULLISH"]
            bearish_inst = [ic for ic in inst_candles if ic["direction"] == "BEARISH"]

            evidence.append(
                f"🏛️ Institutional candles: {len(bullish_inst)} bullish, "
                f"{len(bearish_inst)} bearish (last 5 bars)"
            )

            for ic in inst_candles[-3:]:
                emoji = "🟢" if ic["direction"] == "BULLISH" else "🔴"
                evidence.append(
                    f"  {emoji} {ic['direction']} — body {ic['body_pct']:.1f}%, "
                    f"vol {ic['vol_ratio']}x avg, close {ic['close']}"
                )

            if len(bullish_inst) > len(bearish_inst):
                evidence.append("  💡 More institutional buying candles — accumulating")
                score += 0.15
            elif len(bearish_inst) > len(bullish_inst):
                evidence.append("  💡 More institutional selling candles — distributing")
                score -= 0.15

        data["institutional_candles"] = inst_candles

        # ── Multi-TF Structure Alignment ──
        aligned_dirs = []
        for name, struct in structure_results.items():
            if struct["pattern"] == "HH_HL":
                aligned_dirs.append("bull")
            elif struct["pattern"] == "LH_LL":
                aligned_dirs.append("bear")

        if len(aligned_dirs) == 3 and len(set(aligned_dirs)) == 1:
            bonus = 0.15 if aligned_dirs[0] == "bull" else -0.15
            score += bonus
            evidence.append(
                f"🎯 ALL 3 structure timeframes aligned "
                f"{'bullish' if bonus > 0 else 'bearish'} — strong confirmation"
            )

        # ── Summary ──
        if score > 0.3:
            sma = "Accumulating longs — pushing price up"
        elif score > 0.1:
            sma = "Defending support / adding to longs"
        elif score < -0.3:
            sma = "Distributing into strength — pushing price down"
        elif score < -0.1:
            sma = "Defending resistance / adding to shorts"
        else:
            sma = "No clear directional bias"

        total_sh = sum(len(swings_multi[lb][0]) for lb in [3, 5, 8])
        total_sl = sum(len(swings_multi[lb][1]) for lb in [3, 5, 8])
        evidence.append(f"📊 Swing structure: {total_sh} highs, {total_sl} lows across 3 TFs")
        evidence.append(f"🏛️ Smart money: {sma}")

        # ── Direction / Confidence ──
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 70 + len(all_shifts) * 5 + 15)

        internal_pat = structure_results.get("Internal", {}).get("pattern", "?")
        external_pat = structure_results.get("External", {}).get("pattern", "?")
        bos_count = len([s for s in all_shifts if s["type"] == "BOS"])
        choch_count = len([s for s in all_shifts if s["type"] == "CHoCH"])

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=float(np.clip(score, -1.0, 1.0)),
            evidence=evidence,
            data=data,
            reasoning=(
                f"Structure: Internal={internal_pat}, External={external_pat}. "
                f"{bos_count} BOS, {choch_count} CHoCH. "
                f"Score {score:+.2f} → {direction}."
            ),
        )
