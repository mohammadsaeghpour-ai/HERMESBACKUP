"""
Consensus Agent — Enhanced Multi-Aggregator
=============================================
Aggregates all agent outputs independently per timeframe.

Key features:
- Independent per-TF signals (not merged!)
- Each TF gets its own direction + confidence
- Timeframe alignment bonus: +20% if all 4 TFs agree
- Smart money boost: +15% if 3+ smart money agents agree
- Regime-based dynamic weighting
- Hunter mode: when multiple institutional patterns align
- Conflict detection: flag when whale vs trend disagree
- Minimum confidence threshold: 60%
- Output per-TF results separately

Weight: 0 (doesn't vote)
"""
from hqip.agents.base import BaseAgent, AgentOutput
from hqip.config import TF_WEIGHTS, CONSENSUS, GRADES
import numpy as np


# Smart money agents — when 3+ agree, boost confidence
SMART_MONEY_AGENTS = {"SMC", "Wyckoff", "Liquidity", "Whale", "SmartAction"}
SMART_MONEY_BOOST = 1.15  # +15%

# Hunter mode: institutional patterns that trigger maximum-priority signal
HUNTER_PATTERNS = {
    "BSL_SWEPT", "STOP_HUNT", "DISTRIBUTION",
    "ACCUMULATION", "SPRING", "UPTRAP",
}
HUNTER_BOOST = 1.30  # +30%

# Minimum confidence threshold for actionable signals
MIN_CONFIDENCE = 60.0


class ConsensusAgent(BaseAgent):
    name = "Consensus"
    weight = 0  # doesn't vote

    def analyze(self, df=None, symbol="", timeframe="",
                tf_results=None, **kwargs):
        """
        tf_results: dict of {timeframe_str: [AgentOutput, ...], ...}
                    Keys like "1h", "4h", "1d", "15m", plus "_whale_news"
                    for shared whale/news results.
        """
        if not tf_results:
            return self._out(
                direction="NO_TRADE", confidence=0,
                evidence=["No agent results to aggregate"],
                reasoning="No agent results",
            )

        evidence = []
        per_tf_results = {}   # tf -> {direction, confidence, score, agents}
        all_agent_directions = {}  # agent_name -> [(tf, direction, confidence)]

        # ═══════════════════════════════════════════════════════
        # STEP 1: Compute per-timeframe signals INDEPENDENTLY
        # ═══════════════════════════════════════════════════════
        voting_tfs = [tf for tf in tf_results if tf != "_whale_news"]

        for tf in voting_tfs:
            agent_outputs = tf_results[tf]
            tf_weight = TF_WEIGHTS.get(tf, 1.0)
            buy_score = 0.0
            sell_score = 0.0
            agents_for_tf = []

            # Also include whale/news shared results
            shared_outputs = tf_results.get("_whale_news", [])
            combined = agent_outputs + shared_outputs

            for agent_out in combined:
                if agent_out.weight == 0 and agent_out.agent_name not in (
                    "Whale", "News"
                ):
                    continue  # Skip non-voting agents (except whale/news)

                # Effective weight
                w = agent_out.weight * tf_weight
                is_smart_money = agent_out.agent_name in SMART_MONEY_AGENTS
                if is_smart_money:
                    w *= SMART_MONEY_BOOST

                if agent_out.direction == "BUY":
                    buy_score += w * (agent_out.confidence / 100)
                elif agent_out.direction == "SELL":
                    sell_score += w * (agent_out.confidence / 100)

                agents_for_tf.append(f"{agent_out.agent_name}:{agent_out.direction}")

                # Track all agent directions
                if agent_out.agent_name not in all_agent_directions:
                    all_agent_directions[agent_out.agent_name] = []
                all_agent_directions[agent_out.agent_name].append(
                    (tf, agent_out.direction, agent_out.confidence)
                )

            net = buy_score - sell_score
            tf_direction = "BUY" if net > 0.05 else "SELL" if net < -0.05 else "NEUTRAL"

            # Per-TF confidence
            max_s = max(buy_score, sell_score)
            min_s = min(buy_score, sell_score)
            tf_strength = max_s / max(max_s + min_s + 0.01, 1)
            # Count of voting agents with a direction
            active_agents = sum(1 for a in combined
                               if a.direction in ("BUY", "SELL") and (a.weight > 0 or a.agent_name in SMART_MONEY_AGENTS))
            agreement_ratio = active_agents / max(len(combined), 1)
            tf_confidence = min(100, agreement_ratio * 40 + tf_strength * 60)

            per_tf_results[tf] = {
                "direction": tf_direction,
                "confidence": round(tf_confidence, 1),
                "score": round(net, 3),
                "buy_score": round(buy_score, 3),
                "sell_score": round(sell_score, 3),
                "agent_count": len(combined),
                "agents": agents_for_tf,
            }

            evidence.append(
                f"[{tf}] BUY={buy_score:.2f} SELL={sell_score:.2f} "
                f"Net={net:+.2f} → {tf_direction} ({tf_confidence:.0f}%) "
                f"| {', '.join(agents_for_tf)}"
            )

        # ═══════════════════════════════════════════════════════
        # STEP 2: Compute aggregate direction + confidence
        # ═══════════════════════════════════════════════════════
        if not per_tf_results:
            return self._out(
                direction="NO_TRADE", confidence=0,
                evidence=["No voting timeframes"],
                reasoning="No voting timeframes",
            )

        total_weight = sum(TF_WEIGHTS.get(tf, 1.0) for tf in per_tf_results)
        total_buy = sum(r["buy_score"] for r in per_tf_results.values())
        total_sell = sum(r["sell_score"] for r in per_tf_results.values())
        total_net = total_buy - total_sell

        # ── Timeframe Agreement ──
        buy_tfs = sum(1 for r in per_tf_results.values() if r["direction"] == "BUY")
        sell_tfs = sum(1 for r in per_tf_results.values() if r["direction"] == "SELL")
        n_tfs = max(len(per_tf_results), 1)
        agree_pct = max(buy_tfs, sell_tfs) / n_tfs

        # ── Timeframe Alignment Bonus (+20% if all 4 TFs agree) ──
        all_4_aligned = (buy_tfs == 4) or (sell_tfs == 4)
        tf_alignment_bonus = 1.20 if all_4_aligned else 1.0

        if all_4_aligned:
            aligned_dir = "BUY" if buy_tfs == 4 else "SELL"
            evidence.append(
                f"🎯 ALL 4 TIMEFRAMES ALIGN on {aligned_dir} → +20% confidence bonus"
            )

        # ── Smart Money Consensus (3+ agree) ──
        sm_buy = []
        sm_sell = []
        for agent_name in SMART_MONEY_AGENTS:
            if agent_name in all_agent_directions:
                entries = all_agent_directions[agent_name]
                best = max(entries, key=lambda e: TF_WEIGHTS.get(e[0], 1.0))
                if best[1] == "BUY":
                    sm_buy.append(agent_name)
                elif best[1] == "SELL":
                    sm_sell.append(agent_name)

        smart_money_boost = 1.0
        if len(sm_buy) >= 3:
            evidence.append(
                f"🏦 SMART MONEY BUY: {', '.join(sm_buy)} ({len(sm_buy)} agents agree) "
                f"→ +15% boost"
            )
            smart_money_boost = SMART_MONEY_BOOST
        if len(sm_sell) >= 3:
            evidence.append(
                f"🏦 SMART MONEY SELL: {', '.join(sm_sell)} ({len(sm_sell)} agents agree) "
                f"→ +15% boost"
            )
            smart_money_boost = SMART_MONEY_BOOST

        # ── Conflict Detection (whale vs trend) ──
        contrarian_flag = False
        if "Whale" in all_agent_directions and "Trend" in all_agent_directions:
            whale_primary = max(
                all_agent_directions["Whale"],
                key=lambda e: TF_WEIGHTS.get(e[0], 1.0)
            )
            trend_primary = max(
                all_agent_directions["Trend"],
                key=lambda e: TF_WEIGHTS.get(e[0], 1.0)
            )
            if whale_primary[1] != "NEUTRAL" and trend_primary[1] != "NEUTRAL":
                if whale_primary[1] != trend_primary[1]:
                    contrarian_flag = True
                    evidence.append(
                        f"⚠️ CONFLICT: Whale={whale_primary[1]} vs "
                        f"Trend={trend_primary[1]}. "
                        f"Smart money may be fading the trend."
                    )

        # ── Direction Decision ──
        if total_buy > total_sell:
            raw_direction = "BUY"
        elif total_sell > total_buy:
            raw_direction = "SELL"
        else:
            raw_direction = "NO_TRADE"

        # ── Confidence Calculation ──
        max_score = max(total_buy, total_sell)
        min_score = min(total_buy, total_sell)
        strength_ratio = max_score / max(max_score + min_score + 0.01, 1)
        confidence = agree_pct * 50 + strength_ratio * 50

        # Apply bonuses
        confidence *= tf_alignment_bonus
        confidence *= smart_money_boost

        # ═══════════════════════════════════════════════════════
        # STEP 3: Hunter Mode Detection
        # ═══════════════════════════════════════════════════════
        hunter_mode = False
        detected_hunter_patterns = set()

        for tf in tf_results:
            for agent_out in tf_results[tf]:
                agent_data = getattr(agent_out, "data", {}) or {}

                # SmartAction patterns
                if agent_out.agent_name == "SmartAction":
                    for action in agent_data.get("detected_actions", []):
                        action_upper = str(action).upper()
                        for hp in HUNTER_PATTERNS:
                            if hp in action_upper:
                                detected_hunter_patterns.add(hp)

                # SMC patterns
                if agent_out.agent_name == "SMC":
                    for p in agent_data.get("detected_patterns", []):
                        p_str = str(p).upper()
                        if "BSL" in p_str or "SWEPT" in p_str:
                            detected_hunter_patterns.add("BSL_SWEPT")

                # Wyckoff phases
                if agent_out.agent_name == "Wyckoff":
                    phase = str(agent_data.get("phase", "")).lower()
                    if "distribution" in phase:
                        detected_hunter_patterns.add("DISTRIBUTION")
                    if "spring" in phase:
                        detected_hunter_patterns.add("SPRING")

        if len(detected_hunter_patterns) >= 3:
            hunter_mode = True
            confidence *= HUNTER_BOOST
            evidence.append(
                f"🎯🎯 HUNTER MODE 🎯🎯 — "
                f"{len(detected_hunter_patterns)} institutional patterns: "
                f"{', '.join(sorted(detected_hunter_patterns))}"
            )

        # ═══════════════════════════════════════════════════════
        # STEP 4: Final Direction with Thresholds
        # ═══════════════════════════════════════════════════════
        confidence = min(100, confidence)
        direction = "NO_TRADE"

        if raw_direction in ("BUY", "SELL"):
            if agree_pct >= CONSENSUS["min_weighted_agreement"]:
                direction = raw_direction
            else:
                evidence.append(
                    f"⚠️ Agreement {agree_pct:.0%} < "
                    f"{CONSENSUS['min_weighted_agreement']:.0%} threshold"
                )

        # Minimum confidence threshold
        if direction in ("BUY", "SELL") and confidence < MIN_CONFIDENCE:
            evidence.append(
                f"⏳ WAIT: {direction} but confidence {confidence:.0f}% "
                f"< {MIN_CONFIDENCE:.0f}% minimum"
            )
            direction = "WAIT"

        if direction == "NO_TRADE":
            confidence = min(30, confidence)

        if direction not in ("NO_TRADE", "WAIT"):
            evidence.append(
                f"✅ {direction}: Agreement {agree_pct:.0%} | "
                f"Confidence {confidence:.0f}%"
            )

        # ═══════════════════════════════════════════════════════
        # STEP 5: Grade
        # ═══════════════════════════════════════════════════════
        grade = "C-"
        for g, thresholds in sorted(GRADES.items(), key=lambda x: -x[1]["conf"]):
            conf_norm = confidence / 100
            if conf_norm >= thresholds["conf"] and agree_pct >= thresholds["agree"]:
                grade = g
                break

        strength_map = {
            "A+": "VERY STRONG — highest conviction",
            "A": "STRONG — high conviction",
            "A-": "STRONG — solid signal",
            "B+": "MODERATE-STRONG — good signal",
            "B": "MODERATE — decent signal",
            "B-": "MODERATE-WEAK — marginal",
            "C+": "WEAK — low conviction",
            "C": "VERY WEAK — barely qualifying",
            "C-": "NEGLIGIBLE — below threshold",
        }
        strength_desc = strength_map.get(grade, "UNKNOWN")
        evidence.append(f"Grade: {grade} ({strength_desc})")

        # ── Build reasoning ──
        explanation_parts = []
        for tf, result in per_tf_results.items():
            if result["direction"] == "BUY":
                explanation_parts.append(f"{tf}: bullish (net={result['score']:+.2f})")
            elif result["direction"] == "SELL":
                explanation_parts.append(f"{tf}: bearish (net={result['score']:+.2f})")
            else:
                explanation_parts.append(f"{tf}: neutral")

        reasoning = (
            f"{'Multi-TF ' + direction if direction not in ('NO_TRADE', 'WAIT') else direction}: "
            f"{', '.join(explanation_parts)}. Grade {grade}. "
        )
        if hunter_mode:
            reasoning += f"HUNTER MODE ({len(detected_hunter_patterns)} patterns). "
        if contrarian_flag:
            reasoning += "Whale vs trend conflict. "
        if all_4_aligned:
            reasoning += "All 4 TFs aligned (+20%). "

        # ── Data output ──
        data = {
            "grade": grade,
            "strength_desc": strength_desc,
            "agree_pct": round(agree_pct, 3),
            "buy_tfs": buy_tfs,
            "sell_tfs": sell_tfs,
            "tf_alignment_bonus": tf_alignment_bonus,
            "all_4_tfs_aligned": all_4_aligned,
            "contrarian_flag": contrarian_flag,
            "hunter_mode": hunter_mode,
            "hunter_patterns": sorted(detected_hunter_patterns),
            "smart_money_buy": sm_buy,
            "smart_money_sell": sm_sell,
            "smart_money_boost": smart_money_boost,
            "min_confidence_threshold": MIN_CONFIDENCE,
            "per_tf_results": per_tf_results,
        }

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=float(np.clip(total_net / max(total_weight, 1), -1, 1)),
            evidence=evidence,
            data=data,
            reasoning=reasoning,
        )
