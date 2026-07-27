"""
Consensus Agent
===============
Aggregates all agent outputs across timeframes.
Only this agent outputs BUY/SELL/NO_TRADE.
"""
from hqip.agents.base import BaseAgent, AgentOutput
from hqip.config import TF_WEIGHTS, CONSENSUS, GRADES
import numpy as np

class ConsensusAgent(BaseAgent):
    name = "Consensus"
    weight = 0.0

    def analyze(self, df=None, symbol="", timeframe="",
                tf_results=None, **kwargs):
        """
        tf_results: dict of {timeframe: [AgentOutput, ...]}
        """
        if not tf_results:
            return self._out(direction="NO_TRADE", confidence=0, evidence=["No agent results"])

        evidence = []
        all_signals = []

        for tf, agent_outputs in tf_results.items():
            tf_weight = TF_WEIGHTS.get(tf, 1.0)
            buy_score = 0
            sell_score = 0
            agents_for_tf = []

            for agent_out in agent_outputs:
                if agent_out.weight == 0:  # Skip non-voting agents
                    continue
                w = agent_out.weight * tf_weight
                if agent_out.direction == "BUY":
                    buy_score += w * (agent_out.confidence / 100)
                elif agent_out.direction == "SELL":
                    sell_score += w * (agent_out.confidence / 100)
                agents_for_tf.append(f"{agent_out.agent_name}:{agent_out.direction}")

            net = buy_score - sell_score
            evidence.append(f"[{tf}] BUY={buy_score:.2f} SELL={sell_score:.2f} Net={net:+.2f} | {', '.join(agents_for_tf)}")
            all_signals.append({"tf": tf, "buy_score": buy_score, "sell_score": sell_score, "net": net, "weight": tf_weight})

        # Calculate weighted agreement
        total_weight = sum(s["weight"] for s in all_signals)
        total_buy = sum(s["buy_score"] for s in all_signals)
        total_sell = sum(s["sell_score"] for s in all_signals)
        total_net = total_buy - total_sell

        # Agreement: how many TFs agree on direction
        buy_tfs = sum(1 for s in all_signals if s["net"] > 0)
        sell_tfs = sum(1 for s in all_signals if s["net"] < 0)
        agree_pct = max(buy_tfs, sell_tfs) / max(len(all_signals), 1)

        direction = "NO_TRADE"
        if total_buy > 0 and total_buy > total_sell:
            if agree_pct >= CONSENSUS["min_weighted_agreement"]:
                direction = "BUY"
        elif total_sell > 0 and total_sell > total_buy:
            if agree_pct >= CONSENSUS["min_weighted_agreement"]:
                direction = "SELL"

        # Confidence
        max_score = max(total_buy, total_sell)
        min_score = min(total_buy, total_sell)
        confidence = min(100, (agree_pct * 50 + (max_score / max(max_score + min_score + 0.01, 1)) * 50))

        if direction == "NO_TRADE":
            confidence = min(30, confidence)
            evidence.append(f"⚠️ NO TRADE: Agreement {agree_pct:.0%} < {CONSENSUS['min_weighted_agreement']:.0%} threshold")
        else:
            evidence.append(f"✅ {direction}: Agreement {agree_pct:.0%} | Confidence {confidence:.0f}%")

        # Grade
        grade = "C"
        for g, thresholds in sorted(GRADES.items(), key=lambda x: -x[1]["conf"]):
            if confidence / 100 >= thresholds["conf"] and agree_pct >= thresholds["agree"]:
                grade = g
                break

        evidence.append(f"Grade: {grade}")

        # Build explanation
        explanation_parts = []
        for s in all_signals:
            if s["net"] > 0:
                explanation_parts.append(f"{s['tf']}: bullish (net={s['net']:.2f})")
            elif s["net"] < 0:
                explanation_parts.append(f"{s['tf']}: bearish (net={s['net']:.2f})")
            else:
                explanation_parts.append(f"{s['tf']}: neutral")

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(total_net / max(total_weight, 1), -1, 1),
            evidence=evidence,
            data={"grade": grade, "agree_pct": agree_pct, "buy_tfs": buy_tfs, "sell_tfs": sell_tfs,
                  "tf_details": {s["tf"]: {"net": s["net"], "buy": s["buy_score"], "sell": s["sell_score"]} for s in all_signals}},
            reasoning=f"{'Multi-TF agreement' if direction != 'NO_TRADE' else 'No consensus'}: {', '.join(explanation_parts)}"
        )
