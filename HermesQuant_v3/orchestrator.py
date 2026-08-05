"""
HERMES QUANT v3.0 — Unified Orchestrator
Connects: 25+ Agents + ML Ensemble + Multi-TF Strategy + Risk Manager
"""
import sys
import os

# Ensure both paths are available
_base = "/data/workspace"
for p in [os.path.join(_base, "HermesQuant_v3"), os.path.join(_base, "HermesQuant"), _base]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# ── v3 Core ──
from HermesQuant_v3.core.data_fetcher import fetch_candles
from HermesQuant_v3.core import indicators as ind

# ── v3 ML ──
from HermesQuant_v3.ml.features import FeatureEngine
from HermesQuant_v3.ml.model import EnsembleModel

# ── v3 Strategy ──
from HermesQuant_v3.strategies.multi_tf import MultiTFStrategy

# ── v3 Risk ──
from HermesQuant_v3.risk.risk_manager import RiskManager

# ── Old Agents (HermesQuant v2) ──
from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.momentum_agent_v2 import MomentumAgentV2
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage1_meta.fear_greed_agent import FearGreedAgent
from agents.stage1_meta.funding_rate_agent import FundingRateAgent
from agents.stage1_meta.mtf_agent import MTFConfirmAgent
from agents.stage1_meta.open_interest_agent import OpenInterestAgent
from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage2_structure.cvd_agent import CVDAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent
from agents.stage3_decision.smart_action_agent_v2 import SmartActionAgentV2
from agents.stage4_risk.risk_agent import RiskAgent

from core.data_types import AgentOutput


class HermesOrchestrator:
    """
    Unified orchestrator connecting all components:
    - 25+ Old Agents (HermesQuant v2)
    - ML Ensemble (v3)
    - Multi-Timeframe Strategy (v3)
    - Risk Manager (v3)
    """
    
    def __init__(self, capital=10.0, leverage=20.0, max_daily_loss=3.0):
        self.capital = capital
        self.leverage = leverage
        
        # ── Old Agents ──
        self.agents = self._init_agents()
        
        # ── v3 Components ──
        self.feature_engine = FeatureEngine()
        self.ml_model = EnsembleModel()
        self.strategy = MultiTFStrategy()
        self.risk = RiskManager(capital, leverage, max_daily_loss)
        
        # ── State ──
        self.ml_trained = False
    
    def _init_agents(self):
        """Initialize all old agents"""
        agents = {}
        
        # Stage 0: Independent
        agents["trend"] = TrendAgent()
        agents["momentum"] = MomentumAgent()
        agents["momentum_v2"] = MomentumAgentV2()
        agents["volume"] = VolumeAgent()
        agents["volatility"] = VolatilityAgent()
        agents["pattern"] = PatternAgent()
        
        # Stage 1: Meta
        agents["regime"] = RegimeAgent()
        agents["structure"] = StructureAgent()
        agents["whale"] = WhaleAgent()
        agents["fear_greed"] = FearGreedAgent()
        agents["funding_rate"] = FundingRateAgent()
        agents["mtf"] = MTFConfirmAgent()
        agents["open_interest"] = OpenInterestAgent()
        
        # Stage 2: Structure
        agents["rsi_divergence"] = RSIDivergenceAgent()
        agents["bb_squeeze"] = BBSqueezeAgent()
        agents["liquidity"] = LiquidityAgent()
        agents["wyckoff"] = WyckoffAgent()
        agents["math_brain"] = MathBrainAgent()
        agents["cvd"] = CVDAgent()
        
        # Stage 3: Decision
        agents["game_theory"] = GameTheoryAgent()
        agents["smart_action"] = SmartActionAgent()
        agents["smart_action_v2"] = SmartActionAgentV2()
        
        # Stage 4: Risk
        agents["risk"] = RiskAgent()
        
        return agents
    
    def train_ml(self, df):
        """Train ML model on historical data"""
        features = self.feature_engine.compute(df)
        labels = self.feature_engine.compute_labels(df, horizon=5, threshold=0.001)
        
        if features is None or labels is None:
            return False
        
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        self.ml_trained = self.ml_model.train(X, y)
        return self.ml_trained
    
    def analyze(self, df, symbol="BTC-USDT-SWAP"):
        """
        Full analysis combining all components:
        1. Run all 25+ old agents
        2. Run ML ensemble
        3. Run multi-TF strategy
        4. Combine with risk management
        """
        result = {
            "symbol": symbol,
            "direction": "NEUTRAL",
            "confidence": 0,
            "score": 0,
            "agents": {},
            "ml": {},
            "strategy": {},
            "risk": {},
            "evidence": [],
        }
        
        # ── 1. Run Old Agents ──
        agent_votes = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        agent_scores = []
        agent_details = {}
        
        for name, agent in self.agents.items():
            try:
                r = agent.analyze(df, symbol, "15m")
                agent_votes[r.direction] = agent_votes.get(r.direction, 0) + 1
                agent_scores.append(r.score * r.weight)
                agent_details[name] = {
                    "direction": r.direction,
                    "confidence": r.confidence,
                    "score": r.score,
                    "weight": r.weight,
                }
                result["evidence"].append("%s: %s (%.0f%%)" % (name, r.direction, r.confidence))
            except Exception as e:
                agent_details[name] = {"direction": "ERROR", "error": str(e)}
        
        result["agents"] = agent_details
        
        # Agent vote result
        if agent_votes["BUY"] > agent_votes["SELL"]:
            agent_direction = "BUY"
        elif agent_votes["SELL"] > agent_votes["BUY"]:
            agent_direction = "SELL"
        else:
            agent_direction = "NEUTRAL"
        
        agent_score = np.mean(agent_scores) if agent_scores else 0
        
        # ── 2. Run ML Ensemble ──
        ml_direction = "NEUTRAL"
        ml_prob = 0.5
        ml_uncertainty = 0.5
        
        if self.ml_trained:
            features = self.feature_engine.compute(df)
            if features is not None and len(features) > 0:
                ml_prob, ml_direction, ml_uncertainty = self.ml_model.predict(features)
        
        result["ml"] = {
            "direction": ml_direction,
            "probability": ml_prob,
            "uncertainty": ml_uncertainty,
            "trained": self.ml_trained,
        }
        
        # ── 3. Run Multi-TF Strategy ──
        strat_dir, strat_conf, strat_reasons = self.strategy.analyze(df)
        
        result["strategy"] = {
            "direction": strat_dir,
            "confidence": strat_conf,
            "reasons": strat_reasons,
        }
        
        # ── 4. Combine Signals ──
        weights = {
            "agents": 0.3,
            "ml": 0.4,
            "strategy": 0.3,
        }
        
        dir_to_score = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
        
        combined_score = (
            weights["agents"] * dir_to_score.get(agent_direction, 0) * abs(agent_score) +
            weights["ml"] * (2 * ml_prob - 1) * (1 - ml_uncertainty) +
            weights["strategy"] * dir_to_score.get(strat_dir, 0) * (strat_conf / 100)
        )
        
        if combined_score > 0.1:
            final_direction = "BUY"
        elif combined_score < -0.1:
            final_direction = "SELL"
        else:
            final_direction = "NEUTRAL"
        
        final_confidence = min(abs(combined_score) * 100, 100)
        
        # ── 5. Risk Management ──
        kelly = self.risk.kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        vol_scalar = self.risk.vol_targeting(df["close"].pct_change().dropna())
        position_size = self.risk.calculate_position(
            signal_strength=abs(combined_score),
            kelly_fraction=kelly,
            vol_scalar=vol_scalar,
            meta_confidence=1 - ml_uncertainty
        )
        
        result["risk"] = {
            "kelly": kelly,
            "vol_scalar": vol_scalar,
            "position_size": position_size,
            "daily_pnl": self.risk.daily_pnl,
            "consecutive_losses": self.risk.consecutive_losses,
        }
        
        # ── 6. Final Decision ──
        if position_size > 0 and final_direction != "NEUTRAL":
            result["direction"] = final_direction
            result["confidence"] = final_confidence
            result["score"] = combined_score
        else:
            result["direction"] = "NEUTRAL"
            result["confidence"] = 0
            result["score"] = 0
        
        return result
    
    def format_report(self, result):
        """Format analysis result as readable report"""
        lines = []
        lines.append("=" * 50)
        lines.append("  HERMES QUANT v3.0 — ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append("")
        lines.append("Symbol: %s" % result["symbol"])
        lines.append("Direction: %s" % result["direction"])
        lines.append("Confidence: %.1f%%" % result["confidence"])
        lines.append("Score: %.3f" % result["score"])
        lines.append("")
        
        # Agent Summary
        lines.append("--- Agent Votes ---")
        agents = result.get("agents", {})
        buy_count = sum(1 for a in agents.values() if a.get("direction") == "BUY")
        sell_count = sum(1 for a in agents.values() if a.get("direction") == "SELL")
        neutral_count = sum(1 for a in agents.values() if a.get("direction") in ("NEUTRAL", "ERROR"))
        lines.append("BUY: %d | SELL: %d | NEUTRAL: %d" % (buy_count, sell_count, neutral_count))
        lines.append("")
        
        # ML
        ml = result.get("ml", {})
        lines.append("--- ML Ensemble ---")
        lines.append("Direction: %s" % ml.get("direction", "N/A"))
        lines.append("Probability: %.3f" % ml.get("probability", 0.5))
        lines.append("Uncertainty: %.3f" % ml.get("uncertainty", 0.5))
        lines.append("")
        
        # Strategy
        strat = result.get("strategy", {})
        lines.append("--- Multi-TF Strategy ---")
        lines.append("Direction: %s" % strat.get("direction", "N/A"))
        lines.append("Confidence: %.1f%%" % strat.get("confidence", 0))
        for reason in strat.get("reasons", [])[:3]:
            lines.append("  • %s" % reason)
        lines.append("")
        
        # Risk
        risk = result.get("risk", {})
        lines.append("--- Risk Management ---")
        lines.append("Kelly: %.4f" % risk.get("kelly", 0))
        lines.append("Vol Scalar: %.2f" % risk.get("vol_scalar", 1))
        lines.append("Position Size: $%.4f" % risk.get("position_size", 0))
        lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
