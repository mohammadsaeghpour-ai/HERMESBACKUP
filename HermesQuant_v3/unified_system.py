"""
HERMES QUANT v3.0 — Unified System
25+ Agents + ML + Strategy + Risk — Balanced BUY/SELL
"""
import sys, os
_base = "/data/workspace"
# CRITICAL: reversed() so insert(0) gives: HermesQuant > HermesQuant_v3 > _base
_needed = [os.path.join(_base, "HermesQuant"), os.path.join(_base, "HermesQuant_v3"), _base]
for p in reversed(_needed):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from collections import defaultdict

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

# ── Old Agents ──
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


# ═══════════════════════════════════════════
# AGENT CORRELATION ANALYZER
# ═══════════════════════════════════════════
class AgentCorrelationAnalyzer:
    """Analyze which agents agree/disagree and their accuracy"""
    
    def __init__(self):
        self.history = []
    
    def record(self, agent_outputs, actual_direction):
        """Record agent predictions vs actual"""
        row = {"actual": actual_direction}
        for name, output in agent_outputs.items():
            row[name] = output.get("direction", "NEUTRAL")
        self.history.append(row)
    
    def analyze(self):
        """Compute accuracy and correlation for each agent"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        results = {}
        
        for col in df.columns:
            if col == "actual":
                continue
            
            # Accuracy
            correct = (df[col] == df["actual"]).sum()
            total = len(df)
            accuracy = correct / total * 100 if total > 0 else 0
            
            # BUY accuracy
            buy_mask = df[col] == "BUY"
            buy_correct = ((df[col] == df["actual"]) & buy_mask).sum()
            buy_acc = buy_correct / buy_mask.sum() * 100 if buy_mask.sum() > 0 else 0
            
            # SELL accuracy
            sell_mask = df[col] == "SELL"
            sell_correct = ((df[col] == df["actual"]) & sell_mask).sum()
            sell_acc = sell_correct / sell_mask.sum() * 100 if sell_mask.sum() > 0 else 0
            
            # Direction distribution
            buy_count = buy_mask.sum()
            sell_count = sell_mask.sum()
            neutral_count = (df[col] == "NEUTRAL").sum()
            
            results[col] = {
                "accuracy": accuracy,
                "buy_accuracy": buy_acc,
                "sell_accuracy": sell_acc,
                "buy_count": int(buy_count),
                "sell_count": int(sell_count),
                "neutral_count": int(neutral_count),
                "bias": "BUY" if buy_count > sell_count * 1.5 else ("SELL" if sell_count > buy_count * 1.5 else "BALANCED"),
            }
        
        return results
    
    def get_best_agents(self, top_k=10):
        """Get top K agents by accuracy"""
        results = self.analyze()
        sorted_agents = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        return sorted_agents[:top_k]


# ═══════════════════════════════════════════
# UNIFIED ORCHESTRATOR
# ═══════════════════════════════════════════
class HermesUnified:
    """
    Unified system: 25+ Agents + ML + Strategy + Risk
    Balanced BUY/SELL signals
    """
    
    def __init__(self, capital=10.0, leverage=20.0, max_daily_loss=3.0):
        self.capital = capital
        self.leverage = leverage
        
        # All agents
        self.agents = {
            "trend": TrendAgent(),
            "momentum": MomentumAgent(),
            "momentum_v2": MomentumAgentV2(),
            "volume": VolumeAgent(),
            "volatility": VolatilityAgent(),
            "pattern": PatternAgent(),
            "regime": RegimeAgent(),
            "structure": StructureAgent(),
            "whale": WhaleAgent(),
            "fear_greed": FearGreedAgent(),
            "funding_rate": FundingRateAgent(),
            "mtf": MTFConfirmAgent(),
            "open_interest": OpenInterestAgent(),
            "rsi_divergence": RSIDivergenceAgent(),
            "bb_squeeze": BBSqueezeAgent(),
            "liquidity": LiquidityAgent(),
            "wyckoff": WyckoffAgent(),
            "math_brain": MathBrainAgent(),
            "cvd": CVDAgent(),
            "game_theory": GameTheoryAgent(),
            "smart_action": SmartActionAgent(),
            "smart_action_v2": SmartActionAgentV2(),
            "risk_agent": RiskAgent(),
        }
        
        # v3 components
        self.feature_engine = FeatureEngine()
        self.ml_model = EnsembleModel()
        self.strategy = MultiTFStrategy()
        self.risk = RiskManager(capital, leverage, max_daily_loss)
        self.correlation = AgentCorrelationAnalyzer()
        
        # Learned weights (start equal, update from correlation)
        self.agent_weights = {name: 1.0 for name in self.agents}
        self.ml_trained = False
    
    def train(self, df):
        """Train ML model"""
        features = self.feature_engine.compute(df)
        labels = self.feature_engine.compute_labels(df, horizon=5, threshold=0.001)
        
        if features is None or labels is None:
            return False
        
        common = features.index.intersection(labels.dropna().index)
        self.ml_trained = self.ml_model.train(features.loc[common], labels.loc[common])
        return self.ml_trained
    
    def run_agents(self, df, symbol):
        """Run all agents and collect results"""
        results = {}
        for name, agent in self.agents.items():
            try:
                r = agent.analyze(df, symbol, "15m")
                results[name] = {
                    "direction": r.direction,
                    "confidence": r.confidence,
                    "score": r.score,
                    "weight": r.weight,
                }
            except Exception:
                results[name] = {"direction": "NEUTRAL", "confidence": 0, "score": 0, "weight": 1.0}
        return results
    
    def vote(self, agent_results):
        """
        Weighted voting — NO BIAS toward BUY or SELL
        Each agent gets weight based on correlation analysis
        """
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        for name, r in agent_results.items():
            w = self.agent_weights.get(name, 1.0) * r["weight"]
            conf = r["confidence"] / 100.0  # Normalize to 0-1
            
            if r["direction"] == "BUY":
                buy_score += w * conf
            elif r["direction"] == "SELL":
                sell_score += w * conf
            total_weight += w
        
        if total_weight == 0:
            return "NEUTRAL", 0, 0
        
        buy_pct = buy_score / total_weight
        sell_pct = sell_score / total_weight
        
        # Decision: whichever side is stronger
        if buy_pct > sell_pct and buy_pct > 0.15:
            return "BUY", buy_pct * 100, buy_pct - sell_pct
        elif sell_pct > buy_pct and sell_pct > 0.15:
            return "SELL", sell_pct * 100, sell_pct - buy_pct
        else:
            return "NEUTRAL", 0, 0
    
    def analyze(self, df, symbol):
        """Full analysis — balanced BUY/SELL"""
        # 1. Run agents
        agent_results = self.run_agents(df, symbol)
        agent_dir, agent_conf, agent_margin = self.vote(agent_results)
        
        # 2. ML prediction
        ml_dir, ml_prob, ml_unc = "NEUTRAL", 0.5, 0.5
        if self.ml_trained:
            features = self.feature_engine.compute(df)
            if features is not None and len(features) > 0:
                ml_prob, ml_dir, ml_unc = self.ml_model.predict(features)
        
        # 3. Strategy
        strat_dir, strat_conf, strat_reasons = self.strategy.analyze(df)
        
        # 4. Combine (equal weight, no bias)
        dir_score = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
        
        combined = (
            0.35 * dir_score.get(agent_dir, 0) * (agent_conf / 100) +
            0.35 * (2 * ml_prob - 1) * (1 - ml_unc) +
            0.30 * dir_score.get(strat_dir, 0) * (strat_conf / 100)
        )
        
        if combined > 0.05:
            final_dir = "BUY"
        elif combined < -0.05:
            final_dir = "SELL"
        else:
            final_dir = "NEUTRAL"
        
        final_conf = min(abs(combined) * 100, 100)
        
        # 5. Risk check
        kelly = self.risk.kelly_size(0.55, 0.02, 0.01)
        vol_s = self.risk.vol_targeting(df["close"].pct_change().dropna())
        pos_size = self.risk.calculate_position(abs(combined), kelly, vol_s, 1 - ml_unc)
        
        return {
            "direction": final_dir,
            "confidence": final_conf,
            "score": combined,
            "agents": agent_results,
            "agent_vote": agent_dir,
            "ml": {"direction": ml_dir, "probability": ml_prob, "uncertainty": ml_unc},
            "strategy": {"direction": strat_dir, "confidence": strat_conf, "reasons": strat_reasons},
            "risk": {"kelly": kelly, "vol_scalar": vol_s, "position_size": pos_size},
        }


# ═══════════════════════════════════════════
# WALK-FORWARD BACKTEST (1 MONTH)
# ═══════════════════════════════════════════
class WalkForwardBacktest:
    """Walk-forward backtest with agent correlation analysis"""
    
    def __init__(self, capital=10.0, leverage=20.0, max_daily_loss=3.0,
                 train_size=500, test_size=50, horizon=5, threshold=0.001):
        self.capital = capital
        self.leverage = leverage
        self.max_daily_loss = max_daily_loss
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = horizon
        self.threshold = threshold
    
    def run(self, df, symbol):
        """Run walk-forward backtest"""
        if df is None or len(df) < self.train_size + self.test_size + self.horizon:
            return None
        
        # Storage
        all_trades = []
        agent_predictions = defaultdict(list)
        
        # Also track raw agent outputs for correlation
        agent_outputs_history = []
        actual_directions = []
        
        i = self.train_size
        
        while i + self.test_size + self.horizon <= len(df):
            # Train on past
            train_df = df.iloc[max(0, i - self.train_size):i]
            
            system = HermesUnified(self.capital, self.leverage, self.max_daily_loss)
            system.train(train_df)
            
            # Test on next window
            for j in range(self.test_size):
                idx = i + j
                if idx + self.horizon >= len(df):
                    break
                
                test_df = df.iloc[max(0, idx - 60):idx + 1]
                
                if len(test_df) < 50:
                    continue
                
                # Analyze
                result = system.analyze(test_df, symbol)
                
                if result["direction"] == "NEUTRAL":
                    continue
                
                # Actual outcome
                entry_price = df["close"].iloc[idx]
                future_price = df["close"].iloc[idx + self.horizon]
                actual_return = (future_price - entry_price) / entry_price
                
                # Did we predict correctly?
                if result["direction"] == "BUY":
                    correct = actual_return > self.threshold
                else:  # SELL
                    correct = actual_return < -self.threshold
                
                # P&L
                if correct:
                    pnl = self.capital * 0.02 * 2  # 2% risk, 2:1 R:R
                else:
                    pnl = -self.capital * 0.02
                
                trade = {
                    "idx": idx,
                    "direction": result["direction"],
                    "confidence": result["confidence"],
                    "correct": correct,
                    "pnl": pnl,
                    "actual_return": actual_return,
                    "agent_vote": result["agent_vote"],
                    "ml_dir": result["ml"]["direction"],
                    "ml_prob": result["ml"]["probability"],
                    "strat_dir": result["strategy"]["direction"],
                }
                all_trades.append(trade)
                
                # Record for correlation analysis
                agent_row = {}
                for name, r in result["agents"].items():
                    agent_row[name] = r
                agent_outputs_history.append(agent_row)
                
                actual_dir = "BUY" if actual_return > self.threshold else ("SELL" if actual_return < -self.threshold else "NEUTRAL")
                actual_directions.append(actual_dir)
            
            i += self.test_size
        
        if not all_trades:
            return None
        
        # ── Compute Results ──
        trades_df = pd.DataFrame(all_trades)
        
        # Overall
        total = len(trades_df)
        correct = trades_df["correct"].sum()
        accuracy = correct / total * 100
        
        # BUY/SELL breakdown
        buy_trades = trades_df[trades_df["direction"] == "BUY"]
        sell_trades = trades_df[trades_df["direction"] == "SELL"]
        
        buy_acc = buy_trades["correct"].mean() * 100 if len(buy_trades) > 0 else 0
        sell_acc = sell_trades["correct"].mean() * 100 if len(sell_trades) > 0 else 0
        
        # P&L
        total_pnl = trades_df["pnl"].sum()
        final_capital = self.capital + total_pnl
        
        # Max drawdown
        cum_pnl = trades_df["pnl"].cumsum()
        peak = cum_pnl.cummax()
        dd = (peak - cum_pnl) / (self.capital + peak)
        max_dd = dd.max() * 100
        
        # Agent correlation
        corr_analyzer = AgentCorrelationAnalyzer()
        for row, actual in zip(agent_outputs_history, actual_directions):
            agent_dirs = {name: {"direction": r["direction"]} for name, r in row.items()}
            corr_analyzer.record(agent_dirs, actual)
        
        agent_corr = corr_analyzer.analyze()
        
        return {
            "accuracy": accuracy,
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total": total,
            "total_pnl": total_pnl,
            "final_capital": final_capital,
            "total_return": total_pnl / self.capital * 100,
            "max_drawdown": max_dd,
            "agent_correlation": agent_corr,
        }
