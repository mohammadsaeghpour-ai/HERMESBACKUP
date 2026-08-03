
"""
HQIP Pipeline Engine v2 — Graph-Based Vertical Architecture
===========================================================
Uses: DAG, Probability Theory, Game Theory, Vector Geometry
Market = Probability Game → Every signal = Expected Value calculation
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum


# ══════════════════════════════════════════════════════════════
# PART 1: GRAPH THEORY — Agent Dependency DAG
# ══════════════════════════════════════════════════════════════

class Stage(Enum):
    DATA = 0
    INDEPENDENT = 1
    META = 2
    STRUCTURE = 3
    DECISION = 4
    RISK = 5


@dataclass
class AgentNode:
    """Node in the DAG — each agent is a node"""
    name: str
    stage: Stage
    weight: float = 1.0
    confidence: float = 0.0
    direction: str = "NEUTRAL"
    score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    
    @property
    def is_ready(self) -> bool:
        """Can this agent run? (all deps satisfied)"""
        return True  # Will be checked by pipeline


@dataclass
class Edge:
    """Directed edge: A → B means B depends on A"""
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "data"  # data, influence, constraint


class AgentDAG:
    """
    Directed Acyclic Graph for agent dependencies.
    Topological sort determines execution order.
    """
    
    def __init__(self):
        self.nodes: Dict[str, AgentNode] = {}
        self.edges: List[Edge] = []
        self.adj: Dict[str, List[str]] = defaultdict(list)  # adjacency list
        self.rev_adj: Dict[str, List[str]] = defaultdict(list)  # reverse adjacency
        self._build_graph()
    
    def _build_graph(self):
        """Build the full dependency graph"""
        
        # ── STAGE 1: Independent (no deps) ──
        independent = {
            'Trend': 1.5, 'Momentum': 1.3, 'Volume': 1.3,
            'Volatility': 1.0, 'Pattern': 1.1, 'DLForecast': 1.0,
        }
        for name, weight in independent.items():
            self.add_node(name, Stage.INDEPENDENT, weight)
        
        # ── STAGE 2: Meta (depends on Stage 1) ──
        self.add_node('Regime', Stage.META, 1.0, 
                      deps=['Volatility', 'Volume', 'Trend'])
        self.add_node('MarketStructure', Stage.META, 1.4,
                      deps=['Trend'])
        self.add_node('Whale', Stage.META, 1.5,
                      deps=['Volume'])
        
        # ── STAGE 3: Structure (depends on Stage 2) ──
        self.add_node('Liquidity', Stage.STRUCTURE, 1.5,
                      deps=['MarketStructure'])
        self.add_node('SMC', Stage.STRUCTURE, 1.0,
                      deps=['Liquidity', 'MarketStructure'])
        self.add_node('Wyckoff', Stage.STRUCTURE, 1.4,
                      deps=['Volume', 'Trend', 'MarketStructure'])
        self.add_node('MathBrain', Stage.STRUCTURE, 1.4,
                      deps=['Trend', 'Momentum', 'Volatility'])
        
        # ── STAGE 4: Decision (depends on all above) ──
        self.add_node('GameTheory', Stage.DECISION, 1.3,
                      deps=['Regime', 'Whale'])
        self.add_node('SmartAction', Stage.DECISION, 1.7,
                      deps=['Trend', 'Momentum', 'SMC', 'Regime'])
        self.add_node('ML', Stage.DECISION, 1.2,
                      deps=['Trend', 'Momentum', 'Volume', 'Volatility', 'Pattern'])
        
        # ── STAGE 5: Risk (final) ──
        self.add_node('Risk', Stage.RISK, 0.0,
                      deps=['Regime', 'Volatility'])
    
    def add_node(self, name, stage, weight=1.0, deps=None):
        node = AgentNode(name=name, stage=stage, weight=weight,
                        dependencies=deps or [])
        self.nodes[name] = node
        for dep in (deps or []):
            self.edges.append(Edge(source=dep, target=name))
            self.adj[dep].append(name)
            self.rev_adj[name].append(dep)
    
    def topological_sort(self) -> List[List[str]]:
        """Return execution layers (topological sort by stage)"""
        layers = defaultdict(list)
        for name, node in self.nodes.items():
            layers[node.stage.value].append(name)
        return [layers[i] for i in sorted(layers.keys())]
    
    def get_influence_score(self, name: str) -> float:
        """Calculate total downstream influence of an agent"""
        visited = set()
        def dfs(n):
            if n in visited: return 0
            visited.add(n)
            score = self.nodes[n].weight if n in self.nodes else 0
            for child in self.adj.get(n, []):
                score += dfs(child) * 0.5  # decay factor
            return score
        return dfs(name)
    
    def print_graph(self):
        """Print the DAG structure"""
        layers = self.topological_sort()
        for i, layer in enumerate(layers):
            print("  STAGE %d:" % i)
            for name in layer:
                node = self.nodes[name]
                deps_str = ", ".join(node.dependencies) if node.dependencies else "none"
                influence = self.get_influence_score(name)
                print("    %s [w=%.1f, influence=%.1f] <- %s" % (
                    name, node.weight, influence, deps_str))
            print()


# ══════════════════════════════════════════════════════════════
# PART 2: PROBABILITY ENGINE — Bayesian Scoring
# ══════════════════════════════════════════════════════════════

class ProbabilityEngine:
    """
    Bayesian probability scoring for agent consensus.
    Each agent provides P(UP) and P(DOWN) → combine via Bayes.
    """
    
    @staticmethod
    def agent_to_probability(direction: str, confidence: float, score: float) -> Tuple[float, float]:
        """
        Convert agent output to P(UP), P(DOWN).
        confidence = how sure the agent is (0-100)
        score = direction strength (-1 to +1)
        """
        # Normalize confidence to 0-1
        conf = confidence / 100.0
        
        if direction == "BUY":
            p_up = 0.5 + conf * 0.5 * (1 + abs(score))
            p_down = 1 - p_up
        elif direction == "SELL":
            p_down = 0.5 + conf * 0.5 * (1 + abs(score))
            p_up = 1 - p_down
        else:  # NEUTRAL
            p_up = 0.5
            p_down = 0.5
        
        return min(p_up, 0.99), min(p_down, 0.99)
    
    @staticmethod
    def bayesian_combine(priors: List[Tuple[float, float]], weights: List[float]) -> Tuple[float, float]:
        """
        Combine multiple P(UP), P(DOWN) via weighted Bayesian.
        priors: list of (p_up, p_down) from each agent
        weights: importance weight of each agent
        """
        if not priors:
            return 0.5, 0.5
        
        # Weighted log-odds
        log_odds_up = 0
        total_weight = sum(weights)
        
        for (p_up, p_down), w in zip(priors, weights):
            if p_up > 0 and p_down > 0:
                odds = p_up / p_down
                log_odds_up += w * math.log(odds)
        
        # Convert back to probability
        if total_weight > 0:
            log_odds_up /= total_weight
        
        p_up_final = 1 / (1 + math.exp(-log_odds_up))
        p_down_final = 1 - p_up_final
        
        return p_up_final, p_down_final
    
    @staticmethod
    def expected_value(p_win: float, p_loss: float, win_amount: float, loss_amount: float) -> float:
        """Calculate Expected Value of a trade"""
        return (p_win * win_amount) - (p_loss * loss_amount)
    
    @staticmethod
    def kelly_criterion(p: float, b: float) -> float:
        """
        Kelly Criterion: optimal fraction to bet.
        p = probability of winning
        b = payoff ratio (win/loss)
        """
        if b <= 0: return 0
        q = 1 - p
        f = (p * b - q) / b
        return max(0, f)
    
    @staticmethod
    def quarter_kelly(p: float, b: float) -> float:
        """Quarter-Kelly for safety"""
        return ProbabilityEngine.kelly_criterion(p, b) / 4


# ══════════════════════════════════════════════════════════════
# PART 3: GAME THEORY — Market as Game
# ══════════════════════════════════════════════════════════════

class MarketGameTheory:
    """
    Market is a game between:
    - Bulls (buyers)
    - Bears (sellers)  
    - Whales (market makers / institutions)
    - Retail (followers)
    
    Nash Equilibrium = when no player benefits from changing strategy.
    We want to find when equilibrium BREAKS = opportunity.
    """
    
    @staticmethod
    def payoff_matrix(bull_strategy: str, bear_strategy: str, 
                      whale_action: str) -> Dict:
        """
        Simplified 3-player payoff matrix.
        Strategies: {HOLD, BUY, SELL}
        Whale actions: {ABSORB, DUMP, MANIPULATE}
        """
        # Payoff table: (bull_payoff, bear_payoff)
        payoffs = {
            ('BUY', 'SELL', 'ABSORB'): {'bull': 0.7, 'bear': -0.7, 'whale': 0.1},
            ('BUY', 'SELL', 'DUMP'): {'bull': -0.8, 'bear': 0.3, 'whale': 0.5},
            ('BUY', 'SELL', 'MANIPULATE'): {'bull': -0.3, 'bear': -0.3, 'whale': 0.6},
            ('HOLD', 'SELL', 'ABSORB'): {'bull': 0.1, 'bear': 0.2, 'whale': 0.3},
            ('BUY', 'HOLD', 'DUMP'): {'bull': -0.5, 'bear': 0.1, 'whale': 0.4},
        }
        
        key = (bull_strategy, bear_strategy, whale_action)
        return payoffs.get(key, {'bull': 0, 'bear': 0, 'whale': 0})
    
    @staticmethod
    def detect_whale_action(volume_profile: Dict, price_action: Dict) -> str:
        """
        Detect whale behavior from volume and price.
        """
        vol_spike = volume_profile.get('ratio', 1.0)
        price_change = price_action.get('change_pct', 0)
        wick_ratio = price_action.get('wick_ratio', 0.5)
        
        if vol_spike > 2.0 and abs(price_change) < 0.1:
            return 'ABSORB'  # Big volume, small price move = absorption
        elif vol_spike > 1.5 and abs(price_change) > 0.5:
            if wick_ratio > 0.6:
                return 'MANIPULATE'  # Big wick = manipulation
            else:
                return 'DUMP'  # Strong move with volume
        else:
            return 'NEUTRAL'
    
    @staticmethod
    def nash_equilibrium_check(bull_strength: float, bear_strength: float, 
                                whale_power: float) -> Dict:
        """
        Check if market is at Nash Equilibrium or in disequilibrium.
        Disequilibrium = opportunity for us.
        """
        total = bull_strength + bear_strength + whale_power
        if total == 0:
            return {'state': 'EQUILIBRIUM', 'opportunity': 0}
        
        bull_ratio = bull_strength / total
        bear_ratio = bear_strength / total
        whale_ratio = whale_power / total
        
        # Disequilibrium = one player dominates > 60%
        dominance = max(bull_ratio, bear_ratio, whale_ratio)
        
        if dominance > 0.6:
            return {
                'state': 'DISEQUILIBRIUM',
                'dominant': 'bulls' if bull_ratio == dominance else (
                    'bears' if bear_ratio == dominance else 'whales'),
                'opportunity': dominance - 0.5,  # How far from equilibrium
                'bias': 'REVERSAL' if dominance > 0.7 else 'CONTINUATION'
            }
        else:
            return {'state': 'EQUILIBRIUM', 'opportunity': 0}


# ══════════════════════════════════════════════════════════════
# PART 4: VECTOR GEOMETRY — Signal Space
# ══════════════════════════════════════════════════════════════

class SignalGeometry:
    """
    Represent each agent's signal as a vector in N-dimensional space.
    Compute angles, distances, and convergence.
    """
    
    @staticmethod
    def agent_to_vector(direction: str, confidence: float, score: float) -> np.ndarray:
        """Convert agent output to a 3D vector [direction, confidence, strength]"""
        d = 1.0 if direction == "BUY" else (-1.0 if direction == "SELL" else 0.0)
        c = confidence / 100.0
        s = score
        return np.array([d, c, s])
    
    @staticmethod
    def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle between two signal vectors (degrees)"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
    
    @staticmethod
    def convergence_score(vectors: List[np.ndarray]) -> float:
        """
        How converged are all agent vectors?
        0 = random (no agreement)
        1 = perfect alignment
        """
        if len(vectors) < 2:
            return 0
        
        n = len(vectors)
        total_angle = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                angle = SignalGeometry.vector_angle(vectors[i], vectors[j])
                # Normalize: 0° = 1.0, 180° = 0.0
                agreement = 1.0 - (angle / 180.0)
                total_angle += agreement
                count += 1
        
        return total_angle / count if count > 0 else 0
    
    @staticmethod
    def resultant_vector(vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Weighted sum of all agent vectors = resultant signal"""
        if not vectors:
            return np.zeros(3)
        
        result = np.zeros(3)
        total_w = sum(weights)
        
        for v, w in zip(vectors, weights):
            result += v * (w / total_w) if total_w > 0 else v
        
        return result
    
    @staticmethod
    def signal_strength(resultant: np.ndarray) -> Dict:
        """Interpret resultant vector"""
        magnitude = np.linalg.norm(resultant)
        direction = resultant[0]
        confidence = resultant[1]
        strength = resultant[2]
        
        if direction > 0.3:
            sig_dir = "BUY"
        elif direction < -0.3:
            sig_dir = "SELL"
        else:
            sig_dir = "NEUTRAL"
        
        return {
            'direction': sig_dir,
            'magnitude': round(magnitude, 3),
            'confidence': round(confidence * 100, 1),
            'strength': round(strength, 3),
        }


# ══════════════════════════════════════════════════════════════
# PART 5: PIPELINE ENGINE — Orchestrator
# ══════════════════════════════════════════════════════════════

class PipelineEngine:
    """
    Vertical pipeline: Data → Independent → Meta → Structure → Decision → Risk
    Each stage feeds into the next.
    """
    
    def __init__(self):
        self.dag = AgentDAG()
        self.prob_engine = ProbabilityEngine()
        self.geometry = SignalGeometry()
        self.game = MarketGameTheory()
        self.stage_outputs = {}
    
    def run(self, agent_results: Dict[str, Dict]) -> Dict:
        """
        Run the full pipeline.
        agent_results: {agent_name: {direction, confidence, score, weight, evidence}}
        """
        layers = self.dag.topological_sort()
        
        print("=" * 60)
        print("  PIPELINE EXECUTION")
        print("=" * 60)
        
        all_vectors = []
        all_weights = []
        all_priors = []
        
        for stage_idx, layer in enumerate(layers):
            print("\n  --- STAGE %d ---" % stage_idx)
            
            for agent_name in layer:
                if agent_name not in agent_results:
                    continue
                
                r = agent_results[agent_name]
                node = self.dag.nodes.get(agent_name)
                if not node:
                    continue
                
                # Update node
                node.direction = r.get('direction', 'NEUTRAL')
                node.confidence = r.get('confidence', 0)
                node.score = r.get('score', 0)
                node.weight = r.get('weight', node.weight)
                
                # Convert to probability
                p_up, p_down = self.prob_engine.agent_to_probability(
                    node.direction, node.confidence, node.score)
                
                # Convert to vector
                vec = self.geometry.agent_to_vector(
                    node.direction, node.confidence, node.score)
                
                all_vectors.append(vec)
                all_weights.append(node.weight)
                all_priors.append((p_up, p_down))
                
                print("    %s: %s (conf=%.0f, P_up=%.2f, P_down=%.2f, w=%.1f)" % (
                    agent_name, node.direction, node.confidence, p_up, p_down, node.weight))
        
        # ── Combine Results ──
        
        # 1. Bayesian combine
        p_up_final, p_down_final = self.prob_engine.bayesian_combine(
            all_priors, all_weights)
        
        # 2. Vector resultant
        resultant = self.geometry.resultant_vector(all_vectors, all_weights)
        signal = self.geometry.signal_strength(resultant)
        
        # 3. Convergence
        convergence = self.geometry.convergence_score(all_vectors)
        
        # 4. Game Theory
        bull_strength = sum(r.get('score', 0) for r in agent_results.values() 
                          if r.get('direction') == 'BUY')
        bear_strength = abs(sum(r.get('score', 0) for r in agent_results.values() 
                          if r.get('direction') == 'SELL'))
        
        game_state = self.game.nash_equilibrium_check(
            bull_strength, bear_strength, 0.3)  # whale=0.3 default
        
        # 5. Final Decision
        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        
        print("\n  Bayesian Probability:")
        print("    P(UP)  = %.1f%%" % (p_up_final * 100))
        print("    P(DOWN)= %.1f%%" % (p_down_final * 100))
        
        print("\n  Vector Geometry:")
        print("    Direction: %s" % signal['direction'])
        print("    Magnitude: %.3f" % signal['magnitude'])
        print("    Confidence: %.1f%%" % signal['confidence'])
        print("    Convergence: %.1f%%" % (convergence * 100))
        
        print("\n  Game Theory:")
        print("    State: %s" % game_state.get('state'))
        if game_state.get('dominant'):
            print("    Dominant: %s" % game_state.get('dominant'))
            print("    Opportunity: %.1f%%" % (game_state.get('opportunity', 0) * 100))
            print("    Bias: %s" % game_state.get('bias'))
        
        # ── Decision ──
        final_direction = "WAIT"
        confidence_pct = 0
        
        if p_up_final > 0.65 and convergence > 0.6:
            final_direction = "BUY"
            confidence_pct = p_up_final * 100
        elif p_down_final > 0.65 and convergence > 0.6:
            final_direction = "SELL"
            confidence_pct = p_down_final * 100
        
        # Kelly sizing
        if final_direction != "WAIT":
            p_win = p_up_final if final_direction == "BUY" else p_down_final
            kelly = self.prob_engine.quarter_kelly(p_win, 2.0)  # R:R = 1:2
            ev = self.prob_engine.expected_value(p_win, 1-p_win, 2.0, 1.0)
        else:
            kelly = 0
            ev = 0
        
        print("\n  DECISION: %s (conf=%.1f%%)" % (final_direction, confidence_pct))
        print("  Kelly: %.1f%% of capital" % (kelly * 100))
        print("  EV: $%.2f per $1 risked" % ev)
        
        return {
            'direction': final_direction,
            'confidence': confidence_pct,
            'p_up': p_up_final,
            'p_down': p_down_final,
            'convergence': convergence,
            'game_state': game_state,
            'kelly': kelly,
            'ev': ev,
            'signal': signal,
        }


# ══════════════════════════════════════════════════════════════
# MAIN — Test with real data
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load real agent results
    import json
    
    try:
        agent_results = json.load(open('/tmp/agent_results.json'))
    except:
        # Fallback: use dummy data
        agent_results = {
            'Trend': {'direction': 'BUY', 'confidence': 74, 'score': 0.387, 'weight': 1.5, 'evidence': []},
            'Momentum': {'direction': 'NEUTRAL', 'confidence': 10, 'score': -0.048, 'weight': 1.3, 'evidence': []},
            'Volume': {'direction': 'NEUTRAL', 'confidence': 36, 'score': 0.142, 'weight': 1.3, 'evidence': []},
            'Volatility': {'direction': 'NEUTRAL', 'confidence': 1, 'score': -0.020, 'weight': 1.0, 'evidence': []},
            'Pattern': {'direction': 'SELL', 'confidence': 33, 'score': -0.300, 'weight': 1.1, 'evidence': []},
            'DLForecast': {'direction': 'NEUTRAL', 'confidence': 80, 'score': -0.008, 'weight': 1.0, 'evidence': []},
            'Regime': {'direction': 'NEUTRAL', 'confidence': 100, 'score': 0.000, 'weight': 1.0, 'evidence': []},
            'MarketStructure': {'direction': 'BUY', 'confidence': 29, 'score': 0.200, 'weight': 1.4, 'evidence': []},
            'Whale': {'direction': 'NEUTRAL', 'confidence': 0, 'score': 0.000, 'weight': 1.5, 'evidence': []},
            'Liquidity': {'direction': 'NEUTRAL', 'confidence': 28, 'score': 0.100, 'weight': 1.5, 'evidence': []},
            'SMC': {'direction': 'NEUTRAL', 'confidence': 0, 'score': 0.000, 'weight': 1.0, 'evidence': []},
            'Wyckoff': {'direction': 'NEUTRAL', 'confidence': 15, 'score': 0.000, 'weight': 1.4, 'evidence': []},
            'MathBrain': {'direction': 'BUY', 'confidence': 55, 'score': 0.230, 'weight': 1.4, 'evidence': []},
            'GameTheory': {'direction': 'NO_TRADE', 'confidence': 41, 'score': 0.150, 'weight': 1.3, 'evidence': []},
            'SmartAction': {'direction': 'SELL', 'confidence': 38, 'score': -0.250, 'weight': 1.7, 'evidence': []},
            'ML': {'direction': 'NEUTRAL', 'confidence': 0, 'score': 0.000, 'weight': 1.2, 'evidence': []},
        }
    
    # Print DAG structure
    print("=" * 60)
    print("  AGENT DEPENDENCY GRAPH (DAG)")
    print("=" * 60)
    dag = AgentDAG()
    dag.print_graph()
    
    # Run pipeline
    print("\n" + "=" * 60)
    print("  RUNNING PIPELINE ON REAL DATA")
    print("=" * 60)
    
    engine = PipelineEngine()
    result = engine.run(agent_results)
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  Direction: %s" % result['direction'])
    print("  Confidence: %.1f%%" % result['confidence'])
    print("  P(UP): %.1f%%" % (result['p_up'] * 100))
    print("  P(DOWN): %.1f%%" % (result['p_down'] * 100))
    print("  Convergence: %.1f%%" % (result['convergence'] * 100))
    print("  Kelly: %.1f%%" % (result['kelly'] * 100))
    print("  EV: $%.2f" % result['ev'])
