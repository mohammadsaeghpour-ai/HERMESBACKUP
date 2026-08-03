"""
Consensus Engine — Multi-agent signal aggregation.

Aggregates results from independent analysis agents using weighted
scoring, Bayesian probability updating, and expected-value gating.
Requires minimum agent agreement before issuing a trade signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bayesian import BayesianUpdater
from .expected_value import calculate_ev, minimum_viable_ev


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


@dataclass
class AgentResult:
    """Output produced by a single analysis agent."""
    agent_name: str
    direction: Direction
    score: float          # normalised signal strength 0-100
    confidence: float     # agent's self-assessed confidence 0-100
    weight: float = 1.0   # relative importance of this agent

    # Optional historical stats used for EV calculation
    historical_win_rate: Optional[float] = None
    historical_avg_win: Optional[float] = None
    historical_avg_loss: Optional[float] = None


@dataclass
class ConsensusDecision:
    """Final consensus output."""
    direction: Direction
    confidence: float
    expected_value: float
    kelly_fraction: float
    supporting_agents: List[str]
    conflicting_agents: List[str]
    total_agents: int
    agreeing_agents: int
    prior_probability: float
    posterior_probability: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """True when consensus recommends taking a trade."""
        return (
            self.direction != Direction.NO_TRADE
            and self.confidence >= 60.0
            and self.expected_value > 0
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsensusEngine:
    """
    Multi-agent consensus engine.

    Parameters
    ----------
    min_agents : int
        Minimum number of agents that must agree on the same direction
        before a trade signal is issued.  Default 3.
    confidence_threshold : float
        Minimum consensus confidence (0-100) to pass the gate.  Default 60.
    default_prior : float
        Base prior probability for Bayesian update when no regime info
        is supplied.  Default 0.5.
    max_loss_per_trade : float
        Maximum acceptable loss per trade as fraction of capital, used
        in risk-of-ruin sanity checks.  Default 0.02 (2 %).
    """

    def __init__(
        self,
        min_agents: int = 3,
        confidence_threshold: float = 60.0,
        default_prior: float = 0.5,
        max_loss_per_trade: float = 0.02,
    ) -> None:
        self.min_agents = min_agents
        self.confidence_threshold = confidence_threshold
        self.default_prior = default_prior
        self.max_loss_per_trade = max_loss_per_trade
        self._bayesian = BayesianUpdater()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        agent_results: List[AgentResult],
        regime: Optional[str] = None,
        session_bias: Optional[float] = None,
    ) -> ConsensusDecision:
        """
        Evaluate a set of agent results and produce a consensus decision.

        Parameters
        ----------
        agent_results : list[AgentResult]
            Outputs from each analysis agent.
        regime : str, optional
            Market regime label (e.g. "trending", "ranging", "volatile")
            used to select the Bayesian prior.
        session_bias : float, optional
            Override prior probability directly (0-1).

        Returns
        -------
        ConsensusDecision
        """
        if not agent_results:
            return self._no_trade(["No agent results provided"])

        # Split by direction
        buys = [r for r in agent_results if r.direction == Direction.BUY]
        sells = [r for r in agent_results if r.direction == Direction.SELL]

        # Weighted scores
        buy_score = self._weighted_score(buys)
        sell_score = self._weighted_score(sells)

        # Determine dominant direction
        if buy_score > sell_score:
            dominant = Direction.BUY
            supporters = buys
            conflicts = sells
        elif sell_score > buy_score:
            dominant = Direction.SELL
            supporters = sells
            conflicts = buys
        else:
            return self._no_trade(["Equal buy/sell scores — no dominant direction"])

        # Minimum-agent gate
        if len(supporters) < self.min_agents:
            return self._no_trade([
                f"Only {len(supporters)} agent(s) agree on {dominant.value} "
                f"(need {self.min_agents})"
            ])

        # Confidence — weighted average of supporting agents' confidence
        total_weight = sum(a.weight for a in supporters)
        if total_weight == 0:
            consensus_confidence = 0.0
        else:
            consensus_confidence = sum(
                a.confidence * a.weight for a in supporters
            ) / total_weight

        # Normalise confidence by agreement ratio
        agreement_ratio = len(supporters) / len(agent_results)
        consensus_confidence *= agreement_ratio

        # Bayesian update
        prior = self._resolve_prior(regime, session_bias)
        likelihoods = [self._agent_likelihood(a, dominant) for a in supporters]
        posterior = self._bayesian.update_posterior(
            prior=prior, likelihoods=likelihoods
        )
        # Incorporate posterior into confidence
        consensus_confidence = min(
            100.0, consensus_confidence * (0.7 + 0.3 * posterior)
        )

        # Expected Value
        ev_data = self._compute_ev(supporters, dominant)
        ev = ev_data["ev"]
        kelly = ev_data["kelly"]

        reasons: List[str] = []

        # Gate: confidence & EV
        if consensus_confidence < self.confidence_threshold:
            reasons.append(
                f"Confidence {consensus_confidence:.1f}% < "
                f"threshold {self.confidence_threshold}%"
            )
        if ev <= 0:
            reasons.append(f"Expected value {ev:.4f} ≤ 0")

        if reasons:
            return ConsensusDecision(
                direction=Direction.NO_TRADE,
                confidence=consensus_confidence,
                expected_value=ev,
                kelly_fraction=kelly,
                supporting_agents=[a.agent_name for a in supporters],
                conflicting_agents=[a.agent_name for a in conflicts],
                total_agents=len(agent_results),
                agreeing_agents=len(supporters),
                prior_probability=prior,
                posterior_probability=posterior,
                reasons=reasons,
            )

        return ConsensusDecision(
            direction=dominant,
            confidence=consensus_confidence,
            expected_value=ev,
            kelly_fraction=kelly,
            supporting_agents=[a.agent_name for a in supporters],
            conflicting_agents=[a.agent_name for a in conflicts],
            total_agents=len(agent_results),
            agreeing_agents=len(supporters),
            prior_probability=prior,
            posterior_probability=posterior,
            reasons=["Consensus reached"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_score(agents: List[AgentResult]) -> float:
        """Sum of (score × weight) for a group of agents."""
        return sum(a.score * a.weight for a in agents)

    @staticmethod
    def _agent_likelihood(agent: AgentResult, direction: Direction) -> float:
        """
        Convert an agent result into a likelihood P(data | direction).

        Uses a combination of score and confidence, clamped to (0, 1).
        """
        if agent.direction != direction:
            return 1.0 - (agent.score / 100.0)  # low likelihood
        raw = (agent.score / 100.0 * 0.6) + (agent.confidence / 100.0 * 0.4)
        return float(np.clip(raw, 0.05, 0.99))

    def _resolve_prior(
        self,
        regime: Optional[str],
        session_bias: Optional[float],
    ) -> float:
        """Return the Bayesian prior probability."""
        if session_bias is not None:
            return float(np.clip(session_bias, 0.01, 0.99))
        return self._bayesian.get_regime_prior(regime)

    def _compute_ev(
        self,
        supporters: List[AgentResult],
        direction: Direction,
    ) -> dict:
        """
        Aggregate historical stats from supporters and compute EV.

        Falls back to moderate defaults when agents lack history.
        """
        win_rates = [
            a.historical_win_rate
            for a in supporters
            if a.historical_win_rate is not None
        ]
        avg_wins = [
            a.historical_avg_win
            for a in supporters
            if a.historical_avg_win is not None
        ]
        avg_losses = [
            a.historical_avg_loss
            for a in supporters
            if a.historical_avg_loss is not None
        ]

        # Weighted average using agent weights
        if win_rates:
            weights = [
                a.weight
                for a in supporters
                if a.historical_win_rate is not None
            ]
            win_rate = float(
                np.average(win_rates, weights=weights)
            )
        else:
            win_rate = 0.55  # moderate default

        if avg_wins:
            weights = [
                a.weight
                for a in supporters
                if a.historical_avg_win is not None
            ]
            avg_win = float(np.average(avg_wins, weights=weights))
        else:
            avg_win = 1.5  # 1.5R average win

        if avg_losses:
            weights = [
                a.weight
                for a in supporters
                if a.historical_avg_loss is not None
            ]
            avg_loss = float(np.average(avg_losses, weights=weights))
        else:
            avg_loss = 1.0  # 1R average loss

        return calculate_ev(win_rate, avg_win, avg_loss)

    def _no_trade(self, reasons: List[str]) -> ConsensusDecision:
        """Shortcut to build a NO_TRADE decision."""
        return ConsensusDecision(
            direction=Direction.NO_TRADE,
            confidence=0.0,
            expected_value=0.0,
            kelly_fraction=0.0,
            supporting_agents=[],
            conflicting_agents=[],
            total_agents=0,
            agreeing_agents=0,
            prior_probability=self.default_prior,
            posterior_probability=self.default_prior,
            reasons=reasons,
        )
