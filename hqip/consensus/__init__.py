"""
HQIP v3 — Consensus Engine

Multi-agent signal aggregation with Bayesian updating,
expected value filtering, and minimum-agreement gates.
"""

from .engine import ConsensusEngine, AgentResult, ConsensusDecision
from .bayesian import BayesianUpdater, posterior_probability, update_posterior
from .expected_value import calculate_ev, kelly_criterion, minimum_viable_ev

__all__ = [
    "ConsensusEngine",
    "AgentResult",
    "ConsensusDecision",
    "BayesianUpdater",
    "posterior_probability",
    "update_posterior",
    "calculate_ev",
    "kelly_criterion",
    "minimum_viable_ev",
]
