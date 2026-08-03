"""
Bayesian probability updater for consensus decisions.

Updates prior beliefs about trade direction using agent likelihoods
via Bayes' theorem:  P(H|D) = P(D|H) · P(H) / P(D).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# Regime-to-prior mapping — tunable knobs
_REGIME_Priors: Dict[str, float] = {
    "trending": 0.60,      # trending markets slightly favour continuation
    "ranging": 0.45,       # range-bound markets are noisier
    "volatile": 0.40,      # high vol = lower conviction
    "breakout": 0.55,      # breakout setups have a slight edge
    "low_vol": 0.50,       # low-vol defaults to neutral
}


class BayesianUpdater:
    """
    Stateful Bayesian probability updater.

    Maintains a running posterior that can be updated incrementally
    as new agent evidence arrives.
    """

    def __init__(self, default_prior: float = 0.5) -> None:
        self.default_prior = default_prior
        self._current_posterior: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_regime_prior(regime: Optional[str]) -> float:
        """
        Return a prior probability based on market regime.

        Parameters
        ----------
        regime : str or None
            Regime label.  Falls back to 0.5 when unknown.

        Returns
        -------
        float
            Prior in (0, 1).
        """
        if regime is None:
            return 0.5
        return _REGIME_Priors.get(regime.lower(), 0.5)

    def update_posterior(
        self,
        prior: float,
        likelihoods: List[float],
    ) -> float:
        """
        Compute posterior probability from a prior and a list of
        likelihoods using Bayes' theorem.

        P(H|D₁…Dₙ) = P(D₁…Dₙ|H) · P(H) / P(D₁…Dₙ)

        We assume conditional independence of agents, so
        P(D₁…Dₙ|H) = ∏ P(Dᵢ|H).

        The evidence (normalising constant) is computed via total
        probability:  P(D) = P(D|H)·P(H) + P(D|¬H)·P(¬H).

        Parameters
        ----------
        prior : float
            Prior probability P(H) in (0, 1).
        likelihoods : list[float]
            Each element is P(Dᵢ|H) — the likelihood of agent i's
            observation given the hypothesis is true.  Values in (0, 1).

        Returns
        -------
        float
            Posterior probability clamped to [0.01, 0.99].
        """
        if not likelihoods:
            return prior

        prior = float(np.clip(prior, 0.01, 0.99))

        # Product of likelihoods (conditional independence)
        likelihood_product = float(np.prod([np.clip(l, 0.01, 0.99) for l in likelihoods]))

        # Complementary likelihoods (for ¬H)
        comp_product = float(np.prod([np.clip(1.0 - l, 0.01, 0.99) for l in likelihoods]))

        # Evidence via total probability
        evidence = (likelihood_product * prior) + (comp_product * (1.0 - prior))

        if evidence <= 0:
            return prior

        posterior = (likelihood_product * prior) / evidence
        posterior = float(np.clip(posterior, 0.01, 0.99))

        self._current_posterior = posterior
        return posterior

    @property
    def current_posterior(self) -> Optional[float]:
        """Most recently computed posterior, or None."""
        return self._current_posterior

    def reset(self) -> None:
        """Reset the stored posterior."""
        self._current_posterior = None

    def sequential_update(
        self,
        prior: float,
        new_likelihood: float,
    ) -> float:
        """
        One-step Bayesian update: take the stored posterior (or the
        provided prior) and incorporate a single new likelihood.

        Parameters
        ----------
        prior : float
            Starting prior (used only if no stored posterior exists).
        new_likelihood : float
            P(D|H) for the new piece of evidence.

        Returns
        -------
        float
            Updated posterior.
        """
        base = self._current_posterior if self._current_posterior is not None else prior
        return self.update_posterior(prior=base, likelihoods=[new_likelihood])


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def posterior_probability(prior: float, likelihoods: List[float]) -> float:
    """
    One-shot Bayesian update without state.

    Parameters
    ----------
    prior : float
        Prior probability P(H).
    likelihoods : list[float]
        List of P(Dᵢ|H) values.

    Returns
    -------
    float
        Posterior probability.
    """
    updater = BayesianUpdater()
    return updater.update_posterior(prior, likelihoods)


def update_posterior(prior: float, evidence_list: List[float]) -> float:
    """
    Alias for ``posterior_probability`` matching the original spec.

    Parameters
    ----------
    prior : float
        Prior probability.
    evidence_list : list[float]
        Likelihood values from each agent.

    Returns
    -------
    float
        Posterior probability.
    """
    return posterior_probability(prior, evidence_list)
