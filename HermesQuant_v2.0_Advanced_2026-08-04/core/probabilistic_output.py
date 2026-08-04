"""
Probabilistic Output — Distribution + Uncertainty
Instead of BUY/SELL, output probability distribution
"""
import numpy as np

class ProbabilisticOutput:
    """
    Convert ensemble signals to probability distribution.
    Output: {
        direction: BUY/SELL/WAIT,
        probability: 0.0-1.0,
        uncertainty: 0.0-1.0 (high = less confident),
        expected_return: float,
        confidence_interval: (lower, upper),
    }
    """
    
    @staticmethod
    def compute(agents, bayes_p_up, bayes_p_down, convergence, atr):
        """Compute probabilistic output from agent signals"""
        
        # Base probability from Bayesian combine
        p_up = bayes_p_up
        p_down = bayes_p_down
        
        # Uncertainty from convergence (low convergence = high uncertainty)
        uncertainty = 1.0 - convergence
        
        # Direction
        if p_up > 0.6:
            direction = "BUY"
            prob = p_up
        elif p_down > 0.6:
            direction = "SELL"
            prob = p_down
        else:
            direction = "WAIT"
            prob = max(p_up, p_down)
        
        # Expected return (rough estimate)
        if direction == "BUY":
            expected_return = (prob - 0.5) * atr * 2
        elif direction == "SELL":
            expected_return = (prob - 0.5) * atr * 2
        else:
            expected_return = 0
        
        # Confidence interval (based on ATR and uncertainty)
        margin = atr * (1 + uncertainty)
        ci_lower = -margin
        ci_upper = margin
        
        return {
            "direction": direction,
            "probability": round(prob, 3),
            "uncertainty": round(uncertainty, 3),
            "expected_return": round(expected_return, 2),
            "confidence_interval": (round(ci_lower, 2), round(ci_upper, 2)),
            "p_up": round(p_up, 3),
            "p_down": round(p_down, 3),
        }
