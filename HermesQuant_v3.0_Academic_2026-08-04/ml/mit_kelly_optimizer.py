"""
MIT Kelly Criterion Optimizer — John Kelly (Bell Labs)
Optimal bet sizing based on edge and odds
"""
import numpy as np


class KellyOptimizer:
    """
    Kelly Criterion for optimal position sizing:
    f* = (bp - q) / b
    
    Where:
    - f* = fraction of bankroll to bet
    - b = odds (win/loss ratio)
    - p = probability of winning
    - q = 1 - p = probability of losing
    
    Key insight: Kelly maximizes long-term growth rate
    """
    
    def __init__(self, kelly_fraction=0.25):
        """
        Use fractional Kelly (quarter-Kelly) for safety
        """
        self.kelly_fraction = kelly_fraction
    
    def optimal_bet(self, win_rate, avg_win, avg_loss):
        """
        Calculate optimal bet size
        """
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0
        
        b = avg_win / avg_loss  # odds ratio
        p = win_rate
        q = 1 - p
        
        # Kelly fraction
        f_star = (b * p - q) / b
        
        # Apply fractional Kelly
        f_optimal = f_star * self.kelly_fraction
        
        # Clamp to reasonable range
        f_optimal = max(0, min(f_optimal, 0.25))
        
        return f_optimal
    
    def calculate_position_size(self, capital, win_rate, avg_win, avg_loss, max_risk=0.03):
        """
        Calculate position size in dollars
        """
        kelly = self.optimal_bet(win_rate, avg_win, avg_loss)
        
        # Convert to dollar amount
        position = capital * kelly
        
        # Apply max risk constraint
        max_position = capital * max_risk
        position = min(position, max_position)
        
        return {
            "kelly_fraction": kelly,
            "position_size": position,
            "capital": capital,
            "expected_return": win_rate * avg_win - (1 - win_rate) * avg_loss,
        }
    
    def simulate_growth(self, capital, win_rate, avg_win, avg_loss, trades=100):
        """
        Simulate account growth with Kelly sizing
        """
        balance = capital
        history = [balance]
        
        kelly = self.optimal_bet(win_rate, avg_win, avg_loss)
        
        for _ in range(trades):
            if np.random.random() < win_rate:
                balance *= (1 + kelly * avg_win)
            else:
                balance *= (1 - kelly * avg_loss)
            
            history.append(balance)
        
        return history
