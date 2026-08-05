"""
MIT Risk Parity — Ray Dalio (Bridgewater)
Equal risk contribution from each asset
"""
import numpy as np
import pandas as pd


class RiskParity:
    """
    Risk Parity portfolio optimization:
    1. Calculate risk contribution of each signal
    2. Equalize risk across signals
    3. Dynamic rebalancing based on volatility
    
    Key insight: Diversify by RISK, not by allocation
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def calculate_risk_contribution(self, returns, weights):
        """
        Calculate risk contribution of each signal
        """
        cov = np.cov(returns.T)
        vol = np.sqrt(np.diag(cov))
        
        # Portfolio volatility
        port_vol = np.sqrt(weights @ cov @ weights)
        
        # Risk contribution
        risk_contrib = (weights * vol) / port_vol
        
        return risk_contrib
    
    def optimize_weights(self, returns):
        """
        Find weights that equalize risk contribution
        """
        n_signals = returns.shape[1]
        
        # Initial equal weights
        weights = np.ones(n_signals) / n_signals
        
        # Iterative optimization
        for _ in range(100):
            risk_contrib = self.calculate_risk_contribution(returns, weights)
            
            # Target: equal risk contribution
            target_risk = 1.0 / n_signals
            
            # Adjust weights inversely proportional to risk
            weights = weights * (target_risk / (risk_contrib + 1e-10))
            
            # Normalize
            weights = weights / weights.sum()
        
        return weights
    
    def dynamic_rebalance(self, current_weights, recent_volatilities, target_vol=0.15):
        """
        Dynamic rebalancing based on recent volatility
        """
        # Inverse volatility weighting
        inv_vol = 1.0 / (recent_volatilities + 1e-10)
        new_weights = inv_vol / inv_vol.sum()
        
        # Blend with current weights (50/50)
        blended = 0.5 * current_weights + 0.5 * new_weights
        
        # Apply target volatility scaling
        current_vol = np.sqrt(new_weights @ np.diag(recent_volatilities**2) @ new_weights)
        scale = target_vol / (current_vol + 1e-10)
        scale = min(scale, 2.0)  # Cap at 2x
        
        return blended * scale
