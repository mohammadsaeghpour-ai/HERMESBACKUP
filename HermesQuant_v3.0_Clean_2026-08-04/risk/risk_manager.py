"""
Risk Manager — Kelly + Volatility Targeting + Drawdown Control
"""
import numpy as np
import pandas as pd


class RiskManager:
    """Advanced risk management"""
    
    def __init__(self, capital=10.0, max_leverage=20.0, max_daily_loss=3.0):
        self.capital = capital
        self.max_leverage = max_leverage
        self.max_daily_loss = max_daily_loss
        self.peak_capital = capital
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
    
    def kelly_size(self, win_rate, avg_win, avg_loss):
        """Quarter-Kelly for safety"""
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        return max(0, min(kelly * 0.25, 0.25))
    
    def vol_targeting(self, returns, target_vol=0.10):
        """Adjust position size based on volatility"""
        if returns is None or len(returns) < 20:
            return 1.0
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(365)
        if current_vol == 0 or pd.isna(current_vol):
            return 1.0
        return min(target_vol / current_vol, 2.0)
    
    def check_drawdown(self):
        """Check if drawdown exceeds limits"""
        self.peak_capital = max(self.peak_capital, self.capital + self.daily_pnl)
        current = self.capital + self.daily_pnl
        dd = (self.peak_capital - current) / self.peak_capital
        return dd < 0.05  # 5% daily limit
    
    def check_circuit_breaker(self):
        """FeneFX rules: 3 consecutive losses = stop 4h"""
        if self.consecutive_losses >= 3:
            return False  # Stop trading
        return True
    
    def calculate_position(self, signal_strength, kelly_fraction, vol_scalar, meta_confidence=1.0):
        """Final position size"""
        if not self.check_drawdown():
            return 0
        
        if not self.check_circuit_breaker():
            return 0
        
        base = self.capital * 0.02  # 2% risk per trade
        size = base * signal_strength * meta_confidence * kelly_fraction * vol_scalar
        max_pos = self.capital * 0.1  # 10% max position
        return min(size, max_pos)
    
    def update(self, pnl):
        """Update after trade"""
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
    
    def reset_daily(self):
        """Reset daily stats"""
        self.daily_pnl = 0
        self.consecutive_losses = 0
    
    def get_stats(self):
        """Get current stats"""
        win_rate = self.wins / max(self.total_trades, 1)
        return {
            "capital": self.capital + self.daily_pnl,
            "daily_pnl": self.daily_pnl,
            "win_rate": win_rate,
            "total_trades": self.total_trades,
            "consecutive_losses": self.consecutive_losses,
        }
