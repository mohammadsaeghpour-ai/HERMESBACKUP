"""
Portfolio Risk Manager — Multi-asset risk control
Controls: drawdown, correlation, max exposure
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")


class PortfolioRiskManager:
    """
    Portfolio-level risk management:
    1. Daily drawdown limit
    2. Correlation-aware position sizing
    3. Max simultaneous exposure
    4. Consecutive loss protection
    """
    
    def __init__(self, max_daily_loss=3.0, max_positions=2, 
                 max_correlated_exposure=0.7, max_consecutive_losses=3):
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.max_correlated_exposure = max_correlated_exposure
        self.max_consecutive_losses = max_consecutive_losses
        
        # State
        self.daily_pnl = 0
        self.open_positions = {}  # symbol -> direction
        self.consecutive_losses = 0
        self.trade_history = []
    
    def can_trade(self, symbol, direction, size):
        """Check if trade is allowed"""
        reasons = []
        
        # 1. Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            reasons.append("Daily loss limit reached: $%.2f" % self.daily_pnl)
        
        # 2. Max positions
        if len(self.open_positions) >= self.max_positions:
            if symbol not in self.open_positions:
                reasons.append("Max positions reached: %d/%d" % (len(self.open_positions), self.max_positions))
        
        # 3. Correlated exposure
        if symbol in self.open_positions:
            if self.open_positions[symbol] == direction:
                reasons.append("Already have same direction on %s" % symbol)
        else:
            # Check correlation with existing positions
            for existing_sym, existing_dir in self.open_positions.items():
                if existing_dir == direction:
                    # BTC and ETH are ~0.8 correlated
                    if ("BTC" in symbol and "ETH" in existing_sym) or                        ("ETH" in symbol and "BTC" in existing_sym):
                        reasons.append("Correlated exposure: %s and %s both %s" % (symbol, existing_sym, direction))
        
        # 4. Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            reasons.append("Consecutive losses: %d/%d" % (self.consecutive_losses, self.max_consecutive_losses))
        
        return len(reasons) == 0, reasons
    
    def update(self, symbol, direction, pnl):
        """Update state after trade"""
        self.daily_pnl += pnl
        self.trade_history.append({"symbol": symbol, "direction": direction, "pnl": pnl})
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Track open positions
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        else:
            self.open_positions[symbol] = direction
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
        self.consecutive_losses = 0
    
    def get_status(self):
        """Get current risk status"""
        return {
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "can_trade": self.daily_pnl > -self.max_daily_loss,
        }
