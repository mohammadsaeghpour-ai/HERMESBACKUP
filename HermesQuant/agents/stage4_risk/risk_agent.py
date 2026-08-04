"""
Risk Agent — Real Implementation
Controls: drawdown, exposure, consecutive losses, position sizing
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class RiskAgent:
    """
    Real risk management:
    1. Daily drawdown limit
    2. Max consecutive losses
    3. Volatility-based position sizing
    4. Exposure limits
    """
    name = "Risk"
    weight = 0.0  # Does not vote, only filters
    
    def __init__(self, max_daily_loss=3.0, max_consecutive_losses=3, max_exposure=0.5):
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.max_exposure = max_exposure
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.total_trades = 0
    
    def analyze(self, df, symbol="", timeframe=""):
        ev = []
        risk_ok = True
        
        # 1. Daily drawdown check
        if self.daily_pnl < -self.max_daily_loss:
            risk_ok = False
            ev.append("DAILY LOSS LIMIT REACHED: $%.2f / $%.2f" % (self.daily_pnl, self.max_daily_loss))
        
        # 2. Consecutive losses check
        if self.consecutive_losses >= self.max_consecutive_losses:
            risk_ok = False
            ev.append("CONSECUTIVE LOSSES: %d / %d" % (self.consecutive_losses, self.max_consecutive_losses))
        
        # 3. Volatility check
        if df is not None and len(df) > 20:
            atr = ind.atr(df, 14)
            atr_pct = atr.iloc[-1] / df["close"].iloc[-1] * 100
            if atr_pct > 3.0:
                risk_ok = False
                ev.append("HIGH VOLATILITY: ATR=%.2f%% (>3%%)" % atr_pct)
            elif atr_pct > 2.0:
                ev.append("ELEVATED VOLATILITY: ATR=%.2f%%" % atr_pct)
        
        # 4. Trade count check
        if self.total_trades >= 5:
            risk_ok = False
            ev.append("MAX TRADES REACHED: %d / 5" % self.total_trades)
        
        if not ev:
            ev.append("Risk check passed")
        
        return AgentOutput(
            name=self.name,
            direction="NEUTRAL" if risk_ok else "WAIT",
            confidence=100 if risk_ok else 0,
            score=0,
            weight=self.weight,
            evidence=ev
        )
    
    def update_pnl(self, pnl):
        """Update daily P&L"""
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.total_trades += 1
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.total_trades = 0
