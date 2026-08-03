"""
HQIP Backtester — Event-driven backtesting engine
"""
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    quantity: float
    margin: float
    pnl: float
    pnl_pct: float
    reason: str  # 'TP1', 'TP2', 'TP3', 'SL', 'TIMEOUT'
    entry_time: str
    exit_time: str
    candles_held: int

@dataclass
class BacktestResult:
    trades: List[BacktestTrade]
    equity_curve: List[float]
    metrics: dict
    initial_capital: float
    final_capital: float

class Backtester:
    """Event-driven backtesting engine with multi-timeframe support."""
    
    def __init__(self, capital: float = 100.0, leverage: int = 20,
                 risk_pct: float = 0.08, max_trades_day: int = 6,
                 min_confidence: float = 60):
        self.initial_capital = capital
        self.capital = capital
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.max_trades_day = max_trades_day
        self.min_confidence = min_confidence
    
    def run(self, df: pd.DataFrame, signal_func: Callable,
            sl_mult: float = 1.5, tp1_mult: float = 2.0,
            tp2_mult: float = 3.0, tp3_mult: float = 4.0,
            lookback: int = 200, step: int = 1) -> BacktestResult:
        """
        Run backtest on a single timeframe.
        
        Args:
            df: OHLCV DataFrame
            signal_func: function(df_slice) -> dict(direction, confidence, atr)
            sl_mult/tp_mult: ATR multipliers for SL/TP
            lookback: candles for indicator warmup
            step: candles to advance per iteration
        """
        trades = []
        equity = [self.initial_capital]
        cap = self.initial_capital
        peak = cap
        daily_count = {}
        position = None
        
        for i in range(lookback, len(df) - 20, step):
            ts = df.index[i] if hasattr(df.index[i], 'date') else str(i)
            day = str(ts)[:10]
            
            # Check daily trade limit
            if daily_count.get(day, 0) >= self.max_trades_day:
                equity.append(cap)
                continue
            
            # Check if we have a position
            if position is not None:
                # Check SL/TP on current candle
                result = self._check_position(position, df.iloc[i])
                if result:
                    cap += result['pnl'] + position['margin']
                    trades.append(result['trade'])
                    position = None
                    if cap > peak: peak = cap
                equity.append(cap)
                continue
            
            # Generate signal
            try:
                signal = signal_func(df.iloc[max(0,i-lookback):i+1])
            except:
                equity.append(cap)
                continue
            
            if not signal or signal.get('confidence', 0) < self.min_confidence:
                equity.append(cap)
                continue
            
            direction = signal.get('direction', '')
            if direction not in ('BUY', 'SELL'):
                equity.append(cap)
                continue
            
            atr = signal.get('atr', df.iloc[i]['close'] * 0.01)
            entry = float(df.iloc[i]['close'])
            
            # Calculate SL/TP
            if direction == 'BUY':
                sl = entry - sl_mult * atr
                tp1 = entry + tp1_mult * atr
                tp2 = entry + tp2_mult * atr
                tp3 = entry + tp3_mult * atr
            else:
                sl = entry + sl_mult * atr
                tp1 = entry - tp1_mult * atr
                tp2 = entry - tp2_mult * atr
                tp3 = entry - tp3_mult * atr
            
            # Position sizing
            risk_amount = cap * self.risk_pct
            sl_distance = abs(entry - sl)
            if sl_distance == 0:
                equity.append(cap)
                continue
            
            qty = risk_amount / sl_distance
            margin = qty * entry / self.leverage
            if margin > cap * 0.9:
                margin = cap * 0.9
                qty = margin * self.leverage / entry
            
            position = {
                'direction': direction, 'entry': entry, 'sl': sl,
                'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                'qty': qty, 'margin': margin,
                'entry_time': str(ts), 'candle_idx': i,
                'tp1_hit': False, 'tp2_hit': False
            }
            daily_count[day] = daily_count.get(day, 0) + 1
            equity.append(cap)
        
        # Close remaining position
        if position and len(df) > 0:
            close_price = float(df.iloc[-1]['close'])
            if position['direction'] == 'BUY':
                pnl = (close_price - position['entry']) * position['qty']
            else:
                pnl = (position['entry'] - close_price) * position['qty']
            pnl = max(pnl, -position['margin'])
            cap += position['margin'] + pnl
            trades.append(BacktestTrade(
                symbol='', direction=position['direction'],
                entry_price=position['entry'], exit_price=close_price,
                sl=position['sl'], tp1=position['tp1'],
                tp2=position['tp2'], tp3=position['tp3'],
                quantity=position['qty'], margin=position['margin'],
                pnl=pnl, pnl_pct=pnl/position['margin']*100,
                reason='TIMEOUT', entry_time=position['entry_time'],
                exit_time=str(df.index[-1]), candles_held=0
            ))
        
        equity.append(cap)
        metrics = self._calculate_metrics(trades, equity)
        
        return BacktestResult(
            trades=trades, equity_curve=equity, metrics=metrics,
            initial_capital=self.initial_capital, final_capital=cap
        )
    
    def _check_position(self, pos: dict, candle) -> Optional[dict]:
        """Check if SL/TP hit on current candle."""
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        
        if pos['direction'] == 'BUY':
            if low <= pos['sl']:
                pnl = max(-pos['margin'], (pos['sl'] - pos['entry']) * pos['qty'])
                return self._make_trade(pos, pos['sl'], 'SL', pnl)
            if high >= pos['tp3']:
                pnl = (pos['tp3'] - pos['entry']) * pos['qty']
                return self._make_trade(pos, pos['tp3'], 'TP3', pnl)
            if high >= pos['tp2']:
                if not pos['tp2_hit']:
                    pos['tp2_hit'] = True
                    pos['sl'] = pos['entry']  # Breakeven
            if high >= pos['tp1']:
                if not pos['tp1_hit']:
                    pos['tp1_hit'] = True
                    pos['sl'] = pos['entry']  # Breakeven
        else:
            if high >= pos['sl']:
                pnl = max(-pos['margin'], (pos['entry'] - pos['sl']) * pos['qty'])
                return self._make_trade(pos, pos['sl'], 'SL', pnl)
            if low <= pos['tp3']:
                pnl = (pos['entry'] - pos['tp3']) * pos['qty']
                return self._make_trade(pos, pos['tp3'], 'TP3', pnl)
            if low <= pos['tp2']:
                if not pos['tp2_hit']:
                    pos['tp2_hit'] = True
                    pos['sl'] = pos['entry']
            if low <= pos['tp1']:
                if not pos['tp1_hit']:
                    pos['tp1_hit'] = True
                    pos['sl'] = pos['entry']
        
        return None
    
    def _make_trade(self, pos, exit_price, reason, pnl):
        pnl -= abs(pnl) * 0.001  # Fee
        return {
            'trade': BacktestTrade(
                symbol='', direction=pos['direction'],
                entry_price=pos['entry'], exit_price=exit_price,
                sl=pos['sl'], tp1=pos['tp1'], tp2=pos['tp2'], tp3=pos['tp3'],
                quantity=pos['qty'], margin=pos['margin'],
                pnl=pnl, pnl_pct=pnl/pos['margin']*100,
                reason=reason, entry_time=pos['entry_time'],
                exit_time='', candles_held=0
            ),
            'pnl': pnl
        }
    
    def _calculate_metrics(self, trades, equity) -> dict:
        """Calculate comprehensive metrics."""
        if not trades:
            return {'total_trades': 0}
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        eq = np.array(equity)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        total_return = (eq[-1] - eq[0]) / eq[0] * 100
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0
        
        neg_returns = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(neg_returns) + 1e-10) * np.sqrt(252) if len(neg_returns) > 0 else 0
        
        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        
        return {
            'total_trades': len(trades),
            'total_return_pct': total_return,
            'win_rate': len(wins) / len(pnls) * 100,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'max_consecutive_losses': max_consec,
            'expectancy': np.mean(pnls),
            'tp1_hits': sum(1 for t in trades if t.reason == 'TP1'),
            'tp2_hits': sum(1 for t in trades if t.reason == 'TP2'),
            'tp3_hits': sum(1 for t in trades if t.reason == 'TP3'),
            'sl_hits': sum(1 for t in trades if t.reason == 'SL'),
            'timeouts': sum(1 for t in trades if t.reason == 'TIMEOUT'),
        }
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate formatted report."""
        m = result.metrics
        if m.get('total_trades', 0) == 0:
            return "No trades generated."
        
        lines = [
            "=" * 50,
            "  HQIP BACKTEST REPORT",
            "=" * 50,
            f"  Start Capital:  ${result.initial_capital:.2f}",
            f"  End Capital:    ${result.final_capital:.2f}",
            f"  Total Return:   {m['total_return_pct']:+.1f}%",
            f"  Total Trades:   {m['total_trades']}",
            f"  Win Rate:       {m['win_rate']:.1f}%",
            f"  Avg Win:        ${m['avg_win']:.2f}",
            f"  Avg Loss:       ${m['avg_loss']:.2f}",
            f"  Profit Factor:  {m['profit_factor']:.2f}",
            f"  Sharpe Ratio:   {m['sharpe_ratio']:.2f}",
            f"  Sortino Ratio:  {m['sortino_ratio']:.2f}",
            f"  Max Drawdown:   {m['max_drawdown_pct']:.1f}%",
            f"  Max Consec Loss:{m['max_consecutive_losses']}",
            f"  Expectancy:     ${m['expectancy']:.2f}",
            f"  --- Exit Types ---",
            f"  TP1: {m.get('tp1_hits',0)} | TP2: {m.get('tp2_hits',0)} | TP3: {m.get('tp3_hits',0)}",
            f"  SL:  {m.get('sl_hits',0)} | TIMEOUT: {m.get('timeouts',0)}",
            "=" * 50
        ]
        return "\n".join(lines)


def monte_carlo_simulation(trades: List[BacktestTrade], n_simulations: int = 1000,
                           n_trades: int = 100) -> dict:
    """Run Monte Carlo simulation on trade results."""
    if not trades:
        return {}
    
    pnls = np.array([t.pnl for t in trades])
    final_capitals = []
    
    for _ in range(n_simulations):
        sampled = np.random.choice(pnls, size=n_trades, replace=True)
        final = 100 + np.cumsum(sampled)
        final_capitals.append(final[-1])
    
    final_capitals = np.array(final_capitals)
    
    return {
        'percentiles': {
            '5%': float(np.percentile(final_capitals, 5)),
            '25%': float(np.percentile(final_capitals, 25)),
            '50%': float(np.percentile(final_capitals, 50)),
            '75%': float(np.percentile(final_capitals, 75)),
            '95%': float(np.percentile(final_capitals, 95)),
        },
        'probability_of_profit': float(np.mean(final_capitals > 100) * 100),
        'probability_of_ruin': float(np.mean(final_capitals < 10) * 100),
        'best_case': float(np.max(final_capitals)),
        'worst_case': float(np.min(final_capitals)),
        'expected_case': float(np.mean(final_capitals)),
    }
