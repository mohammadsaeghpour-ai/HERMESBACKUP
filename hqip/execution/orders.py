"""
HQIP Execution Engine — Order Management + Paper Trading
"""
import ccxt
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[float] = None
    status: str = 'pending'
    filled_price: float = 0.0
    filled_at: str = ''
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    pnl: float = 0.0

@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    entry_price: float
    current_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: str = ''

class OrderManager:
    """Multi-exchange order manager with retry logic."""
    
    def __init__(self, exchange_name='okx', paper=True):
        self.paper = paper
        self.exchange = None
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[dict] = []
        
        if not paper:
            try:
                if exchange_name == 'okx':
                    self.exchange = ccxt.okx({'enableRateLimit': True})
                elif exchange_name == 'binance':
                    self.exchange = ccxt.binanceusdm({'enableRateLimit': True})
                elif exchange_name == 'bybit':
                    self.exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
                self.exchange.load_markets()
                print(f"✅ Connected to {exchange_name}")
            except Exception as e:
                print(f"⚠️ Connection failed: {e}, using paper trading")
                self.paper = True
    
    def create_order(self, symbol: str, side: str, qty: float, order_type: str = 'market',
                     price: float = None, sl: float = None, tp1: float = None,
                     tp2: float = None, tp3: float = None) -> dict:
        """Create a new order."""
        order_id = f"ORD_{int(time.time()*1000)}"
        
        if self.paper:
            # Paper trading — instant fill
            fill_price = price if price else self._get_sim_price(symbol)
            if side == 'buy':
                fill_price *= 1.0001  # Simulate slippage
            else:
                fill_price *= 0.9999
            
            order = Order(
                id=order_id, symbol=symbol, side=side, qty=qty,
                order_type=order_type, price=price, status='filled',
                filled_price=fill_price, filled_at=datetime.utcnow().isoformat(),
                sl=sl, tp1=tp1, tp2=tp2, tp3=tp3
            )
            self.orders[order_id] = order
            
            # Update position
            self.positions[symbol] = Position(
                symbol=symbol, side=side, qty=qty, entry_price=fill_price,
                current_price=fill_price, sl=sl or 0, tp1=tp1 or 0,
                tp2=tp2 or 0, tp3=tp3 or 0,
                opened_at=datetime.utcnow().isoformat()
            )
            
            return {'order_id': order_id, 'status': 'filled', 'price': fill_price}
        
        # Live trading
        try:
            for attempt in range(3):
                try:
                    if order_type == 'market':
                        result = self.exchange.create_order(symbol, 'market', side, qty)
                    elif order_type == 'limit':
                        result = self.exchange.create_order(symbol, 'limit', side, qty, price)
                    else:
                        result = self.exchange.create_order(symbol, 'market', side, qty)
                    
                    order = Order(
                        id=result['id'], symbol=symbol, side=side, qty=qty,
                        order_type=order_type, price=price, status='filled',
                        filled_price=float(result.get('average', price or 0)),
                        filled_at=datetime.utcnow().isoformat(),
                        sl=sl, tp1=tp1, tp2=tp2, tp3=tp3
                    )
                    self.orders[order.id] = order
                    return {'order_id': order.id, 'status': 'filled', 'price': order.filled_price}
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        return {'order_id': None, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            return {'order_id': None, 'status': 'error', 'error': str(e)}
    
    def close_position(self, symbol: str) -> dict:
        """Close an open position."""
        if symbol not in self.positions:
            return {'status': 'no_position'}
        
        pos = self.positions[symbol]
        close_side = 'sell' if pos.side == 'buy' else 'buy'
        result = self.create_order(symbol, close_side, pos.qty, 'market')
        
        if result['status'] == 'filled':
            if pos.side == 'buy':
                pnl = (result['price'] - pos.entry_price) * pos.qty
            else:
                pnl = (pos.entry_price - result['price']) * pos.qty
            
            self.trade_history.append({
                'symbol': symbol, 'side': pos.side, 'qty': pos.qty,
                'entry': pos.entry_price, 'exit': result['price'],
                'pnl': pnl, 'opened': pos.opened_at,
                'closed': datetime.utcnow().isoformat()
            })
            del self.positions[symbol]
            return {'status': 'closed', 'pnl': pnl}
        
        return result
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for symbol."""
        if symbol in self.positions:
            return asdict(self.positions[symbol])
        return None
    
    def get_all_positions(self) -> List[dict]:
        """Get all open positions."""
        return [asdict(p) for p in self.positions.values()]
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                if pos.side == 'buy':
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.qty
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.qty
    
    def _get_sim_price(self, symbol: str) -> float:
        """Get simulated price for paper trading."""
        try:
            ex = ccxt.okx({'enableRateLimit': True})
            ticker = ex.fetch_ticker(symbol)
            return ticker['last']
        except:
            return 63000.0 if 'BTC' in symbol else 1800.0


class PaperTrader:
    """Paper trading simulator with full position tracking."""
    
    def __init__(self, initial_capital: float = 100.0, leverage: int = 20):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.positions: Dict[str, dict] = {}
        self.trade_history: List[dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
    
    def open_position(self, symbol: str, side: str, entry: float, qty: float,
                      sl: float, tp1: float, tp2: float, tp3: float) -> dict:
        """Open a new position."""
        margin = qty * entry / self.leverage
        if margin > self.capital * 0.9:
            return {'status': 'insufficient_margin', 'needed': margin, 'available': self.capital}
        
        self.capital -= margin * 0.001  # Trading fee
        
        self.positions[symbol] = {
            'side': side, 'entry': entry, 'qty': qty, 'margin': margin,
            'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'opened_at': datetime.utcnow().isoformat(),
            'tp1_hit': False, 'tp2_hit': False
        }
        
        return {'status': 'opened', 'margin': margin, 'capital_remaining': self.capital}
    
    def check_candle(self, symbol: str, high: float, low: float, close: float) -> Optional[dict]:
        """Check if SL/TP hit on a candle."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pnl = 0.0
        exit_reason = None
        exit_price = 0.0
        
        if pos['side'] == 'buy':
            # Check SL
            if low <= pos['sl']:
                exit_price, exit_reason = pos['sl'], 'SL'
            # Check TP3
            elif high >= pos['tp3']:
                exit_price, exit_reason = pos['tp3'], 'TP3'
            # Check TP2
            elif high >= pos['tp2']:
                if not pos['tp2_hit']:
                    pos['tp2_hit'] = True
                    pos['sl'] = pos['entry']  # Move SL to breakeven
            # Check TP1
            elif high >= pos['tp1']:
                if not pos['tp1_hit']:
                    pos['tp1_hit'] = True
                    pos['sl'] = pos['entry']  # Move SL to breakeven
        else:  # sell
            if high >= pos['sl']:
                exit_price, exit_reason = pos['sl'], 'SL'
            elif low <= pos['tp3']:
                exit_price, exit_reason = pos['tp3'], 'TP3'
            elif low <= pos['tp2']:
                if not pos['tp2_hit']:
                    pos['tp2_hit'] = True
                    pos['sl'] = pos['entry']
            elif low <= pos['tp1']:
                if not pos['tp1_hit']:
                    pos['tp1_hit'] = True
                    pos['sl'] = pos['entry']
        
        if exit_reason:
            if pos['side'] == 'buy':
                pnl = (exit_price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - exit_price) * pos['qty']
            
            pnl -= abs(pnl) * 0.001  # Fee
            
            self.capital += pos['margin'] + pnl
            
            trade = {
                'symbol': symbol, 'side': pos['side'], 'entry': pos['entry'],
                'exit': exit_price, 'pnl': pnl, 'reason': exit_reason,
                'opened': pos['opened_at'], 'closed': datetime.utcnow().isoformat()
            }
            self.trade_history.append(trade)
            del self.positions[symbol]
            
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            dd = (self.peak_capital - self.capital) / self.peak_capital
            if dd > self.max_drawdown:
                self.max_drawdown = dd
            
            self.equity_curve.append(self.capital)
            return trade
        
        return None
    
    def get_summary(self) -> dict:
        """Get trading summary."""
        if not self.trade_history:
            return {'total_trades': 0, 'capital': self.capital}
        
        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'total_trades': len(self.trade_history),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'max_drawdown': self.max_drawdown * 100,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        }
