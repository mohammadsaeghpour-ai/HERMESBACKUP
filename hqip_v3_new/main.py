#!/usr/bin/env python3
"""
HQIP v3 — Professional Trading Intelligence Platform
=====================================================
Main entry point. Run: python -m hqip.main
"""
import sys
sys.path.insert(0, "/data/workspace")

from hqip.data.exchange import ExchangeConnector
from hqip.data.sessions import SessionManager
from hqip.indicators import trend as trend_ind
from hqip.indicators import momentum as mom_ind
from hqip.indicators import volatility as vol_ind
from hqip.indicators import volume as volm_ind
from hqip.indicators import fibonacci as fib_ind
from hqip.ict.structure import MarketStructure
from hqip.ict.order_blocks import OrderBlockDetector
from hqip.ict.fvg import FVGDetector
from hqip.ict.liquidity import LiquidityAnalyzer
from hqip.ict.killzones import KillZoneManager
from hqip.ict.pd_arrays import PremiumDiscount
from hqip.consensus.engine import ConsensusEngine
from hqip.risk.position_sizer import calculate_sltp
from hqip.execution.orders import PaperTrader
from hqip.ml_engine import MLEnsemble, DLForecastEnsemble, FeatureEngine
from hqip.dashboard.html_dashboard import generate_dashboard
import numpy as np
import pandas as pd
import time
import os

SIGNALS_FILE = "/data/workspace/hqip/data/signals.json"
TRADES_FILE = "/data/workspace/hqip/data/trades.json"
DASHBOARD_FILE = "/data/workspace/hqip/dashboard.html"


def analyze_symbol(exchange, symbol: str, capital: float, risk: float, leverage: int) -> list:
    """Full multi-timeframe analysis for a symbol."""
    signals = []
    sm = SessionManager()
    session = sm.get_current_session()
    
    # Fetch data for each timeframe
    for tf in ['4h', '1h', '30m', '15m', '5m']:
        try:
            df = exchange.fetch_ohlcv(symbol, tf, 200)
            if df is None or len(df) < 60:
                continue
            
            price = float(df['close'].iloc[-1])
            
            # Indicators
            trend_score = trend_ind.trend_strength(df)
            mom_score = mom_ind.momentum_score(df)
            vol_score = volm_ind.volume_score(df)
            
            # ICT
            ms = MarketStructure(df)
            structure = ms.analyze()
            ob = OrderBlockDetector(df)
            bull_obs = ob.detect_bull_ob()
            bear_obs = ob.detect_bear_ob()
            fvg = FVGDetector(df)
            bull_fvgs = fvg.detect_bull_fvg()
            bear_fvgs = fvg.detect_bear_fvg()
            
            # ATR
            atr_series = vol_ind.atr(df)
            atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else price * 0.01
            
            # RSI
            rsi_series = mom_ind.rsi(df)
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            
            # Consensus
            agents = [
                {'direction': 'BUY' if trend_score > 10 else 'SELL' if trend_score < -10 else 'NEUTRAL', 'score': abs(trend_score), 'weight': 1.0},
                {'direction': 'BUY' if mom_score > 10 else 'SELL' if mom_score < -10 else 'NEUTRAL', 'score': abs(mom_score), 'weight': 0.8},
                {'direction': 'BUY' if vol_score > 10 else 'SELL' if vol_score < -10 else 'NEUTRAL', 'score': abs(vol_score), 'weight': 0.7},
            ]
            
            # Structure
            if structure.get('trend') == 'bullish':
                agents.append({'direction': 'BUY', 'score': 60, 'weight': 1.2})
            elif structure.get('trend') == 'bearish':
                agents.append({'direction': 'SELL', 'score': 60, 'weight': 1.2})
            
            # OB/FVG
            if bull_obs:
                agents.append({'direction': 'BUY', 'score': 50, 'weight': 0.9})
            if bear_obs:
                agents.append({'direction': 'SELL', 'score': 50, 'weight': 0.9})
            if bull_fvgs:
                agents.append({'direction': 'BUY', 'score': 40, 'weight': 0.6})
            if bear_fvgs:
                agents.append({'direction': 'SELL', 'score': 40, 'weight': 0.6})
            
            # Session filter
            if session == 'asia':
                for a in agents: a['weight'] *= 0.7
            
            # Consensus
            consensus = ConsensusEngine()
            result = consensus.decide(agents)
            
            if result['direction'] in ('BUY', 'SELL'):
                sltp = calculate_sltp(price, atr, result['direction'], 1.5, 2.0, 3.0, 4.0)
                
                signals.append({
                    'symbol': symbol, 'tf': tf, 'direction': result['direction'],
                    'confidence': result['confidence'], 'entry': price,
                    'sl': sltp['sl'], 'tp1': sltp['tp1'], 'tp2': sltp['tp2'], 'tp3': sltp['tp3'],
                    'atr': atr, 'rsi': rsi, 'trend_score': trend_score,
                    'session': session, 'n_agents': len([a for a in agents if a['direction'] == result['direction']]),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M'),
                })
            
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️ Error {symbol}/{tf}: {e}")
    
    return signals


def main():
    print("=" * 60)
    print("  🎯 HQIP v3 — Professional Trading Intelligence")
    print("=" * 60)
    
    # Config
    symbols = ['BTC/USDT', 'ETH/USDT']
    capital = float(input("💰 سرمایه ($): ") or "20")
    risk = float(input("🛡 حداکثر ضرر ($): ") or "4")
    leverage = int(input("⚡ اهرم: ") or "20")
    
    # Connect
    exchange = ExchangeConnector()
    
    all_signals = []
    all_trades = []
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        signals = analyze_symbol(exchange, symbol, capital, risk, leverage)
        all_signals.extend(signals)
        
        for s in signals:
            d_fa = '🟢 خرید' if s['direction'] == 'BUY' else '🔴 فروش'
            print(f"  {s['tf']:4} {d_fa} | {s['confidence']}% | Entry: ${s['entry']:,.2f}")
            print(f"        SL: ${s['sl']:,.2f} | TP1: ${s['tp1']:,.2f} | TP2: ${s['tp2']:,.2f} | TP3: ${s['tp3']:,.2f}")
    
    # Generate dashboard
    metrics = {
        'final_capital': capital, 'total_return_pct': 0,
        'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0,
        'max_drawdown_pct': 0
    }
    
    html = generate_dashboard(all_signals, all_trades, metrics, [capital])
    with open(DASHBOARD_FILE, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Dashboard: {DASHBOARD_FILE}")
    print(f"📊 Total signals: {len(all_signals)}")
    print(f"⚠️ این سیستم مشاوره مالی نیست")


if __name__ == '__main__':
    main()
