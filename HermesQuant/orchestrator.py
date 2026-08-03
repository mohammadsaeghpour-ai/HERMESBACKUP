"""
HermesQuant Orchestrator — Vertical Pipeline
Data → Independent → Meta → Structure → Decision → Risk → Signal
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.config import *
from core.data_fetcher import fetch_candles, fetch_ticker
from core import indicators as ind
from core.data_types import AgentOutput, SignalOutput

# Import all agents
from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage0_independent.dl_forecast_agent import DLForecastAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent
from agents.stage3_decision.ml_agent import MLAgent
from agents.stage4_risk.risk_agent import RiskAgent
from agents.stage4_risk.signal_builder import build_signal

tz = timezone(timedelta(hours=3, minutes=30))

class MasterOrchestrator:
    def __init__(self, capital=10, leverage=20, max_daily_loss=3.0):
        self.capital = capital
        self.leverage = leverage
        self.max_daily_loss = max_daily_loss
        
        # Stage 0
        self.trend = TrendAgent()
        self.momentum = MomentumAgent()
        self.volume = VolumeAgent()
        self.volatility = VolatilityAgent()
        self.pattern = PatternAgent()
        self.dl_forecast = DLForecastAgent()
        
        # Stage 1
        self.regime = RegimeAgent()
        self.structure = StructureAgent()
        self.whale = WhaleAgent()
        
        # Stage 2
        self.rsi_div = RSIDivergenceAgent()
        self.bb_squeeze = BBSqueezeAgent()
        self.liquidity = LiquidityAgent()
        self.wyckoff = WyckoffAgent()
        self.math_brain = MathBrainAgent()
        
        # Stage 3
        self.game_theory = GameTheoryAgent()
        self.smart_action = SmartActionAgent()
        self.ml = MLAgent()
        
        # Stage 4
        self.risk = RiskAgent()
    
    def run(self, symbol="ETH-USDT-SWAP", timeframe="15m"):
        now = datetime.now(tz)
        print("\n" + "="*60)
        print("  HermesQuant v2.0 — Vertical Pipeline")
        print("  %s | %s | %s" % (symbol, timeframe, now.strftime('%H:%M:%S Tehran')))
        print("="*60)
        
        # Fetch data
        print("\n  [DATA] Fetching %s %s..." % (symbol, timeframe))
        df = fetch_candles(symbol, timeframe, 200)
        price = df.iloc[-1]["close"]
        print("  Price: $%.2f" % price)
        
        # Stage 0: Independent agents
        print("\n  [STAGE 0] Independent Agents:")
        s0 = []
        for agent in [self.trend, self.momentum, self.volume, 
                      self.volatility, self.pattern, self.dl_forecast]:
            r = agent.analyze(df, symbol, timeframe)
            s0.append(r)
            icon = {"BUY":"BUY","SELL":"SELL","NEUTRAL":"---","CALM":"CALM","CHAOTIC":"CHAOT","CRISIS":"CRISIS"}.get(r.direction, r.direction[:4])
            print("    %12s: %5s (conf=%3.0f%% score=%+.3f)" % (r.name, icon, r.confidence, r.score))
        
        # Stage 1: Meta agents
        print("\n  [STAGE 1] Meta Agents:")
        s1 = []
        for agent in [self.regime, self.structure, self.whale]:
            r = agent.analyze(df, symbol, timeframe)
            s1.append(r)
            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
        
        # Stage 2: Structure agents
        print("\n  [STAGE 2] Structure Agents:")
        s2 = []
        for agent in [self.rsi_div, self.bb_squeeze, self.liquidity, self.wyckoff, self.math_brain]:
            r = agent.analyze(df, symbol, timeframe)
            s2.append(r)
            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
        
        # Stage 3: Decision agents
        print("\n  [STAGE 3] Decision Agents:")
        all_prev = s0 + s1 + s2
        s3 = []
        for agent in [self.game_theory, self.smart_action]:
            r = agent.analyze(df, symbol, timeframe)
            s3.append(r)
            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
        ml_r = self.ml.analyze(df, symbol, timeframe, agent_results=all_prev)
        s3.append(ml_r)
        print("    %15s: %5s (conf=%3.0f%%)" % (ml_r.name, ml_r.direction[:5], ml_r.confidence))
        
        # Stage 4: Risk
        self.risk.analyze(df, symbol, timeframe)
        
        # Build signal
        all_agents = s0 + s1 + s2 + s3
        signal = build_signal(all_agents, df, self.capital, self.leverage, self.max_daily_loss)
        
        # 7-Gate Filter
        print("\n  [FILTERS] 7-Gate Check:")
        buy_w = sum(r.weight for r in all_agents if r.direction == "BUY")
        sell_w = sum(r.weight for r in all_agents if r.direction == "SELL")
        total = buy_w + sell_w if buy_w + sell_w > 0 else 1
        vote_ok = buy_w/total >= 0.75 or sell_w/total >= 0.75
        
        h4_df = fetch_candles(symbol, "4H", 100)
        _, h4_st = ind.supertrend(h4_df)
        h4_price = h4_df.iloc[-1]["close"]
        h4_st_val = h4_st.iloc[-1]
        h4_ratio = abs(h4_price - h4_st_val) / h4_price
        if h4_ratio < 0.002:
            h4_agrees = True  # 4H neutral — don't block
        else:
            h4_dir = 1 if h4_price > h4_st_val else -1
            h4_agrees = (h4_dir == 1 and signal.direction == "BUY") or (h4_dir == -1 and signal.direction == "SELL")
        
        adx_v, _, _ = ind.adx(df)
        adx_ok = adx_v.iloc[-1] > 28
        
        vr = ind.volume_ratio(df).iloc[-1]
        vol_ok = vr > 0.8
        
        hour = now.hour
        session_ok = (8 <= hour <= 14) or (14 <= hour <= 22)
        
        all_same = buy_w > 0 and sell_w == 0 or sell_w > 0 and buy_w == 0
        trap_ok = not all_same
        
        ev_ok = signal.ev > 0
        
        checks = [
            ("Vote (4/5)", vote_ok),
            ("4H agrees", h4_agrees),
            ("ADX>25", adx_ok),
            ("Volume>1x", vol_ok),
            ("Session", session_ok),
            ("EV>0", ev_ok),
            ("Not trap", trap_ok),
        ]
        
        passed = 0
        for name, ok in checks:
            icon = "PASS" if ok else "FAIL"
            print("    %s: %s" % (name, icon))
            if ok: passed += 1
        
        signal.filters_passed = passed
        signal.filters_total = len(checks)
        
        # Final — ONLY signal if ALL 7 filters pass
        print("\\n" + "="*60)
        if passed < 7:
            signal.direction = "WAIT"
            signal.confidence = 0
            signal.kelly = 0
            signal.ev = 0
            print("  SIGNAL: WAIT (only %d/7 filters passed)" % passed)
        else:
            print("  SIGNAL: %s (ALL 7 FILTERS PASS)" % signal.direction)
        print("  Filters: %d/%d" % (passed, len(checks)))
        print("  Entry: $%.2f" % signal.entry)
        if signal.direction != "WAIT":
            print("  SL:    $%.2f" % signal.sl)
            print("  TP1:   $%.2f  TP2:   $%.2f  TP3:   $%.2f" % (signal.tp1, signal.tp2, signal.tp3))
            print("  Kelly: %.1f%% | EV: $%.3f" % (signal.kelly, signal.ev))
        print("  P(UP): %.1f%% | P(DOWN): %.1f%%" % (signal.p_up*100, signal.p_down*100))
        print("  Convergence: %.1f%%" % signal.convergence)
        print("="*60)
        
        return signal
