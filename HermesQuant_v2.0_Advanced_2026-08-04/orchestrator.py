1|"""
2|HermesQuant Orchestrator — Vertical Pipeline
3|Data → Independent → Meta → Structure → Decision → Risk → Signal
4|"""
5|import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
6|import pandas as pd
7|from datetime import datetime, timezone, timedelta
8|from core.config import *
9|from core.data_fetcher import fetch_candles, fetch_ticker
10|from core import indicators as ind
11|from core.data_types import AgentOutput, SignalOutput
12|
13|# Import all agents
14|from agents.stage0_independent.trend_agent import TrendAgent
15|from agents.stage0_independent.momentum_agent import MomentumAgent
16|from agents.stage0_independent.volume_agent import VolumeAgent
17|from agents.stage0_independent.volatility_agent import VolatilityAgent
18|from agents.stage0_independent.pattern_agent import PatternAgent
19|from agents.stage3_decision.ml_real_agent import MLRealAgent
20|from agents.stage1_meta.regime_agent import RegimeAgent
21|from agents.stage1_meta.structure_agent import StructureAgent
22|from agents.stage1_meta.whale_agent import WhaleAgent
23|from agents.stage1_meta.mtf_agent import MTFConfirmAgent
24|from agents.stage1_meta.funding_rate_agent import FundingRateAgent
25|from agents.stage1_meta.open_interest_agent import OpenInterestAgent
26|from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
27|from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
28|from agents.stage2_structure.liquidity_agent import LiquidityAgent
29|from agents.stage2_structure.wyckoff_agent import WyckoffAgent
30|from agents.stage2_structure.math_brain_agent import MathBrainAgent
31|from agents.stage3_decision.game_theory_agent import GameTheoryAgent
32|from agents.stage3_decision.smart_action_agent import SmartActionAgent
33|from agents.stage3_decision.ml_agent import MLAgent
34|from agents.stage4_risk.risk_agent import RiskAgent
35|from agents.stage4_risk.signal_builder import build_signal
36|
37|tz = timezone(timedelta(hours=3, minutes=30))
38|
39|class MasterOrchestrator:
40|    def __init__(self, capital=10, leverage=20, max_daily_loss=3.0):
41|        self.capital = capital
42|        self.leverage = leverage
43|        self.max_daily_loss = max_daily_loss
44|        
45|        # Stage 0
46|        self.trend = TrendAgent()
47|        self.momentum = MomentumAgent()
48|        self.volume = VolumeAgent()
49|        self.volatility = VolatilityAgent()
50|        self.pattern = PatternAgent()
51|        self.ml_real = MLRealAgent()
52|        
53|        # Stage 1
54|        self.regime = RegimeAgent()
55|        self.structure = StructureAgent()
56|        self.whale = WhaleAgent()
57|        self.mtf = MTFConfirmAgent()
58|        self.funding = FundingRateAgent()
59|        self.open_interest = OpenInterestAgent()
60|        
61|        # Stage 2
62|        self.rsi_div = RSIDivergenceAgent()
63|        self.bb_squeeze = BBSqueezeAgent()
64|        self.liquidity = LiquidityAgent()
65|        self.wyckoff = WyckoffAgent()
66|        self.math_brain = MathBrainAgent()
67|        
68|        # Stage 3
69|        self.game_theory = GameTheoryAgent()
70|        self.smart_action = SmartActionAgent()
71|        self.ml = MLAgent()
72|        
73|        # Stage 4
74|        self.risk = RiskAgent()
75|    
76|    def run(self, symbol="ETH-USDT-SWAP", timeframe="15m"):
77|        now = datetime.now(tz)
78|        print("\n" + "="*60)
79|        print("  HermesQuant v2.0 — Vertical Pipeline")
80|        print("  %s | %s | %s" % (symbol, timeframe, now.strftime('%H:%M:%S Tehran')))
81|        print("="*60)
82|        
83|        # Fetch data
84|        print("\n  [DATA] Fetching %s %s..." % (symbol, timeframe))
85|        df = fetch_candles(symbol, timeframe, 200)
86|        price = df.iloc[-1]["close"]
87|        print("  Price: $%.2f" % price)
88|        
89|        # Stage 0: Independent agents
90|        print("\n  [STAGE 0] Independent Agents:")
91|        s0 = []
92|        for agent in [self.trend, self.momentum, self.volume, 
93|                      self.volatility, self.pattern]:
94|            r = agent.analyze(df, symbol, timeframe)
95|            s0.append(r)
96|            icon = {"BUY":"BUY","SELL":"SELL","NEUTRAL":"---","CALM":"CALM","CHAOTIC":"CHAOT","CRISIS":"CRISIS"}.get(r.direction, r.direction[:4])
97|            print("    %12s: %5s (conf=%3.0f%% score=%+.3f)" % (r.name, icon, r.confidence, r.score))
98|        
99|        # Stage 1: Meta agents
100|        print("\n  [STAGE 1] Meta Agents:")
101|        s1 = []
102|        for agent in [self.regime, self.structure, self.whale, self.mtf, self.funding, self.open_interest]:
103|            r = agent.analyze(df, symbol, timeframe)
104|            s1.append(r)
105|            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
106|        
107|        # Stage 2: Structure agents
108|        print("\n  [STAGE 2] Structure Agents:")
109|        s2 = []
110|        for agent in [self.rsi_div, self.bb_squeeze, self.liquidity, self.wyckoff, self.math_brain]:
111|            r = agent.analyze(df, symbol, timeframe)
112|            s2.append(r)
113|            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
114|        
115|        # Stage 3: Decision agents
116|        print("\n  [STAGE 3] Decision Agents:")
117|        all_prev = s0 + s1 + s2
118|        s3 = []
119|        for agent in [self.game_theory, self.smart_action]:
120|            r = agent.analyze(df, symbol, timeframe)
121|            s3.append(r)
122|            print("    %15s: %5s (conf=%3.0f%%)" % (r.name, r.direction[:5], r.confidence))
123|        ml_r = self.ml.analyze(df, symbol, timeframe, agent_results=all_prev)
124|        s3.append(ml_r)
125|        
126|        # Real ML
127|        ml_real_r = self.ml_real.analyze(df, symbol, timeframe)
128|        s3.append(ml_real_r)
130|        print("    %15s: %5s (conf=%3.0f%%)" % (ml_r.name, ml_r.direction[:5], ml_r.confidence))
131|        
132|        # Stage 4: Risk
133|        self.risk.analyze(df, symbol, timeframe)
134|        
135|        # Build signal
136|        all_agents = s0 + s1 + s2 + s3
137|        signal = build_signal(all_agents, df, self.capital, self.leverage, self.max_daily_loss)
138|        
139|        # 7-Gate Filter
140|        print("\n  [FILTERS] 7-Gate Check:")
141|        buy_w = sum(r.weight for r in all_agents if r.direction == "BUY")
142|        sell_w = sum(r.weight for r in all_agents if r.direction == "SELL")
143|        total = buy_w + sell_w if buy_w + sell_w > 0 else 1
144|        # More agents = higher threshold needed
145|        is_btc = "BTC" in symbol.upper()
146|        vote_thresh = 0.78 if is_btc else 0.75
147|        vote_ok = buy_w/total >= vote_thresh or sell_w/total >= vote_thresh
148|        
149|        h4_df = fetch_candles(symbol, "4H", 100)
150|        _, h4_st = ind.supertrend(h4_df)
151|        h4_price = h4_df.iloc[-1]["close"]
152|        h4_st_val = h4_st.iloc[-1]
153|        h4_ratio = abs(h4_price - h4_st_val) / h4_price
154|        if h4_ratio < 0.002:
155|            h4_agrees = True  # 4H neutral — don't block
156|        else:
157|            h4_dir = 1 if h4_price > h4_st_val else -1
158|            h4_agrees = (h4_dir == 1 and signal.direction == "BUY") or (h4_dir == -1 and signal.direction == "SELL")
159|        
160|        adx_v, _, _ = ind.adx(df)
161|        adx_ok = adx_v.iloc[-1] > 28
162|        
163|        vr = ind.volume_ratio(df).iloc[-1]
164|        vol_ok = vr > 0.8
165|        
166|        hour = now.hour
167|        session_ok = (8 <= hour <= 14) or (14 <= hour <= 22)
168|        
169|        all_same = buy_w > 0 and sell_w == 0 or sell_w > 0 and buy_w == 0
170|        trap_ok = not all_same
171|        
172|        ev_ok = signal.ev > 0
173|        
174|        checks = [
175|            ("Vote (4/5)", vote_ok),
176|            ("4H agrees", h4_agrees),
177|            ("ADX>25", adx_ok),
178|            ("Volume>1x", vol_ok),
179|            ("Session", session_ok),
180|            ("EV>0", ev_ok),
181|            ("Not trap", trap_ok),
182|        ]
183|        
184|        passed = 0
185|        for name, ok in checks:
186|            icon = "PASS" if ok else "FAIL"
187|            print("    %s: %s" % (name, icon))
188|            if ok: passed += 1
189|        
190|        signal.filters_passed = passed
191|        signal.filters_total = len(checks)
192|        
193|        # Final — ONLY signal if ALL 7 filters pass
194|        print("\\n" + "="*60)
195|        if passed < 7:
196|            signal.direction = "WAIT"
197|            signal.confidence = 0
198|            signal.kelly = 0
199|            signal.ev = 0
200|            print("  SIGNAL: WAIT (only %d/7 filters passed)" % passed)
201|        else:
202|            print("  SIGNAL: %s (ALL 7 FILTERS PASS)" % signal.direction)
203|        print("  Filters: %d/%d" % (passed, len(checks)))
204|        print("  Entry: $%.2f" % signal.entry)
205|        if signal.direction != "WAIT":
206|            print("  SL:    $%.2f" % signal.sl)
207|            print("  TP1:   $%.2f  TP2:   $%.2f  TP3:   $%.2f" % (signal.tp1, signal.tp2, signal.tp3))
208|            print("  Kelly: %.1f%% | EV: $%.3f" % (signal.kelly, signal.ev))
209|        print("  P(UP): %.1f%% | P(DOWN): %.1f%%" % (signal.p_up*100, signal.p_down*100))
210|        print("  Convergence: %.1f%%" % signal.convergence)
211|        print("="*60)
212|        
213|        return signal
214|