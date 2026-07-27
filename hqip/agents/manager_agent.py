"""
Manager Agent — Master Orchestrator
====================================
Coordinates ALL 22 agents, resolves conflicts, produces final recommendation.
"""
from hqip.agents.base import BaseAgent, AgentOutput
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators
from hqip.config import TIMEFRAMES

class ManagerAgent:
    def __init__(self):
        self.dp = DataPlatform()
        self.agents = {}
        self._load_agents()

    def _load_agents(self):
        agent_map = {
            "data_quality": ("hqip.agents.data_quality_agent", "DataQualityAgent"),
            "feature_engineering": ("hqip.agents.feature_engineering_agent", "FeatureEngineeringAgent"),
            "market_structure": ("hqip.agents.market_structure_agent", "MarketStructureAgent"),
            "smc": ("hqip.agents.smc_agent", "SMCAgent"),
            "liquidity": ("hqip.agents.liquidity_agent", "LiquidityAgent"),
            "wyckoff": ("hqip.agents.wyckoff_agent", "WyckoffAgent"),
            "supply_demand": ("hqip.agents.supply_demand_agent", "SupplyDemandAgent"),
            "smart_action": ("hqip.agents.smart_action_agent", "SmartActionAgent"),
            "trend": ("hqip.agents.trend_agent", "TrendAgent"),
            "momentum": ("hqip.agents.momentum_agent", "MomentumAgent"),
            "volume": ("hqip.agents.volume_agent", "VolumeAgent"),
            "volatility": ("hqip.agents.volatility_agent", "VolatilityAgent"),
            "pattern": ("hqip.agents.pattern_agent", "PatternAgent"),
            "regime": ("hqip.agents.regime_agent", "RegimeAgent"),
            "ml": ("hqip.agents.ml_agent", "MLAgent"),
            "dl": ("hqip.agents.dl_forecast_agent", "DLForecastAgent"),
            "whale": ("hqip.agents.whale_agent", "WhaleAgent"),
            "news": ("hqip.agents.news_agent", "NewsAgent"),
            "risk": ("hqip.agents.risk_agent", "RiskAgent"),
            "consensus": ("hqip.agents.consensus_agent", "ConsensusAgent"),
            "explainability": ("hqip.agents.explainability_agent", "ExplainabilityAgent"),
        }

        for key, (module_path, class_name) in agent_map.items():
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.agents[key] = cls()
            except Exception as e:
                print(f"  ⚠️ {key}: {str(e)[:40]}")

        print(f"[Manager] Loaded {len(self.agents)}/21 agents")

    def _run_agent(self, agent, **kwargs):
        try:
            return agent.analyze(**kwargs)
        except Exception as e:
            return AgentOutput(agent_name=agent.name, confidence=0, error=str(e)[:100],
                             evidence=[f"Error: {str(e)[:50]}"])

    def scan(self, symbol, capital=10000, max_loss=100, leverage=5):
        print(f"\n{'='*60}")
        print(f"  🔍 HQIP HUNTER: {symbol}")
        print(f"  {'='*58}")

        # 1. Fetch data
        print(f"[1/5] 📡 Fetching market data...")
        data = self.dp.fetch_all_timeframes(symbol, TIMEFRAMES)
        for tf, df in data.items():
            if not df.empty:
                data[tf] = calculate_all_indicators(df)
                print(f"  ✅ {tf}: {len(df)} candles")

        # 2. Run SMC/Smart Money agents (THE HUNTERS)
        print(f"[2/5] 🎯 Running Smart Money hunters...")
        tf_results = {}
        all_results = {}

        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty: continue

            results = []
            # ── Smart Money Agents (highest priority) ──
            for name in ["smc", "liquidity", "wyckoff", "supply_demand", "smart_action", "market_structure"]:
                if name in self.agents:
                    out = self._run_agent(self.agents[name], df=df, symbol=symbol, timeframe=tf)
                    results.append(out)

            # ── Traditional Technical Agents ──
            for name in ["trend", "momentum", "volume", "volatility", "pattern", "regime"]:
                if name in self.agents:
                    out = self._run_agent(self.agents[name], df=df, symbol=symbol, timeframe=tf)
                    results.append(out)

            # ── ML/DL ──
            for name in ["ml", "dl"]:
                if name in self.agents:
                    out = self._run_agent(self.agents[name], df=df, symbol=symbol, timeframe=tf)
                    results.append(out)

            tf_results[tf] = results
            all_results[tf] = results

            buy_count = sum(1 for r in results if r.direction == "BUY")
            sell_count = sum(1 for r in results if r.direction == "SELL")
            print(f"  [{tf}] {len(results)} agents: {buy_count} 🟢 | {sell_count} 🔴")

        # 3. Run Whale & News
        print(f"[3/5] 🐋 Running whale & news analysis...")
        for name in ["whale", "news"]:
            if name in self.agents:
                out = self._run_agent(self.agents[name], symbol=symbol, timeframe="1h", df=data.get("1h"))
                for tf in tf_results:
                    tf_results[tf].append(out)
                print(f"  ✅ {name}: {out.direction} ({out.confidence:.0f}%)")

        # 4. Consensus
        print(f"[4/5] 🧠 Running consensus engine...")
        consensus = self._run_agent(self.agents["consensus"], symbol=symbol, tf_results=tf_results)
        grade = consensus.data.get("grade", "?")
        print(f"  📊 Result: {consensus.direction} (Grade: {grade}, Confidence: {consensus.confidence:.0f}%)")

        # 5. Risk
        risk_data = {}
        if consensus.direction in ("BUY", "SELL"):
            risk_out = self._run_agent(self.agents["risk"], df=data.get("1h", data.get("4h")),
                symbol=symbol, direction=consensus.direction, capital=capital, max_loss=max_loss, leverage=leverage)
            risk_data = risk_out.data

        # 6. Explainability
        explain = self._run_agent(self.agents["explainability"], symbol=symbol,
            all_results=all_results, consensus_result=consensus)

        # 7. Build result
        result = {
            "symbol": symbol, "direction": consensus.direction,
            "grade": grade, "confidence": consensus.confidence,
            "score": consensus.score,
            "entry": risk_data.get("entry"), "sl": risk_data.get("sl"),
            "tp1": risk_data.get("tp1"), "tp2": risk_data.get("tp2"),
            "tp3": risk_data.get("tp3"),
            "position_size": risk_data.get("position_size"),
            "position_value": risk_data.get("position_value"),
            "risk_reward": risk_data.get("rr1"),
            "leverage": leverage, "capital": capital,
            "agent_results": {tf: [r.to_dict() for r in results] for tf, results in all_results.items()},
            "consensus_reasoning": consensus.reasoning,
            "explanation": explain.evidence,
        }

        print(f"[5/5] ✅ Signal ready!")
        return result
