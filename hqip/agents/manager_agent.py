"""
Manager Agent
=============
Master orchestrator. Coordinates all agents, resolves conflicts,
validates evidence, and produces final recommendation.
"""
from hqip.agents.base import BaseAgent, AgentOutput
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators
from hqip.config import TIMEFRAMES
import traceback

class ManagerAgent:
    def __init__(self):
        self.dp = DataPlatform()
        self.agents = {}
        self._load_agents()

    def _load_agents(self):
        try:
            from hqip.agents.data_quality_agent import DataQualityAgent
            from hqip.agents.feature_engineering_agent import FeatureEngineeringAgent
            from hqip.agents.market_structure_agent import MarketStructureAgent
            from hqip.agents.trend_agent import TrendAgent
            from hqip.agents.momentum_agent import MomentumAgent
            from hqip.agents.volume_agent import VolumeAgent
            from hqip.agents.volatility_agent import VolatilityAgent
            from hqip.agents.pattern_agent import PatternAgent
            from hqip.agents.regime_agent import RegimeAgent
            from hqip.agents.ml_agent import MLAgent
            from hqip.agents.dl_forecast_agent import DLForecastAgent
            from hqip.agents.whale_agent import WhaleAgent
            from hqip.agents.news_agent import NewsAgent
            from hqip.agents.risk_agent import RiskAgent
            from hqip.agents.consensus_agent import ConsensusAgent
            from hqip.agents.explainability_agent import ExplainabilityAgent

            self.agents = {
                "data_quality": DataQualityAgent(),
                "feature_engineering": FeatureEngineeringAgent(),
                "market_structure": MarketStructureAgent(),
                "trend": TrendAgent(),
                "momentum": MomentumAgent(),
                "volume": VolumeAgent(),
                "volatility": VolatilityAgent(),
                "pattern": PatternAgent(),
                "regime": RegimeAgent(),
                "ml": MLAgent(),
                "dl": DLForecastAgent(),
                "whale": WhaleAgent(),
                "news": NewsAgent(),
                "risk": RiskAgent(),
                "consensus": ConsensusAgent(),
                "explainability": ExplainabilityAgent(),
            }
            print(f"[Manager] Loaded {len(self.agents)} agents")
        except Exception as e:
            print(f"[Manager] Error loading agents: {e}")

    def _run_agent(self, agent, **kwargs):
        try:
            return agent.analyze(**kwargs)
        except Exception as e:
            return AgentOutput(agent_name=agent.name, confidence=0, error=str(e)[:100],
                             evidence=[f"Agent error: {str(e)[:50]}"])

    def scan(self, symbol, capital=10000, max_loss=100, leverage=5):
        print(f"\n{'='*60}")
        print(f"  HQIP SCAN: {symbol}")
        print(f"{'='*60}")

        # 1. Fetch data for all timeframes
        print(f"[1/4] Fetching data...")
        data = self.dp.fetch_all_timeframes(symbol, TIMEFRAMES)
        for tf, df in data.items():
            if df.empty:
                print(f"  ⚠️ {tf}: No data")
                continue
            data[tf] = calculate_all_indicators(df)
            print(f"  ✅ {tf}: {len(df)} candles")

        # 2. Run agents per timeframe
        print(f"[2/4] Running agents per timeframe...")
        tf_results = {}  # {tf: [AgentOutput, ...]}
        all_results = {}  # {tf: [AgentOutput, ...]}

        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty:
                continue

            results = []
            # Technical agents
            for name in ["data_quality", "market_structure", "trend", "momentum",
                        "volume", "volatility", "pattern", "regime"]:
                agent = self.agents.get(name)
                if agent:
                    out = self._run_agent(agent, df=df, symbol=symbol, timeframe=tf)
                    results.append(out)

            # ML & DL
            for name in ["ml", "dl"]:
                agent = self.agents.get(name)
                if agent:
                    out = self._run_agent(agent, df=df, symbol=symbol, timeframe=tf)
                    results.append(out)

            tf_results[tf] = results
            all_results[tf] = results

            buy_count = sum(1 for r in results if r.direction == "BUY")
            sell_count = sum(1 for r in results if r.direction == "SELL")
            print(f"  [{tf}] {len(results)} agents: {buy_count} BUY, {sell_count} SELL")

        # 3. Run whale & news (timeframe-agnostic, use 1h data)
        print(f"[3/4] Running whale & news analysis...")
        for name in ["whale", "news"]:
            agent = self.agents.get(name)
            if agent:
                out = self._run_agent(agent, symbol=symbol, timeframe="1h", df=data.get("1h"))
                # Add to all timeframes
                for tf in tf_results:
                    tf_results[tf].append(out)
                print(f"  ✅ {name}: {out.direction} ({out.confidence:.0f}%)")

        # 4. Consensus
        print(f"[4/4] Running consensus...")
        consensus = self._run_agent(
            self.agents["consensus"],
            symbol=symbol, tf_results=tf_results
        )
        print(f"  📊 Result: {consensus.direction} (Grade: {consensus.data.get('grade', '?')})")

        # 5. Risk calculation
        risk_data = {}
        if consensus.direction in ("BUY", "SELL"):
            risk_out = self._run_agent(
                self.agents["risk"],
                df=data.get("1h", data.get("4h")),
                symbol=symbol,
                direction=consensus.direction,
                capital=capital,
                max_loss=max_loss,
                leverage=leverage
            )
            risk_data = risk_out.data

        # 6. Explainability
        explain = self._run_agent(
            self.agents["explainability"],
            symbol=symbol,
            all_results=all_results,
            consensus_result=consensus
        )

        # 7. Build final result
        result = {
            "symbol": symbol,
            "direction": consensus.direction,
            "grade": consensus.data.get("grade", "C"),
            "confidence": consensus.confidence,
            "score": consensus.score,
            "entry": risk_data.get("entry"),
            "sl": risk_data.get("sl"),
            "tp1": risk_data.get("tp1"),
            "tp2": risk_data.get("tp2"),
            "tp3": risk_data.get("tp3"),
            "position_size": risk_data.get("position_size"),
            "position_value": risk_data.get("position_value"),
            "risk_reward": risk_data.get("rr1"),
            "leverage": leverage,
            "capital": capital,
            "agent_results": {tf: [r.to_dict() for r in results] for tf, results in all_results.items()},
            "consensus_reasoning": consensus.reasoning,
            "explanation": explain.evidence,
        }

        return result
