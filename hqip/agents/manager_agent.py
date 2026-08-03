"""
Manager Agent — Enhanced Master Orchestrator
=============================================
Coordinates ALL agents, resolves conflicts, produces final recommendation.

Key features:
- Run all agents per timeframe independently
- Whale/news added ONCE (not duplicated across TFs)
- Graceful error recovery (if one agent fails, continue)
- Progress reporting
- Return structured result with per-TF signals

Weight: 0
"""
from hqip.agents.base import BaseAgent, AgentOutput
from hqip.config import TIMEFRAMES


class ManagerAgent:
    def __init__(self):
        self.dp = None
        self.agents = {}
        self._load_agents()

    def _load_agents(self):
        """Dynamically load all agents with graceful error handling."""
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
            "math_brain": ("hqip.agents.math_brain_agent", "MathBrainAgent"),
            "game_theory": ("hqip.agents.game_theory_agent", "GameTheoryAgent"),
            "absolute_zero": ("hqip.agents.absolute_zero_agent", "AbsoluteZeroAgent"),
        }

        loaded = 0
        for key, (module_path, class_name) in agent_map.items():
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.agents[key] = cls()
                loaded += 1
            except Exception as e:
                print(f"  ⚠️ {key}: failed to load — {str(e)[:60]}")

        print(f"[Manager] Loaded {loaded}/{len(agent_map)} agents")
        self._loaded_count = loaded
        self._total_count = len(agent_map)

    def _run_agent(self, agent, **kwargs):
        """Run a single agent with graceful error recovery."""
        try:
            return agent.analyze(**kwargs)
        except Exception as e:
            return AgentOutput(
                agent_name=getattr(agent, "name", "Unknown"),
                confidence=0,
                error=str(e)[:200],
                evidence=[f"Error: {str(e)[:80]}"],
            )

    def _get_smart_money_agents(self):
        """Return the list of smart money agent keys."""
        return ["smc", "liquidity", "wyckoff", "supply_demand", "smart_action", "market_structure"]

    def _get_technical_agents(self):
        """Return the list of technical analysis agent keys."""
        return ["trend", "momentum", "volume", "volatility", "pattern", "regime"]

    def _get_ml_agents(self):
        """Return the list of ML/DL agent keys."""
        return ["ml", "dl"]

    def scan(self, symbol, capital=10000, max_loss=100, leverage=5):
        """
        Full market scan for a symbol.

        Steps:
        1. Fetch market data for all timeframes
        2. Run Data Quality + Feature Engineering (once per TF)
        3. Run smart money agents per TF
        4. Run technical agents per TF
        5. Run ML/DL agents per TF
        6. Run whale + news ONCE (not per TF)
        7. Run consensus (per-TF independent aggregation)
        8. Run risk management
        9. Run explainability
        10. Return structured result
        """
        print(f"\n{'=' * 60}")
        print(f"  🔍 HQIP HUNTER: {symbol}")
        print(f"  {'=' * 58}")

        # ═══════════════════════════════════════════════════════
        # STEP 1: Fetch market data
        # ═══════════════════════════════════════════════════════
        print(f"[1/8] 📡 Fetching market data...")
        try:
            from hqip.data_platform import DataPlatform
            from hqip.indicators import calculate_all_indicators
            self.dp = DataPlatform()
            data = self.dp.fetch_all_timeframes(symbol, TIMEFRAMES)
        except Exception as e:
            print(f"  ❌ Data fetch failed: {e}")
            return self._build_error_result(symbol, f"Data fetch failed: {e}")

        for tf, df in data.items():
            if not df.empty:
                try:
                    data[tf] = calculate_all_indicators(df)
                    print(f"  ✅ {tf}: {len(df)} candles")
                except Exception as e:
                    print(f"  ⚠️ {tf}: indicators failed — {str(e)[:40]}")

        # ═══════════════════════════════════════════════════════
        # STEP 2: Data Quality check (per TF)
        # ═══════════════════════════════════════════════════════
        print(f"[2/8] 📊 Running data quality checks...")
        quality_results = {}
        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty:
                continue
            if "data_quality" in self.agents:
                out = self._run_agent(
                    self.agents["data_quality"],
                    df=df, symbol=symbol, timeframe=tf,
                )
                quality_results[tf] = out
                qs = out.data.get("quality_score", 0)
                print(f"  [{tf}] Quality: {qs}/100")

        # ═══════════════════════════════════════════════════════
        # STEP 3: Feature Engineering (per TF)
        # ═══════════════════════════════════════════════════════
        print(f"[3/8] 🔧 Running feature engineering...")
        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty:
                continue
            if "feature_engineering" in self.agents:
                out = self._run_agent(
                    self.agents["feature_engineering"],
                    df=df, symbol=symbol, timeframe=tf,
                )
                # Merge features back into the dataframe
                if hasattr(out, "data") and out.data:
                    for col_name in out.data.get("columns", []):
                        if col_name in out.data:
                            df[col_name] = out.data[col_name]
                    data[tf] = df

        # ═══════════════════════════════════════════════════════
        # STEP 4-6: Run all analysis agents PER TIMEFRAME
        # ═══════════════════════════════════════════════════════
        tf_results = {}
        all_results = {}

        sm_agents = self._get_smart_money_agents()
        tech_agents = self._get_technical_agents()
        ml_agents = self._get_ml_agents()
        all_analysis = sm_agents + tech_agents + ml_agents

        print(f"[4/8] 🎯 Running analysis agents per timeframe...")
        for tf in TIMEFRAMES:
            df = data.get(tf)
            if df is None or df.empty:
                print(f"  [{tf}] ⚠️ No data — skipping")
                continue

            results = []
            for agent_key in all_analysis:
                if agent_key in self.agents:
                    out = self._run_agent(
                        self.agents[agent_key],
                        df=df, symbol=symbol, timeframe=tf,
                    )
                    results.append(out)

            tf_results[tf] = results
            all_results[tf] = results

            buy_count = sum(1 for r in results if r.direction == "BUY")
            sell_count = sum(1 for r in results if r.direction == "SELL")
            print(f"  [{tf}] {len(results)} agents: {buy_count} 🟢 | {sell_count} 🔴")

        # ═══════════════════════════════════════════════════════
        # STEP 7: Run Whale & News ONCE (shared across all TFs)
        # ═══════════════════════════════════════════════════════
        print(f"[5/8] 🐋 Running whale & news (shared)...")
        whale_news_results = []

        for agent_key in ["whale", "news"]:
            if agent_key in self.agents:
                out = self._run_agent(
                    self.agents[agent_key],
                    symbol=symbol,
                    timeframe="1h",
                    df=data.get("1h", data.get("4h")),
                )
                whale_news_results.append(out)
                print(f"  ✅ {agent_key}: {out.direction} ({out.confidence:.0f}%)")

        # Store under shared key — added to every TF during consensus
        tf_results["_whale_news"] = whale_news_results

        # ═══════════════════════════════════════════════════════
        # STEP 8: Consensus
        # ═══════════════════════════════════════════════════════
        print(f"[6/8] 🧠 Running consensus engine...")
        consensus = self._run_agent(
            self.agents.get("consensus", self._make_null_agent("Consensus")),
            symbol=symbol,
            tf_results=tf_results,
        )
        grade = consensus.data.get("grade", "?")
        print(
            f"  📊 Result: {consensus.direction} "
            f"(Grade: {grade}, Confidence: {consensus.confidence:.0f}%)"
        )

        # ═══════════════════════════════════════════════════════
        # STEP 9: Risk Management
        # ═══════════════════════════════════════════════════════
        print(f"[7/8] ⚡ Running risk management...")
        risk_data = {}
        if consensus.direction in ("BUY", "SELL") and "risk" in self.agents:
            risk_out = self._run_agent(
                self.agents["risk"],
                df=data.get("1h", data.get("4h")),
                symbol=symbol,
                direction=consensus.direction,
                capital=capital,
                max_loss=max_loss,
                leverage=leverage,
            )
            risk_data = risk_out.data
            print(
                f"  ✅ SL={risk_data.get('sl', '?')} "
                f"TP1={risk_data.get('tp1', '?')} "
                f"TP2={risk_data.get('tp2', '?')} "
                f"TP3={risk_data.get('tp3', '?')}"
            )
        else:
            print("  ℹ️ No position to risk (NO_TRADE)")

        # ═══════════════════════════════════════════════════════
        # STEP 10: Explainability
        # ═══════════════════════════════════════════════════════
        explain = self._run_agent(
            self.agents.get("explainability", self._make_null_agent("Explainability")),
            symbol=symbol,
            all_results=all_results,
            consensus_result=consensus,
        )

        # ═══════════════════════════════════════════════════════
        # BUILD FINAL RESULT
        # ═══════════════════════════════════════════════════════
        result = {
            "symbol": symbol,
            "direction": consensus.direction,
            "grade": grade,
            "confidence": consensus.confidence,
            "score": consensus.score,
            # Risk levels
            "entry": risk_data.get("entry"),
            "sl": risk_data.get("sl"),
            "tp1": risk_data.get("tp1"),
            "tp2": risk_data.get("tp2"),
            "tp3": risk_data.get("tp3"),
            "position_size": risk_data.get("position_size"),
            "position_value": risk_data.get("position_value"),
            "risk_reward": risk_data.get("rr1"),
            "kelly_pct": risk_data.get("kelly_pct"),
            "leverage": leverage,
            "capital": capital,
            # Detailed results
            "per_tf_signals": {
                tf: r["direction"]
                for tf, r in consensus.data.get("per_tf_results", {}).items()
            },
            "agent_results": {
                tf: [r.to_dict() for r in agents]
                for tf, agents in all_results.items()
            },
            "quality_scores": {
                tf: qr.data.get("quality_score", 0)
                for tf, qr in quality_results.items()
            },
            "consensus_reasoning": consensus.reasoning,
            "explanation": explain.evidence,
            "farsi_explanation": explain.data.get("farsi_explanation", ""),
            "confidence_score": explain.data.get("confidence_score", 0),
            # Metadata
            "agents_loaded": self._loaded_count,
            "agents_total": self._total_count,
        }

        print(f"[8/8] ✅ Signal ready!")
        print(f"  Direction: {result['direction']}")
        print(f"  Grade: {result['grade']}")
        print(f"  Confidence: {result['confidence']:.0f}%")
        if result.get("entry"):
            print(f"  Entry: {result['entry']}")
            print(f"  SL: {result['sl']} | TP1: {result['tp1']} | TP2: {result['tp2']} | TP3: {result['tp3']}")

        return result

    def _make_null_agent(self, agent_name):
        """Create a minimal null agent for when consensus/explainability isn't loaded."""

        class NullAgent(BaseAgent):
            weight = 0

            def analyze(self, **kwargs):
                return self._out(
                    direction="NEUTRAL", confidence=0,
                    evidence=[f"Agent not loaded"],
                )

        null = NullAgent()
        null.name = agent_name
        return null

    def _build_error_result(self, symbol, error_msg):
        """Build a minimal error result when the scan fails."""
        return {
            "symbol": symbol,
            "direction": "ERROR",
            "grade": "F",
            "confidence": 0,
            "score": 0,
            "entry": None, "sl": None, "tp1": None, "tp2": None, "tp3": None,
            "position_size": None, "position_value": None, "risk_reward": None,
            "kelly_pct": None, "leverage": None, "capital": None,
            "per_tf_signals": {},
            "agent_results": {},
            "quality_scores": {},
            "consensus_reasoning": error_msg,
            "explanation": [f"❌ {error_msg}"],
            "farsi_explanation": f"خطا در تحلیل {symbol}: {error_msg}",
            "confidence_score": 0,
            "agents_loaded": self._loaded_count,
            "agents_total": self._total_count,
            "error": error_msg,
        }
