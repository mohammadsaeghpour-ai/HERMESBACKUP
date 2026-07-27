"""
HQIP Orchestrator
=================
Main orchestration engine that coordinates all agents, feeds results to
consensus, and produces final trading signals.
"""

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hqip.config import GRADE_THRESHOLDS, SYMBOLS, TF_WEIGHTS, TIMEFRAMES

logger = logging.getLogger("hqip.orchestrator")


# ── Data Structures ────────────────────────────────────────

@dataclass
class AgentResult:
    """Result from a single agent analysis."""
    agent_name: str
    direction: str          # "long", "short", "neutral"
    confidence: float       # 0.0 - 1.0
    weight: float = 1.0     # importance weight for consensus
    reasoning: str = ""     # human-readable explanation
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class Signal:
    """Final trading signal produced by the orchestrator."""
    symbol: str
    timeframe: str
    direction: str          # "LONG", "SHORT", "NO_TRADE"
    grade: str              # "A+", "A", "B+", "B", "C", "NO_GRADE"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size_usd: float
    leverage: int
    risk_reward: float
    agent_results: List[AgentResult] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)
    regime_type: str = "unknown"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "grade": self.grade,
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "position_size_usd": self.position_size_usd,
            "leverage": self.leverage,
            "risk_reward": round(self.risk_reward, 2),
            "contributing_factors": self.contributing_factors,
            "failed_agents": self.failed_agents,
            "regime_type": self.regime_type,
            "timestamp": self.timestamp,
            "agent_details": [
                {
                    "agent": r.agent_name,
                    "direction": r.direction,
                    "confidence": round(r.confidence, 4),
                    "reasoning": r.reasoning,
                    "error": r.error,
                }
                for r in self.agent_results
            ],
        }


# ── Orchestrator ───────────────────────────────────────────

class Orchestrator:
    """
    Main HQIP orchestrator.

    For each (symbol, timeframe) pair:
      1. Fetch OHLCV data via DataPlatform
      2. Calculate indicators
      3. Run all technical agents in sequence (fail-safe)
      4. Run whale & news agents
      5. Optionally run ML agent
      6. Feed results → ConsensusManager → final Signal
    """

    # Technical agent name → import path (lazy-loaded)
    TECHNICAL_AGENTS = {
        "trend": ("hqip.agents.trend_agent", "TrendAgent"),
        "momentum": ("hqip.agents.momentum_agent", "MomentumAgent"),
        "volume": ("hqip.agents.volume_agent", "VolumeAgent"),
        "volatility": ("hqip.agents.volatility_agent", "VolatilityAgent"),
        "pattern": ("hqip.agents.pattern_agent", "PatternAgent"),
        "market_structure": ("hqip.agents.market_structure_agent", "MarketStructureAgent"),
        "regime": ("hqip.agents.regime_agent", "RegimeAgent"),
    }

    WHALE_AGENT = ("hqip.agents.whale_agent", "WhaleAgent")
    NEWS_AGENT = ("hqip.agents.news_agent", "NewsAgent")
    ML_AGENT = ("hqip.agents.ml_agent", "MLAgent")

    def __init__(self, symbols: Optional[List[str]] = None,
                 timeframes: Optional[List[str]] = None,
                 capital: Optional[float] = None):
        """
        Parameters
        ----------
        symbols : list[str], optional
            Override default symbols from config.
        timeframes : list[str], optional
            Override default timeframes from config.
        capital : float, optional
            Trading capital (default from config).
        """
        self.symbols = symbols or SYMBOLS
        self.timeframes = timeframes or TIMEFRAMES
        self.capital = capital

        # Lazy-loaded singletons
        self._data_platform = None
        self._consensus = None
        self._risk_manager = None
        self._agents: Dict[str, Any] = {}

        # Agent instantiation cache
        self._agent_instances: Dict[str, Any] = {}

    # ── Lazy loaders ────────────────────────────────────

    @property
    def data_platform(self):
        if self._data_platform is None:
            from hqip.data_platform import DataPlatform
            self._data_platform = DataPlatform()
        return self._data_platform

    @property
    def consensus(self):
        if self._consensus is None:
            from hqip.consensus import ConsensusManager
            self._consensus = ConsensusManager()
        return self._consensus

    @property
    def risk_manager(self):
        if self._risk_manager is None:
            from hqip.risk_manager import RiskManager
            self._risk_manager = RiskManager(capital=self.capital)
        return self._risk_manager

    def _get_agent(self, key: str, module_path: str, class_name: str):
        """Get or lazily instantiate an agent."""
        if key not in self._agent_instances:
            try:
                import importlib
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self._agent_instances[key] = cls()
            except Exception as e:
                logger.warning("Failed to load agent %s: %s", key, e)
                return None
        return self._agent_instances.get(key)

    # ── Public scan methods ─────────────────────────────

    def scan_all(self) -> List[Signal]:
        """Scan all configured symbols across all timeframes."""
        signals = []
        total = len(self.symbols)
        for idx, symbol in enumerate(self.symbols, 1):
            logger.info("[%d/%d] Scanning %s …", idx, total, symbol)
            try:
                symbol_signals = self.scan_symbol(symbol)
                signals.extend(symbol_signals)
            except Exception as e:
                logger.error("Error scanning %s: %s\n%s", symbol, e, traceback.format_exc())
        return signals

    def scan_symbol(self, symbol: str) -> List[Signal]:
        """Scan a single symbol across all configured timeframes."""
        signals = []
        for tf in self.timeframes:
            try:
                signal = self.scan_single(symbol, tf)
                signals.append(signal)
            except Exception as e:
                logger.error("Error scanning %s/%s: %s", symbol, tf, e)
                signals.append(self._error_signal(symbol, tf, str(e)))
        return signals

    def scan_single(self, symbol: str, timeframe: str) -> Signal:
        """
        Run the full analysis pipeline for one (symbol, timeframe) pair.

        Returns a Signal with all agent results merged via consensus.
        """
        logger.info("▶ %s %s — fetching data", symbol, timeframe)

        # ── 1. Fetch data ───────────────────────────────
        try:
            df = self.data_platform.fetch_ohlcv(symbol, timeframe)
            if df is None or df.empty:
                logger.warning("No data for %s %s", symbol, timeframe)
                return self._error_signal(symbol, timeframe, "No data returned from exchange")
        except Exception as e:
            logger.error("Data fetch failed for %s %s: %s", symbol, timeframe, e)
            return self._error_signal(symbol, timeframe, f"Data fetch failed: {e}")

        # ── 2. Calculate indicators ─────────────────────
        try:
            from hqip.indicators import calculate_all_indicators
            df = calculate_all_indicators(df)
        except Exception as e:
            logger.error("Indicator calculation failed for %s %s: %s", symbol, timeframe, e)
            return self._error_signal(symbol, timeframe, f"Indicator calc failed: {e}")

        current_price = float(df["close"].iloc[-1])

        # ── 3. Run technical agents ─────────────────────
        agent_results: List[AgentResult] = []
        for agent_key, (mod_path, cls_name) in self.TECHNICAL_AGENTS.items():
            result = self._run_agent(agent_key, mod_path, cls_name, df, current_price)
            agent_results.append(result)

        # ── 4. Run whale agent ──────────────────────────
        whale_key = "whale"
        whale_mod, whale_cls = self.WHALE_AGENT
        whale_result = self._run_agent(whale_key, whale_mod, whale_cls, df, current_price)
        agent_results.append(whale_result)

        # ── 5. Run news agent ───────────────────────────
        news_key = "news"
        news_mod, news_cls = self.NEWS_AGENT
        news_result = self._run_agent(news_key, news_mod, news_cls, df, current_price)
        agent_results.append(news_result)

        # ── 6. Optionally run ML agent ──────────────────
        try:
            ml_key = "ml"
            ml_mod, ml_cls = self.ML_AGENT
            ml_result = self._run_agent(ml_key, ml_mod, ml_cls, df, current_price)
            if ml_result is not None and not ml_result.failed:
                agent_results.append(ml_result)
            else:
                logger.info("ML agent skipped / failed — continuing without it")
        except Exception:
            logger.info("ML agent not available — continuing without it")

        # ── 7. Feed to consensus ────────────────────────
        try:
            consensus_output = self.consensus.calculate(agent_results)
        except Exception as e:
            logger.error("Consensus failed for %s %s: %s", symbol, timeframe, e)
            return self._error_signal(symbol, timeframe, f"Consensus failed: {e}")

        direction = consensus_output.get("direction", "NO_TRADE")
        confidence = consensus_output.get("confidence", 0.0)
        agreement = consensus_output.get("agreement", 0.0)
        contributing_factors = consensus_output.get("contributing_factors", [])
        regime_type = consensus_output.get("regime_type", "unknown")

        # ── 8. Determine grade ──────────────────────────
        grade = self._determine_grade(confidence, agreement)

        # ── 9. Risk parameters ──────────────────────────
        failed_agents = [r.agent_name for r in agent_results if r.failed]

        if direction == "NO_TRADE":
            return Signal(
                symbol=symbol,
                timeframe=timeframe,
                direction="NO_TRADE",
                grade="NO_GRADE",
                confidence=confidence,
                entry_price=current_price,
                stop_loss=0.0,
                take_profit_1=0.0,
                take_profit_2=0.0,
                take_profit_3=0.0,
                position_size_usd=0.0,
                leverage=0,
                risk_reward=0.0,
                agent_results=agent_results,
                contributing_factors=contributing_factors,
                failed_agents=failed_agents,
                regime_type=regime_type,
            )

        try:
            risk = self.risk_manager.calculate_risk(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                confidence=confidence,
                grade=grade,
                agent_results=agent_results,
            )
        except Exception as e:
            logger.error("Risk calc failed: %s — using fallback", e)
            risk = self._fallback_risk(direction, current_price, confidence)

        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            grade=grade,
            confidence=confidence,
            entry_price=risk.get("entry_price", current_price),
            stop_loss=risk.get("stop_loss", 0.0),
            take_profit_1=risk.get("take_profit_1", 0.0),
            take_profit_2=risk.get("take_profit_2", 0.0),
            take_profit_3=risk.get("take_profit_3", 0.0),
            position_size_usd=risk.get("position_size", 0.0),
            leverage=risk.get("leverage", 1),
            risk_reward=risk.get("risk_reward", 0.0),
            agent_results=agent_results,
            contributing_factors=contributing_factors,
            failed_agents=failed_agents,
            regime_type=regime_type,
        )

    # ── Agent runner (fail-safe) ────────────────────────

    def _run_agent(self, key: str, module_path: str, class_name: str,
                   df: Any, current_price: float) -> AgentResult:
        """
        Run a single agent, catching any errors so one failure
        doesn't block the rest.
        """
        agent = self._get_agent(key, module_path, class_name)
        if agent is None:
            return AgentResult(
                agent_name=key,
                direction="neutral",
                confidence=0.0,
                reasoning="Agent could not be loaded",
                error=f"Failed to load {class_name} from {module_path}",
            )

        try:
            result = agent.analyze(df, current_price)
            # Normalise result to AgentResult
            if isinstance(result, AgentResult):
                return result
            if isinstance(result, dict):
                return AgentResult(
                    agent_name=key,
                    direction=result.get("direction", "neutral"),
                    confidence=float(result.get("confidence", 0.0)),
                    weight=float(result.get("weight", 1.0)),
                    reasoning=result.get("reasoning", ""),
                    data=result.get("data", {}),
                )
            return AgentResult(
                agent_name=key,
                direction="neutral",
                confidence=0.0,
                reasoning=f"Unexpected return type: {type(result)}",
                error=f"Unexpected return type: {type(result)}",
            )
        except Exception as e:
            logger.warning("Agent %s failed: %s", key, e)
            return AgentResult(
                agent_name=key,
                direction="neutral",
                confidence=0.0,
                reasoning="Agent execution failed",
                error=str(e),
            )

    # ── Helpers ─────────────────────────────────────────

    @staticmethod
    def _determine_grade(confidence: float, agreement: float) -> str:
        for grade, thresholds in GRADE_THRESHOLDS.items():
            if (confidence >= thresholds["confidence"]
                    and agreement >= thresholds["agreement"]):
                return grade
        return "NO_GRADE"

    @staticmethod
    def _fallback_risk(direction: str, entry_price: float,
                       confidence: float) -> Dict[str, Any]:
        """Minimal risk calculation if RiskManager fails."""
        if direction == "LONG":
            sl = entry_price * 0.97
            tp1 = entry_price * 1.03
            tp2 = entry_price * 1.05
            tp3 = entry_price * 1.08
        else:
            sl = entry_price * 1.03
            tp1 = entry_price * 0.97
            tp2 = entry_price * 0.95
            tp3 = entry_price * 0.92

        risk = abs(entry_price - sl)
        reward = abs(tp1 - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        return {
            "entry_price": entry_price,
            "stop_loss": round(sl, 8),
            "take_profit_1": round(tp1, 8),
            "take_profit_2": round(tp2, 8),
            "take_profit_3": round(tp3, 8),
            "position_size": round(10000 * confidence, 2),
            "leverage": 5,
            "risk_reward": round(rr, 2),
        }

    @staticmethod
    def _error_signal(symbol: str, timeframe: str, error_msg: str) -> Signal:
        """Produce a NO_TRADE signal when something goes wrong."""
        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            direction="NO_TRADE",
            grade="NO_GRADE",
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            take_profit_3=0.0,
            position_size_usd=0.0,
            leverage=0,
            risk_reward=0.0,
            contributing_factors=[f"⚠️ Error: {error_msg}"],
            failed_agents=[],
            regime_type="error",
        )

    # ── Format ──────────────────────────────────────────

    def format_signal(self, signal: Signal) -> str:
        """Return a Telegram-Markdown formatted signal string."""
        from hqip.display import format_signal_telegram
        return format_signal_telegram(signal)
