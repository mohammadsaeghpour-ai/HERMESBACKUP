"""Data types for HermesQuant"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class AgentOutput:
    name: str = ""
    direction: str = "NEUTRAL"
    confidence: float = 0.0
    score: float = 0.0
    weight: float = 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class SignalOutput:
    direction: str = "WAIT"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    confidence: float = 0.0
    kelly: float = 0.0
    ev: float = 0.0
    regime: str = "UNKNOWN"
    convergence: float = 0.0
    filters_passed: int = 0
    filters_total: int = 7
    evidence: List[str] = field(default_factory=list)
    p_up: float = 0.5
    p_down: float = 0.5
    risk_per_trade: float = 0.0
    position_size: float = 0.0
