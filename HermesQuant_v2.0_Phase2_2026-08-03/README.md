# HermesQuant v2.0

Vertical Pipeline Trading System for BTC/ETH

## Architecture
```
Stage 0: Independent Agents (Trend, Momentum, Volume, Volatility, Pattern, DL)
    ↓
Stage 1: Meta Agents (Regime, Structure, Whale)
    ↓
Stage 2: Structure Agents (SMC, Liquidity, Wyckoff, MathBrain)
    ↓
Stage 3: Decision Agents (GameTheory, SmartAction, ML)
    ↓
Stage 4: Risk + Signal Output (Kelly, Tail Risk, SL/TP)
```

## Usage
```bash
python cli.py --symbol ETH-USDT-SWAP --timeframe 15m
python cli.py --symbol BTC-USDT-SWAP --timeframe 1H
```

## Key Features
- 17 agents in 5-stage vertical pipeline
- Bayesian probability combining
- Vector geometry convergence
- Game theory Nash equilibrium
- 7-gate filter system
- Kelly criterion position sizing
