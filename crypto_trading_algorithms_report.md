# GitHub Crypto Trading Algorithm Research Report
## Repos with 70%+ Win Rate Claims & Backtest Results

Generated: August 4, 2026

---

## TIER 1: HIGH-ACCURACY ML TRADING SYSTEMS (70%+ Claims)

### 1. stefanoviana/deepalpha ⭐39
- **URL**: https://github.com/stefanoviana/deepalpha
- **Accuracy Claim**: 84.6% walk-forward validated directional accuracy (Pro); 70.9% (base)
- **ML Stack**: LightGBM + XGBoost + Random Forest ensemble
- **Features**: 72 engineered features from L2 orderbook data, funding rates, market microstructure
  - RSI, ATR, EMA, Momentum, Volume
  - Hurst exponent, VPIN, volatility regime, fractal efficiency, multi-timeframe alignment
  - HMM 3-state regime detection (bull/bear/sideways)
  - TFT (Temporal Fusion Transformer) + TransformerGRU neural models
- **Validation**: 4-window walk-forward validation, Optuna hyperparameter tuning (200+ trials)
- **Exchanges**: 18 (Bybit, Binance, OKX, Gate.io, KuCoin, etc.)
- **Pump Scanner**: Monitors 500+ coins for volume spikes (5x+ above average)
- **Live Results**: ZEREBRO +7.7% in 72min, B Token +12.7% in 2min
- **CAUTION**: Pro model (84.6%) requires $39/mo subscription; free version has 15 basic features only

### 2. SashRajj/Momentum-Based-Crypto-Trading ⭐5
- **URL**: https://github.com/SashRajj/Momentum-Based-Crypto-Trading
- **Models**: LSTM, SVM, Random Forest
- **Backtest Results** (BTC/ETH/BNB/SOL/ADA):
  - Gross Return: 1,137.37%
  - Net Return: 769.64%
  - Monthly Win Rate: 58.33%
  - Gross Sharpe: 1.93 / Net Sharpe: 1.69
  - Max Drawdown: -29.11%
- **Note**: Monthly win rate 58%, but high absolute returns via momentum captures

---

## TIER 2: FREQTRADE STRATEGIES (Proven Backtest Results)

### 3. iterativv/NostalgiaForInfinity ⭐3,350
- **URL**: https://github.com/iterativv/NostalgiaForInfinity
- **Framework**: Freqtrade (5m timeframe)
- **Backtest**: Results posted in commit messages; community-validated
- **Recommendations**: 6-12 open trades, 40-80 pair volume pairlist, USDT pairs
- **Status**: Actively maintained, Docker auto-updater included
- **Key**: Most popular community freqtrade strategy; extensively backtested

### 4. freqtrade/freqtrade-strategies ⭐5,338
- **URL**: https://github.com/freqtrade/freqtrade-strategies
- **Backtest Results** (2018-01-10 to 2018-01-30):
  - Strategy 005: 180 buys, 1.16% avg profit, highest total profit
  - Strategy 002: 9 buys, 3.21% avg profit (highest per-trade)
- **Note**: Educational; results on short backtest period

### 5. paulcpk/freqtrade-strategies-that-work ⭐327
- **URL**: https://github.com/paulcpk/freqtrade-strategies-that-work
- **Backtest** (2018-03 to 2020-03, 1h, 8 USDT pairs):
  - EMAPriceCrossoverWithThreshold: 272 buys, 1.31% avg, **118.53% total**
  - DoubleEMACrossoverWithTrend: 655 buys, 0.56% avg, **122.50% total**
  - MACDCrossoverWithTrend: 300 buys, 0.49% avg, 49.42% total
  - RSIDirectionalWithTrendSlow: 108 buys, 0.91% avg, 32.75% total
- **Key Insight**: EMA crossover with trend filter produces best risk-adjusted returns

### 6. Rikj000/MoniGoMani ⭐1,027
- **URL**: https://github.com/Rikj000/MoniGoMani
- **Framework**: Freqtrade with Weighted Hyperopt
- **Status**: Pre-release/experimental; heavy development
- **Key**: Genetic algorithm + weighted hyperparameter optimization framework

### 7. imsatoshi/GeneTrader ⭐198
- **URL**: https://github.com/imsatoshi/GeneTrader
- **Method**: Genetic Algorithm for strategy parameter optimization
- **Features**: Walk-forward validation, selection bar (luck test), checkpoint/resume
- **Key**: GA-based optimization that searches parameter spaces systematically

### 8. jilv220/BB_RPB_TSL ⭐213
- **URL**: https://github.com/jilv220/BB_RPB_TSL
- **Method**: Bollinger Bands + RSI Pullback + Trailing Stop Loss
- **Note**: Built on NostalgiaForInfinity; optimized for Kucoin
- **Warning**: Overfitted to Kucoin; needs re-hyperopt for other exchanges

### 9. nateemma/strategies ⭐433
- **URL**: https://github.com/nateemma/strategies
- **ML Stack**: Neural Networks + GANs for data augmentation
- **Key Finding**: Author's statistical study concludes "unlikely to do better through improved algorithms/models because there isn't any more information to extract from the data"
- **Framework**: Custom entry/exit processing with normalization
- **Note**: Honest assessment of diminishing returns from better models

### 10. eovie/freqtrade_strs ⭐649
- **URL**: https://github.com/eovie/freqtrade_strs
- **Status**: Active, with Binance copy-trading link (started 2026/07/25)
- **Key**: Has real copy-trading evidence on Binance

---

## TIER 3: ML-BASED TRADING FRAMEWORKS

### 11. AI4Finance-Foundation/FinRL ⭐15,916
- **URL**: https://github.com/AI4Finance-Foundation/FinRL
- **Framework**: Deep Reinforcement Learning for Finance
- **Status**: Original research framework; FinRL-X is production successor
- **Models**: DRL algorithms (PPO, A2C, DDPG, SAC, TD3)
- **Scope**: Stocks, crypto, forex, portfolio management
- **Key**: Academic paper-backed; most cited DRL trading framework

### 12. AI4Finance-Foundation/FinRL_DeepSeek ⭐328
- **URL**: https://github.com/benstaf/FinRL_DeepSeek
- **Paper**: "FinRL-DeepSeek: LLM-Infused Risk-Sensitive RL for Trading Agents"
- **Key**: Combines LLM reasoning with RL trading decisions

### 13. microsoft/qlib ⭐47,011
- **URL**: https://github.com/microsoft/qlib
- **Framework**: AI-oriented quantitative investment platform
- **Models**: DNN, LSTM, Transformer, GBDT, Linear models
- **Scope**: Primarily Chinese A-share markets; adaptable to crypto
- **Key**: Microsoft-backed; most starred quant platform

### 14. UFund-Me/Qbot ⭐18,245
- **URL**: https://github.com/UFund-Me/Qbot
- **Framework**: Built on Microsoft Qlib
- **Features**: AI quant research platform with full backtesting
- **Key**: Chinese-focused but architecture applicable globally

### 15. OpenByteInc/QuantDinger ⭐10,233
- **URL**: https://github.com/OpenByteInc/QuantDinger
- **Framework**: Full AI Trading OS
- **Stack**: Python 3.12, PostgreSQL, Redis, Docker Compose
- **Features**: Strategy generation → backtest → paper/live → monitoring
- **Key**: End-to-end production system with AI agents & MCP

---

## TIER 4: XGBOOST/LIGHTGBM SPECIFIC

### 16. tzelalouzeir/XGB_CryptoStrategy ⭐12
- **URL**: https://github.com/tzelalouzeir/XGB_CryptoStrategy
- **ML Model**: XGBoost for BTC position prediction (long/short/neutral)
- **Features**: 12 technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- **Backtest**: Simulated trading with feature importance visualization
- **Key**: Clean, educational XGBoost implementation for crypto

### 17. OVIE-web/MyTradingProject-crypto_trading_bot ⭐5
- **URL**: https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot
- **Stack**: FastAPI + XGBoost + PostgreSQL + LangGraph
- **Features**: ML-powered trade predictions, risk guardrails, portfolio monitoring
- **Key**: Production-oriented architecture with AI agent orchestration

### 18. Kowsi/Mind-Bot-Algorithmic-Crypto-Trading-Bot ⭐28
- **URL**: https://github.com/Kowsi/Mind-Bot-Algorithmic-Crypto-Trading-Bot
- **ML Models**: 2 Random Forest Classifiers
  - Model 1: EMA (close) + EMA (volume) + Bollinger Bands
  - Model 2: SMA + EMA + RSI + Stochastic + MACD + Bollinger Bands
- **Backtest**: Visualized results with buy/sell signals
- **Key**: Good starting point for ML trading bot architecture

---

## TIER 5: FEATURE ENGINEERING FOR CRYPTO

### 19. DeepAlpha V11 Feature Set (from stefanoviana/deepalpha)
The most comprehensive crypto feature engineering set documented:
- **Base (15 features)**: RSI, ATR, EMA, Momentum, Volume, SMA, MACD, Bollinger Bands
- **Extended (72 features)**:
  - L2 orderbook depth & imbalance
  - Funding rates
  - Market microstructure signals
  - Hurst exponent (mean-reversion vs trending)
  - VPIN (Volume-synchronized Probability of Informed Trading)
  - Volatility regime features
  - Fractal efficiency ratio
  - Multi-timeframe alignment
  - HMM regime detection (bull/bear/sideways)
- **Advanced (Pro/78 features)**: TFT + TransformerGRU neural features

### 20. LualApex624/drw-crypto-market-prediction ⭐2
- **URL**: https://github.com/LualApex624/drw-crypto-market-prediction
- **Method**: Ridge Regression on order book & trade data features
- **Focus**: Short-horizon cryptocurrency return prediction

### 21. markdregan/FreqAI-Marcos-Lopez-De-Prado ⭐88
- **URL**: https://github.com/markdregan/FreqAI-Marcos-Lopez-De-Prado
- **Focus**: "Advances in Financial Machine Learning" strategies in FreqAI
- **Key**: Academic feature engineering approaches from de Prado's book

### 22. just-nilux/awesome-freqtrade ⭐95
- **URL**: https://github.com/just-nilux/awesome-freqtrade
- **Key**: Curated collection of FreqAI + Freqtrade snippets
- **Includes**: RL integration, custom models, hyperopt tips, code snippets

---

## JESSE-AI ECOSYSTEM

### 23. jesse-ai/jesse ⭐8,290
- **URL**: https://github.com/jesse-ai/jesse
- **Description**: Advanced crypto trading bot in Python
- **Features**: Backtesting, paper trading, live trading
- **Note**: Strategy-specific repos are less popular than freqtrade ecosystem

### 24. ysdede/jesse_strategies ⭐71
- **URL**: https://github.com/ysdede/jesse_strategies
- **Note**: Collection of jesse strategies with subfolder documentation

---

## KEY TAKEAWAYS

### What Actually Works (Evidence-Based)
1. **EMA crossovers with trend filters** — Simple but consistent 100%+ returns over 2 years (paulcpk)
2. **NostalgiaForInfinity** — Community-validated, most battle-tested freqtrade strategy
3. **Walk-forward validation** is critical — random splits overfit badly
4. **Feature engineering > model complexity** — nateemma/strategies found diminishing returns from better models

### Red Flags / Caveats
- **84.6% accuracy** (DeepAlpha) requires paying $39/mo for pre-trained models; free version only has 15 basic features
- Most "win rate" claims on GitHub lack independent verification
- Walk-forward validated accuracy (DeepAlpha: 70.9%) is more honest than headline numbers (84.6%)
- GAN data augmentation (nateemma) suggests data scarcity is a fundamental limitation
- Many repos are clearly overfitted to specific periods/exchanges

### Best Starting Points by Use Case
- **Freqtrade ML**: robcaulk/freqai + jerome-benoit/freqai-strategies
- **XGBoost/LightGBM**: stefanoviana/deepalpha (features), tzelalouzeir/XGB_CryptoStrategy (simple)
- **Reinforcement Learning**: AI4Finance-Foundation/FinRL
- **Full Platform**: OpenByteInc/QuantDinger, microsoft/qlib
- **Feature Engineering**: DeepAlpha V11 feature set, markdregan/FreqAI-Marcos-Lopez-De-Prado
