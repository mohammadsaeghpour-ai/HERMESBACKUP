
"""
Deep Analysis of Weak Agents
Identify WHY they fail and IF they can be fixed
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import requests, pandas as pd, numpy as np
from core import indicators as ind
from core.data_types import AgentOutput

# Import weak agents
from agents.stage2_structure.smc_agent import SMCAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage0_independent.pattern_agent import PatternAgent

def fetch_all(instId, bar, limit=300):
    r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                     params={"instId": instId, "bar": bar, "limit": limit}, timeout=15)
    data = r.json()["data"]
    rows = [{"open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
             "close": float(c[4]), "volume": float(c[5]), "ts": int(c[0])}
            for c in reversed(data)]
    return pd.DataFrame(rows)

# Test each agent individually
for instId in ["ETH-USDT-SWAP", "BTC-USDT-SWAP"]:
    df = fetch_all(instId, "15m", 280)
    
    print("="*60)
    print("  %s — Agent Deep Analysis" % instId)
    print("="*60)
    
    agents = {
        "SMC": SMCAgent(),
        "Liquidity": LiquidityAgent(),
        "MathBrain": MathBrainAgent(),
        "Pattern": PatternAgent(),
    }
    
    for name, agent in agents.items():
        buy_correct = 0
        buy_wrong = 0
        sell_correct = 0
        sell_wrong = 0
        neutral = 0
        errors = 0
        signal_details = []
        
        for i in range(50, len(df) - 10):
            window = df.iloc[i-50:i+1].copy().reset_index(drop=True)
            price = window["close"].iloc[-1]
            future = df["close"].iloc[i+1:i+6].values
            
            if len(future) < 3:
                continue
            
            try:
                r = agent.analyze(window, instId, "15m")
            except Exception as e:
                errors += 1
                continue
            
            if r.direction == "NEUTRAL":
                neutral += 1
                continue
            
            # Check if signal was correct
            future_return = (future[-1] - price) / price * 100
            signal_correct = (r.direction == "BUY" and future_return > 0) or \
                           (r.direction == "SELL" and future_return < 0)
            
            if r.direction == "BUY":
                if signal_correct:
                    buy_correct += 1
                else:
                    buy_wrong += 1
            elif r.direction == "SELL":
                if signal_correct:
                    sell_correct += 1
                else:
                    sell_wrong += 1
            
            signal_details.append({
                "dir": r.direction, "conf": r.confidence,
                "score": r.score, "future": future_return,
                "correct": signal_correct
            })
        
        total_signals = buy_correct + buy_wrong + sell_correct + sell_wrong
        total_correct = buy_correct + sell_correct
        accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
        
        print("\n  %s Agent:" % name)
        print("    Signals: %d (neutral: %d, errors: %d)" % (total_signals, neutral, errors))
        print("    BUY: %d/%d (%.0f%%) | SELL: %d/%d (%.0f%%)" % (
            buy_correct, buy_correct+buy_wrong, 
            buy_correct/(buy_correct+buy_wrong)*100 if buy_correct+buy_wrong > 0 else 0,
            sell_correct, sell_correct+sell_wrong,
            sell_correct/(sell_correct+sell_wrong)*100 if sell_correct+sell_wrong > 0 else 0))
        print("    Overall: %d/%d (%.0f%%)" % (total_correct, total_signals, accuracy))
        
        # Analyze what the agent is doing wrong
        if signal_details:
            # Check confidence distribution
            correct_conf = [s["conf"] for s in signal_details if s["correct"]]
            wrong_conf = [s["conf"] for s in signal_details if not s["correct"]]
            
            if correct_conf:
                print("    Avg correct confidence: %.0f%%" % np.mean(correct_conf))
            if wrong_conf:
                print("    Avg wrong confidence: %.0f%%" % np.mean(wrong_conf))
            
            # Check future returns
            correct_returns = [s["future"] for s in signal_details if s["correct"]]
            wrong_returns = [s["future"] for s in signal_details if not s["correct"]]
            
            if correct_returns:
                print("    Avg correct return: %.3f%%" % np.mean(correct_returns))
            if wrong_returns:
                print("    Avg wrong return: %.3f%%" % np.mean(wrong_returns))
            
            # Check if agent is better at BUY or SELL
            buy_acc = buy_correct/(buy_correct+buy_wrong)*100 if buy_correct+buy_wrong > 0 else 0
            sell_acc = sell_correct/(sell_correct+sell_wrong)*100 if sell_correct+sell_wrong > 0 else 0
            
            if buy_acc > sell_acc + 20:
                print("    >> Agent is BETTER at BUY signals")
            elif sell_acc > buy_acc + 20:
                print("    >> Agent is BETTER at SELL signals")
            else:
                print("    >> Agent is BALANCED (both weak)")
