"""Signal Builder — Final SignalOutput with SL/TP"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import SignalOutput, AgentOutput
from core import indicators as ind
from pipeline.probability import bayesian_combine, agent_to_prob, quarter_kelly, expected_value
from pipeline.geometry import agent_to_vector, convergence, resultant, signal_strength
import numpy as np

def build_signal(agent_outputs, df, capital=10, leverage=20, max_loss=3.0):
    if df is None or len(df) < 20:
        return SignalOutput()
    
    price = df.iloc[-1]["close"]
    atr_val = ind.atr(df).iloc[-1]
    
    # Bayesian
    priors = [agent_to_prob(r.direction, r.confidence, r.score) for r in agent_outputs]
    weights = [r.weight for r in agent_outputs]
    p_up, p_down = bayesian_combine(priors, weights)
    
    # Vector geometry
    vectors = [agent_to_vector(r.direction, r.confidence, r.score) for r in agent_outputs]
    res = resultant(vectors, weights)
    sig = signal_strength(res)
    conv = convergence(vectors)
    
    # Vote
    buy_w = sum(r.weight for r in agent_outputs if r.direction == "BUY")
    sell_w = sum(r.weight for r in agent_outputs if r.direction == "SELL")
    total = buy_w + sell_w if buy_w + sell_w > 0 else 1
    
    if buy_w > sell_w:
        direction = "BUY"
    elif sell_w > buy_w:
        direction = "SELL"
    else:
        direction = "WAIT"
    
    # Thresholds
    if p_up < 0.65 and p_down < 0.65:
        direction = "WAIT"
    if conv < 0.55:
        direction = "WAIT"
    
    # SL/TP
    if direction == "BUY":
        sl = price - atr_val * 1.5
        tp1 = price + atr_val * 2.0
        tp2 = price + atr_val * 3.0
        tp3 = price + atr_val * 5.0
    elif direction == "SELL":
        sl = price + atr_val * 1.5
        tp1 = price - atr_val * 2.0
        tp2 = price - atr_val * 3.0
        tp3 = price - atr_val * 5.0
    else:
        sl = tp1 = tp2 = tp3 = 0
    
    # Kelly + EV
    if direction != "WAIT":
        pw = p_up if direction == "BUY" else p_down
        kelly = quarter_kelly(pw, 2.0)
        ev = expected_value(pw, 1-pw, 2.0, 1.0)
        risk = max_loss * min(kelly, 1.0)  # Quarter-Kelly applied correctly
        pos = risk / (abs(sl - price) / price) if sl != price else 0
    else:
        kelly = ev = risk = pos = 0
    
    ev_list = ["Bayes: P_UP=%.1f%% P_DOWN=%.1f%%" % (p_up*100, p_down*100),
               "Conv=%.1f%%" % (conv*100),
               "Vote: BUY_w=%.1f SELL_w=%.1f" % (buy_w, sell_w)]
    
    return SignalOutput(
        direction=direction, entry=round(price, 2),
        sl=round(sl, 2), tp1=round(tp1, 2), tp2=round(tp2, 2), tp3=round(tp3, 2),
        confidence=round(sig["confidence"], 1), kelly=round(kelly*100, 1),
        ev=round(ev, 3), convergence=round(conv*100, 1),
        p_up=round(p_up, 3), p_down=round(p_down, 3),
        risk_per_trade=round(risk, 2), position_size=round(pos, 4),
        evidence=ev_list
    )
