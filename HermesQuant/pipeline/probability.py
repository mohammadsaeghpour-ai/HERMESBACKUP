"""Bayesian Probability Engine"""
import math

def bayesian_combine(priors, weights):
    if not priors: return 0.5, 0.5
    log_odds = 0
    tw = sum(weights)
    for (pu, pw), w in zip(priors, weights):
        if pu > 0 and pw > 0:
            log_odds += w * math.log(pu / pw)
    log_odds /= tw if tw > 0 else 1
    pu = 1 / (1 + math.exp(-log_odds))
    return pu, 1 - pu

def expected_value(p_win, p_loss, win_amt, loss_amt):
    return (p_win * win_amt) - (p_loss * loss_amt)

def kelly(p, b):
    if b <= 0: return 0
    return max(0, (p * b - (1 - p)) / b)

def quarter_kelly(p, b):
    return kelly(p, b) / 4

def agent_to_prob(direction, confidence, score):
    c = confidence / 100.0
    if direction == "BUY":
        pu = 0.5 + c * 0.5 * (1 + abs(score))
    elif direction == "SELL":
        pu = 0.5 - c * 0.5 * (1 + abs(score))
    else:
        pu = 0.5
    return min(max(pu, 0.01), 0.99), 1 - min(max(pu, 0.01), 0.99)
