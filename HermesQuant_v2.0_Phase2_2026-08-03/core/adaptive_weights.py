"""Adaptive Weight System — learns from recent performance"""
import json
import os

WEIGHTS_FILE = "/data/workspace/HermesQuant/agent_weights.json"

DEFAULT_WEIGHTS = {
    "Trend": 1.5, "Momentum": 1.3, "Volume": 1.3,
    "Volatility": 1.0, "DLForecast": 1.0,
    "Regime": 1.0, "MarketStructure": 1.4, "Whale": 1.5,
    "RSI_Divergence": 1.2, "BB_Squeeze": 1.1,
    "Liquidity": 1.5, "Wyckoff": 1.4, "MathBrain": 1.4,
    "GameTheory": 1.3, "SmartAction": 1.7, "ML": 1.2,
}

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()

def save_weights(weights):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)

def update_weights(agent_name, correct):
    """Update weight based on last trade result"""
    w = load_weights()
    if agent_name not in w:
        w[agent_name] = 1.0
    
    if correct:
        w[agent_name] = min(w[agent_name] * 1.05, 3.0)  # Increase by 5%, max 3.0
    else:
        w[agent_name] = max(w[agent_name] * 0.95, 0.3)  # Decrease by 5%, min 0.3
    
    save_weights(w)
    return w[agent_name]

def get_weights():
    return load_weights()
