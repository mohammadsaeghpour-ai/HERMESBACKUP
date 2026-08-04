"""Signal Vector Geometry"""
import numpy as np
import math

def agent_to_vector(direction, confidence, score):
    d = 1.0 if direction == "BUY" else (-1.0 if direction == "SELL" else 0.0)
    return np.array([d, confidence / 100.0, score])

def vector_angle(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))

def convergence(vectors):
    if len(vectors) < 2: return 0
    total, cnt = 0, 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            total += 1.0 - (vector_angle(vectors[i], vectors[j]) / 180.0)
            cnt += 1
    return total / cnt if cnt > 0 else 0

def resultant(vectors, weights):
    r = np.zeros(3)
    tw = sum(weights)
    for v, w in zip(vectors, weights):
        r += v * (w / tw) if tw > 0 else v
    return r

def signal_strength(resultant):
    mag = np.linalg.norm(resultant)
    d = resultant[0]
    if d > 0.3: sig = "BUY"
    elif d < -0.3: sig = "SELL"
    else: sig = "NEUTRAL"
    return {'direction': sig, 'magnitude': round(mag, 3),
            'confidence': round(resultant[1] * 100, 1),
            'strength': round(resultant[2], 3)}
