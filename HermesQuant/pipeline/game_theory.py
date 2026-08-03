"""Market Game Theory — Nash Equilibrium"""
def detect_whale_action(vol_ratio, price_change, wick_ratio):
    if vol_ratio > 2.0 and abs(price_change) < 0.1:
        return 'ABSORB'
    if vol_ratio > 1.5 and abs(price_change) > 0.5:
        return 'MANIPULATE' if wick_ratio > 0.6 else 'DUMP'
    return 'NEUTRAL'

def nash_check(bull, bear, whale=0.3):
    t = bull + bear + whale
    if t == 0: return {'state': 'EQUILIBRIUM', 'opp': 0}
    br, ber, wr = bull/t, bear/t, whale/t
    dom = max(br, ber, wr)
    if dom > 0.6:
        d = 'bulls' if br == dom else ('bears' if ber == dom else 'whales')
        return {'state': 'DISEQUILIBRIUM', 'dominant': d, 'opp': dom - 0.5}
    return {'state': 'EQUILIBRIUM', 'opp': 0}
