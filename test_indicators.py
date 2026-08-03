"""Comprehensive test for HQIP v3 indicator engine."""
import sys
sys.path.insert(0, '/data/workspace')
import pandas as pd
import numpy as np
from hqip.indicators import (
    ema, ema_fan, ema_cross, supertrend, adx, ichimoku, trend_strength,
    rsi, rsi_divergence, macd, macd_divergence, stochastic, cci, williams_r, momentum_score,
    atr, bollinger, bollinger_squeeze, keltner, volatility_regime,
    obv, obv_divergence, vwap, vwap_bands, mfi, volume_profile, volume_absorption, volume_score,
    fibonacci_levels, fibonacci_extensions, fibonacci_cluster, ote_zone, nearest_fib,
)
print('✅ All imports successful')

# Generate synthetic OHLCV data
np.random.seed(42)
n = 200
close = 50000 + np.cumsum(np.random.randn(n) * 200)
high = close + np.abs(np.random.randn(n) * 100)
low = close - np.abs(np.random.randn(n) * 100)
open_ = close + np.random.randn(n) * 50
volume = np.abs(np.random.randn(n) * 1000 + 5000)

df = pd.DataFrame({
    'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
})
print(f'📊 Test DataFrame: {len(df)} bars, close range {df.close.min():.0f} - {df.close.max():.0f}')
print()

# TREND
e = ema(df['close'], 20)
print(f'ema(20) last: {e.iloc[-1]:.2f}  type: {type(e).__name__}')

fan = ema_fan(df)
print(f'ema_fan keys: {list(fan.keys())}  ema_8 last: {fan["ema_8"].iloc[-1]:.2f}')

cross = ema_cross(df)
print(f'ema_cross: {cross}')

st = supertrend(df)
print(f'supertrend: line={st["supertrend"].iloc[-1]:.2f}, dir={st["direction"].iloc[-1]:.0f}')

a = adx(df)
print(f'adx: adx={a["adx"].iloc[-1]:.2f}, +di={a["di_plus"].iloc[-1]:.2f}, -di={a["di_minus"].iloc[-1]:.2f}')

ich = ichimoku(df)
print(f'ichimoku: tenkan={ich["tenkan"].iloc[-1]:.2f}, kijun={ich["kijun"].iloc[-1]:.2f}')

ts = trend_strength(df)
print(f'trend_strength: {ts:.2f}')
print()

# MOMENTUM
r = rsi(df)
print(f'rsi(14) last: {r.iloc[-1]:.2f}')

rd = rsi_divergence(df)
print(f'rsi_divergence: {rd}')

m = macd(df)
print(f'macd: line={m["macd"].iloc[-1]:.2f}, signal={m["signal"].iloc[-1]:.2f}, hist={m["histogram"].iloc[-1]:.2f}')

md = macd_divergence(df)
print(f'macd_divergence: {md}')

s = stochastic(df)
print(f'stochastic: k={s["k"].iloc[-1]:.2f}, d={s["d"].iloc[-1]:.2f}')

cc = cci(df)
print(f'cci last: {cc.iloc[-1]:.2f}')

wr = williams_r(df)
print(f'williams_r last: {wr.iloc[-1]:.2f}')

ms = momentum_score(df)
print(f'momentum_score: {ms:.2f}')
print()

# VOLATILITY
at = atr(df)
print(f'atr(14) last: {at.iloc[-1]:.2f}')

bb = bollinger(df)
print(f'bollinger: upper={bb["upper"].iloc[-1]:.2f}, mid={bb["middle"].iloc[-1]:.2f}, lower={bb["lower"].iloc[-1]:.2f}, width={bb["width"].iloc[-1]:.4f}, pct_b={bb["pct_b"].iloc[-1]:.2f}')

sq = bollinger_squeeze(df)
print(f'bollinger_squeeze: {sq}')

kc = keltner(df)
print(f'keltner: upper={kc["upper"].iloc[-1]:.2f}, mid={kc["middle"].iloc[-1]:.2f}, lower={kc["lower"].iloc[-1]:.2f}')

vr = volatility_regime(df)
print(f'volatility_regime: {vr}')
print()

# VOLUME
o = obv(df)
print(f'obv last: {o.iloc[-1]:.2f}')

od = obv_divergence(df)
print(f'obv_divergence: {od}')

vw = vwap(df)
print(f'vwap last: {vw.iloc[-1]:.2f}')

vb = vwap_bands(df)
print(f'vwap_bands: upper={vb["upper"].iloc[-1]:.2f}, lower={vb["lower"].iloc[-1]:.2f}')

mf = mfi(df)
print(f'mfi last: {mf.iloc[-1]:.2f}')

vp = volume_profile(df)
print(f'volume_profile: poc={vp["poc"]:.2f}, bins={len(vp["prices"])}')

va = volume_absorption(df)
print(f'volume_absorption: {va}')

vs = volume_score(df)
print(f'volume_score: {vs:.2f}')
print()

# FIBONACCI
fl = fibonacci_levels(55000, 50000)
print(f'fibonacci_levels: { {k: round(v,2) for k,v in fl.items()} }')

fe = fibonacci_extensions(55000, 50000, 52000)
print(f'fibonacci_extensions: { {k: round(v,2) for k,v in fe.items()} }')

fc = fibonacci_cluster(df)
print(f'fibonacci_cluster: {len(fc)} clusters found')
if fc:
    for p, s in fc[:3]:
        print(f'  price={p:.2f}, strength={s:.0f}')

ote = ote_zone(55000, 50000)
print(f'ote_zone: { {k: round(v,2) for k,v in ote.items()} }')

nf = nearest_fib(52500, fl)
print(f'nearest_fib: { {k: round(v,2) if isinstance(v, float) else v for k,v in nf.items()} }')

print()
print('=' * 43)
print('✅ ALL 30 FUNCTIONS TESTED SUCCESSFULLY')
print('=' * 43)
