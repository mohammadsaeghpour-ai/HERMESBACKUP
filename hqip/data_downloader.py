#!/usr/bin/env python3
"""
HQIP Data Downloader — Fetch complete historical market data
Sources: Binance Vision (spot), OKX API, CryptoDataDownload
"""
import os, sys, zipfile, io, time
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "/data/workspace/hqip/data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_binance_vision(symbol, interval, start_date, end_date):
    """Download from data.binance.vision"""
    all_data = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name), header=None,
                    names=["timestamp","open","high","low","close","volume","close_time",
                           "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
                all_data.append(df)
                print(f"  OK {date_str} ({len(df)} candles)")
            else:
                print(f"  SKIP {date_str}")
        except Exception as e:
            print(f"  ERR {date_str}: {e}")
        
        current += timedelta(days=1)
        time.sleep(0.3)  # Rate limit
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], unit="ms")
        combined.drop_duplicates(subset=["timestamp"], inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined.set_index("timestamp", inplace=True)
        return combined
    return None


def download_okx_api(symbol, tf, limit=300):
    """Fetch from OKX via ccxt"""
    import ccxt
    ex = ccxt.okx()
    all_data = []
    since = None
    
    for batch in range(10):
        try:
            ohlcv = ex.fetch_ohlcv(symbol.replace("USDT","-USDT-SWAP") if symbol.endswith("USDT") else symbol, tf, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"  OKX batch {batch+1}: {len(ohlcv)} candles")
            time.sleep(0.5)
        except:
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df
    return None


print("=" * 50)
print("  HQIP DATA DOWNLOADER")
print("=" * 50)

# ── 1. Download from Binance Vision (90 days) ──
print("\n1. Binance Vision — 90 days BTCUSDT 15m + 1h + 4h + 1d")
end_date = datetime(2026, 7, 28)
start_date = end_date - timedelta(days=90)

for tf in ["15m", "1h", "4h", "1d"]:
    print(f"\n  [{tf}] Downloading...")
    df = download_binance_vision("BTCUSDT", tf, start_date, end_date)
    if df is not None and len(df) > 0:
        filepath = f"{DATA_DIR}/BTCUSDT_{tf}_binance.csv"
        df.to_csv(filepath)
        print(f"  SAVED: {filepath} ({len(df)} candles, {df.index[0]} to {df.index[-1]})")
    else:
        print(f"  FAILED")

# ── 2. Download from Binance Vision — ETH ──
print("\n2. Binance Vision — 90 days ETHUSDT")
for tf in ["15m", "1h"]:
    print(f"\n  [{tf}] Downloading ETH...")
    df = download_binance_vision("ETHUSDT", tf, start_date, end_date)
    if df is not None and len(df) > 0:
        filepath = f"{DATA_DIR}/ETHUSDT_{tf}_binance.csv"
        df.to_csv(filepath)
        print(f"  SAVED: {filepath} ({len(df)} candles)")

# ── 3. Check data completeness ──
print("\n" + "=" * 50)
print("  DATA SUMMARY")
print("=" * 50)

for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith(".csv"):
        fp = os.path.join(DATA_DIR, f)
        df = pd.read_csv(fp)
        size_mb = os.path.getsize(fp) / 1024 / 1024
        print(f"  {f}: {len(df)} candles, {size_mb:.1f} MB")

print("\nDONE.")
