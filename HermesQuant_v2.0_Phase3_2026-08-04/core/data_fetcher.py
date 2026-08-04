"""OKX Data Fetcher"""
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

tz = timezone(timedelta(hours=3, minutes=30))
BASE = "https://www.okx.com/api/v5"

def fetch_candles(instId="ETH-USDT-SWAP", bar="15m", limit=200):
    r = requests.get(f"{BASE}/market/candles",
                     params={"instId": instId, "bar": bar, "limit": limit},
                     timeout=10)
    data = r.json()["data"]
    rows = [{"open": float(c[1]), "high": float(c[2]),
             "low": float(c[3]), "close": float(c[4]),
             "volume": float(c[5])} for c in reversed(data)]
    return pd.DataFrame(rows)

def fetch_ticker(instId="ETH-USDT-SWAP"):
    r = requests.get(f"{BASE}/market/ticker",
                     params={"instId": instId}, timeout=10)
    d = r.json()["data"][0]
    return {"last": float(d["last"]), "high24h": float(d["high24h"]),
            "low24h": float(d["low24h"])}
