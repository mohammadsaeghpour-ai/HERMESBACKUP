"""
Data Fetcher — OKX API
"""
import requests
import pandas as pd
import time


def fetch_candles(symbol="BTC-USDT-SWAP", timeframe="15m", limit=200):
    """Fetch OHLCV candles from OKX"""
    tf_map = {"5m": "5m", "15m": "15m", "30m": "30m", "1H": "1H", "4H": "4H", "1h": "1H", "4h": "4H"}
    tf = tf_map.get(timeframe, timeframe)
    
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": symbol, "bar": tf, "limit": str(min(limit, 300))}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("code") != "0" or not data.get("data"):
            return None
        
        rows = []
        for c in reversed(data["data"]):
            rows.append({
                "ts": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        return df
    
    except Exception as e:
        print("Error fetching %s: %s" % (symbol, e))
        return None
