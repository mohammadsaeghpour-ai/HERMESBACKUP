"""
Data Fetcher — OKX API with pagination for 1+ month of data
"""
import requests
import pandas as pd
import time


def fetch_candles(symbol="BTC-USDT-SWAP", timeframe="15m", limit=2880):
    """Fetch OHLCV candles from OKX with pagination"""
    tf_map = {"5m": "5m", "15m": "15m", "30m": "30m", "1H": "1H", "4h": "4H", "1h": "1H"}
    tf = tf_map.get(timeframe, timeframe)
    
    all_rows = []
    after = None
    remaining = limit
    
    while remaining > 0:
        batch = min(remaining, 300)
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": tf, "limit": str(batch)}
        if after:
            params["after"] = str(after)
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if data.get("code") != "0" or not data.get("data"):
                break
            
            candles = data["data"]
            if not candles:
                break
            
            for c in candles:
                all_rows.append({
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })
            
            after = candles[-1][0]  # oldest timestamp for next page
            remaining -= len(candles)
            
            if len(candles) < batch:
                break
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print("Error fetching %s: %s" % (symbol, e))
            break
    
    if not all_rows:
        return None
    
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df
