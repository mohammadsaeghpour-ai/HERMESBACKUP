"""HQIP Data Platform - Fetch OHLCV from Binance via ccxt."""
import ccxt, pandas as pd, time, hashlib, os

CACHE_DIR = "/tmp/hqip_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class DataPlatform:
    def __init__(self):
        self.exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        self._cache = {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        key = f"{symbol}_{timeframe}_{limit}"
        cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < 300:
            return pd.read_pickle(cache_file)
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            print(f"[DataPlatform] Error fetching {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def fetch_all_timeframes(self, symbol: str, timeframes: list, limit: int = 300) -> dict:
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit)
            time.sleep(0.1)
        return result

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            print(f"[DataPlatform] Error fetching order book {symbol}: {e}")
            return {"bids": [], "asks": []}

    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> list:
        try:
            return self.exchange.fetch_trades(symbol, limit=limit)
        except Exception as e:
            print(f"[DataPlatform] Error fetching trades {symbol}: {e}")
            return []
