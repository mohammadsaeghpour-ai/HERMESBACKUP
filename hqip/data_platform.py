"""HQIP Data Platform - Fetch OHLCV from multiple exchanges via ccxt."""
import ccxt, pandas as pd, time, hashlib, os

CACHE_DIR = "/tmp/hqip_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Symbol mapping per exchange
SYMBOL_MAP = {
    "okx": {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT",
            "BNBUSDT": "BNB/USDT", "XRPUSDT": "XRP/USDT"},
    "bybit": {"BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT", "SOLUSDT": "SOLUSDT",
              "BNBUSDT": "BNBUSDT", "XRPUSDT": "XRPUSDT"},
}

class DataPlatform:
    def __init__(self):
        self.exchange = self._init_exchange()
        self.exchange_id = self.exchange.id

    def _init_exchange(self):
        for ex_id in ["bybit", "okx", "gate", "kucoin"]:
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
                ex.load_markets()
                print(f"[DataPlatform] Using {ex_id}")
                return ex
            except Exception as e:
                print(f"[DataPlatform] {ex_id}: {str(e)[:50]}")
        return ccxt.binance({"enableRateLimit": True})

    def _convert_symbol(self, symbol):
        mapping = SYMBOL_MAP.get(self.exchange_id, {})
        if symbol in mapping:
            return mapping[symbol]
        # Generic conversion: BTCUSDT -> BTC/USDT
        if "/" not in symbol and len(symbol) > 4:
            for quote in ["USDT", "USDC", "BUSD"]:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    return f"{base}/{quote}"
        return symbol

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        ccxt_symbol = self._convert_symbol(symbol)
        key = f"{self.exchange_id}_{ccxt_symbol}_{timeframe}_{limit}"
        cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.pkl")
        if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < 600:
            return pd.read_pickle(cache_file)
        try:
            raw = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            print(f"[DataPlatform] Error {ccxt_symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def fetch_all_timeframes(self, symbol: str, timeframes: list, limit: int = 300) -> dict:
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit)
            time.sleep(0.3)
        return result

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        ccxt_symbol = self._convert_symbol(symbol)
        try:
            return self.exchange.fetch_order_book(ccxt_symbol, limit=limit)
        except Exception as e:
            return {"bids": [], "asks": []}

    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> list:
        ccxt_symbol = self._convert_symbol(symbol)
        try:
            return self.exchange.fetch_trades(ccxt_symbol, limit=limit)
        except:
            return []
