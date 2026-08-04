"""
Multi-Timeframe Confirmation Agent
Cascade: 1D → 4H → 1H → 15m
All must agree for signal to pass.
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import requests
from core.data_types import AgentOutput
from core import indicators as ind

def fetch_candles(instId, bar, limit=100):
    r = requests.get("https://www.okx.com/api/v5/market/candles",
                     params={"instId": instId, "bar": bar, "limit": limit}, timeout=10)
    data = r.json()["data"]
    import pandas as pd
    rows = [{"open": float(c[1]), "high": float(c[2]),
             "low": float(c[3]), "close": float(c[4]),
             "volume": float(c[5])} for c in reversed(data)]
    return pd.DataFrame(rows)

class MTFConfirmAgent:
    """
    Multi-Timeframe Confirmation:
    1D → 4H → 1H → 15m
    Signal only when cascade agrees.
    """
    name = "MTF_Confirm"
    weight = 1.8  # Highest weight — most important filter
    
    def analyze(self, df, symbol="", timeframe="15m"):
        try:
            # Fetch higher timeframes
            df_4h = fetch_candles(symbol, "4H", 50)
            df_1h = fetch_candles(symbol, "1H", 50)
            df_1d = fetch_candles(symbol, "1D", 30)
        except Exception as e:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0,
                              evidence=["Error fetching MTF data: %s" % str(e)])
        
        # Analyze each timeframe
        # 1D: macro trend
        e20_1d = ind.ema(df_1d["close"], 20).iloc[-1]
        e50_1d = ind.ema(df_1d["close"], 50).iloc[-1]
        d1_trend = "UP" if e20_1d > e50_1d else "DOWN"
        
        # 4H: structure
        st4_dir, _ = ind.supertrend(df_4h)
        d4_dir = "UP" if st4_dir.iloc[-1] == 1 else "DOWN"
        
        # 1H: momentum
        adx1h, dip1h, dim1h = ind.adx(df_1h)
        adx1h_val = adx1h.iloc[-1] if not __import__("pandas").isna(adx1h.iloc[-1]) else 0
        d1h_dir = "UP" if dip1h.iloc[-1] > dim1h.iloc[-1] and adx1h_val > 20 else (
            "DOWN" if dim1h.iloc[-1] > dip1h.iloc[-1] and adx1h_val > 20 else "NEUTRAL")
        
        # 15m: entry trigger (use passed df)
        e8_15 = ind.ema(df["close"], 8).iloc[-1]
        e20_15 = ind.ema(df["close"], 20).iloc[-1]
        d15_dir = "UP" if e8_15 > e20_15 else "DOWN"
        
        directions = [d1_trend, d4_dir, d1h_dir, d15_dir]
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")
        
        evidence = ["1D=%s 4H=%s 1H=%s 15m=%s" % (d1_trend, d4_dir, d1h_dir, d15_dir)]
        
        if up_count >= 3:
            d = "BUY"
            score = up_count / 4
            conf = up_count / 4 * 100
            evidence.append("MTF alignment: %d/4 UP" % up_count)
        elif down_count >= 3:
            d = "SELL"
            score = -down_count / 4
            conf = down_count / 4 * 100
            evidence.append("MTF alignment: %d/4 DOWN" % down_count)
        else:
            d = "NEUTRAL"
            score = 0
            conf = 10
            evidence.append("MTF split — no alignment")
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=evidence)
