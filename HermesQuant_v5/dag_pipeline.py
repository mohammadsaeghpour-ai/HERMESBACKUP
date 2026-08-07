"""
HermesQuant v5.0 — REAL DAG Pipeline
Agents NOT independent. Each stage feeds the next.
Stage 0 (Independent) → Stage 1 (Meta) → Stage 2 (Structure) → Stage 3 (Decision) → Risk
"""
import requests, numpy as np, pandas as pd, time, datetime
import warnings; warnings.filterwarnings("ignore")


def fetch(symbol, tf, limit):
    rows = []; after = None; remaining = limit
    while remaining > 0:
        batch = min(remaining, 300)
        params = {"instId": symbol, "bar": tf, "limit": str(batch)}
        if after: params["after"] = str(after)
        try:
            r = requests.get("https://www.okx.com/api/v5/market/candles", params=params, timeout=10).json()
            if r.get("code") != "0" or not r.get("data"): break
            for c in r["data"]:
                rows.append({"ts": int(c[0]), "o": float(c[1]), "h": float(c[2]),
                             "l": float(c[3]), "c": float(c[4]), "v": float(c[5])})
            after = r["data"][-1][0]; remaining -= len(r["data"])
            if len(r["data"]) < batch: break
            time.sleep(0.1)
        except: break
    if not rows: return None
    df = pd.DataFrame(rows); df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def ema(s, p): return s.ewm(span=p).mean()
def rsi(s, p=14):
    d = s.diff(); g = d.where(d > 0, 0).rolling(p).mean(); l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)
def macd_calc(s, f=12, sl=26, sig=9):
    ef = ema(s, f); es = ema(s, sl); ml = ef - es; sl2 = ema(ml, sig); return ml, sl2, ml - sl2
def atr(df, p=14):
    tr = pd.concat([df["h"]-df["l"], (df["h"]-df["c"].shift()).abs(), (df["l"]-df["c"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def adx_calc(df, p=14):
    pdm = df["h"].diff(); mdm = -df["l"].diff()
    pdm[pdm < 0] = 0; mdm[mdm < 0] = 0
    tr = pd.concat([df["h"]-df["l"], (df["h"]-df["c"].shift()).abs(), (df["l"]-df["c"].shift()).abs()], axis=1).max(axis=1)
    a = tr.rolling(p).mean(); pdi = 100*pdm.rolling(p).mean()/a; mdi = 100*mdm.rolling(p).mean()/a
    dx = (pdi-mdi).abs()/(pdi+mdi)*100; adx_v = dx.rolling(p).mean()
    return adx_v, pdi, mdi


class DAGPipeline:
    """
    Real DAG: each stage depends on previous stages.
    Stage 0: Independent indicators (raw data)
    Stage 1: Meta-agents (depend on Stage 0)
    Stage 2: Structure agents (depend on Stage 0 + 1)
    Stage 3: Decision agents (depend on Stage 0 + 1 + 2)
    Stage 4: Risk gate (depends on all)
    """
    
    def __init__(self, df5, df15, df1h, df4h, fg_val=50):
        self.df5 = df5
        self.df15 = df15
        self.df1h = df1h
        self.df4h = df4h
        self.fg_val = fg_val
        
        self.stage0 = {}
        self.stage1 = {}
        self.stage2 = {}
        self.stage3 = {}
        self.stage4 = {}
    
    # ═══════════════════════════════════════
    # STAGE 0: Independent Indicators
    # ═══════════════════════════════════════
    def stage_0_independent(self):
        """Raw indicator calculation — no dependencies"""
        for name, df in [("4H", self.df4h), ("1H", self.df1h), ("15m", self.df15), ("5m", self.df5)]:
            if df is None or len(df) < 50:
                continue
            e20 = ema(df["c"], 20).iloc[-1]
            e50 = ema(df["c"], 50).iloc[-1]
            r = rsi(df["c"]).iloc[-1]
            ml, sl2, hist = macd_calc(df["c"])
            a = atr(df).iloc[-1]
            adx_v, pdi, mdi = adx_calc(df)
            
            self.stage0[name] = {
                "trend": "UP" if e20 > e50 else "DOWN",
                "rsi": r,
                "macd": "BULL" if hist.iloc[-1] > 0 else "BEAR",
                "macd_cross": "UP" if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0 else
                              "DOWN" if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0 else "NONE",
                "atr": a,
                "adx": adx_v.iloc[-1],
                "pdi": pdi.iloc[-1],
                "mdi": mdi.iloc[-1],
                "ema20": e20,
                "ema50": e50,
            }
        
        # Volume (5m)
        vol = self.df5["v"].iloc[-1]
        vol_avg = self.df5["v"].rolling(20).mean().iloc[-1]
        self.stage0["volume"] = vol / vol_avg if vol_avg > 0 else 1
        
        # Candles
        last5 = self.df5.tail(5)
        self.stage0["bull_candles"] = sum(1 for i in range(5) if last5["c"].iloc[i] > last5["o"].iloc[i])
        
        # Fear/Greed
        self.stage0["fg"] = self.fg_val
    
    # ═══════════════════════════════════════
    # STAGE 1: Meta-Agents (depend on Stage 0)
    # ═══════════════════════════════════════
    def stage_1_meta(self):
        """Regime detection — depends on Stage 0 indicators"""
        
        # REGIME AGENT: determines market state
        adx_1h = self.stage0.get("1H", {}).get("adx", 20)
        rsi_1h = self.stage0.get("1H", {}).get("rsi", 50)
        vol_ratio = self.stage0.get("volume", 1)
        
        if adx_1h > 25 and vol_ratio > 1.0:
            regime = "TRENDING"
        elif adx_1h < 20 and vol_ratio < 0.8:
            regime = "RANGING"
        else:
            regime = "TRANSITION"
        
        self.stage1["regime"] = regime
        
        # STRUCTURE AGENT: HH/HL or LH/LL (depends on regime)
        if regime == "TRENDING":
            # In trending, check if price making HH/HL or LH/LL
            highs_1h = self.df1h["h"].tail(20)
            lows_1h = self.df1h["l"].tail(20)
            hh = highs_1h.iloc[-1] > highs_1h.iloc[-10:].max() * 0.999
            hl = lows_1h.iloc[-1] > lows_1h.iloc[-10:].min() * 1.001
            
            if hh and hl:
                structure = "HH_HL"
                self.stage1["structure"] = "BULLISH"
            elif not hh and not hl:
                structure = "LH_LL"
                self.stage1["structure"] = "BEARISH"
            else:
                structure = "MIXED"
                self.stage1["structure"] = "NEUTRAL"
        else:
            self.stage1["structure"] = "NEUTRAL"
        
        # VOLUME PROFILE: accumulation vs distribution (depends on trend + volume)
        trend_1h = self.stage0.get("1H", {}).get("trend", "NEUTRAL")
        
        if trend_1h == "UP" and vol_ratio > 1.2:
            self.stage1["volume_profile"] = "ACCUMULATION"
        elif trend_1h == "DOWN" and vol_ratio > 1.2:
            self.stage1["volume_profile"] = "DISTRIBUTION"
        elif vol_ratio < 0.5:
            self.stage1["volume_profile"] = "NO_INTEREST"
        else:
            self.stage1["volume_profile"] = "NEUTRAL"
        
        # FEAR/GREED AGENT: (depends on RSI + volume)
        rsi_15 = self.stage0.get("15m", {}).get("rsi", 50)
        
        if self.fg_val < 25 and rsi_15 < 30:
            self.stage1["sentiment"] = "EXTREME_FEAR_OVERSOLD"
        elif self.fg_val > 75 and rsi_15 > 70:
            self.stage1["sentiment"] = "EXTREME_GREED_OVERBOUGHT"
        elif self.fg_val < 35:
            self.stage1["sentiment"] = "FEAR"
        elif self.fg_val > 65:
            self.stage1["sentiment"] = "GREED"
        else:
            self.stage1["sentiment"] = "NEUTRAL"
    
    # ═══════════════════════════════════════
    # STAGE 2: Structure Agents (depend on 0 + 1)
    # ═══════════════════════════════════════
    def stage_2_structure(self):
        """Depends on Stage 0 (indicators) + Stage 1 (regime)"""
        
        regime = self.stage1.get("regime", "RANGING")
        trend_1h = self.stage0.get("1H", {}).get("trend", "NEUTRAL")
        macd_1h = self.stage0.get("1H", {}).get("macd", "NEUTRAL")
        rsi_1h = self.stage0.get("1H", {}).get("rsi", 50)
        rsi_15 = self.stage0.get("15m", {}).get("rsi", 50)
        adx_1h = self.stage0.get("1H", {}).get("adx", 20)
        vol_ratio = self.stage0.get("volume", 1)
        
        # S/R AGENT: uses regime to weight S/R importance
        if regime == "RANGING":
            # In ranging, S/R is king
            self.stage2["sr_weight"] = 1.5  # High importance
        elif regime == "TRENDING":
            # In trending, S/R less important, trend is king
            self.stage2["sr_weight"] = 0.7  # Low importance
        else:
            self.stage2["sr_weight"] = 1.0
        
        # DIVERGENCE AGENT: MACD + RSI divergence (depends on trend + MACD)
        ml_15, sl_15, hist_15 = macd_calc(self.df15["c"]) if self.df15 is not None and len(self.df15) > 26 else (None, None, None)
        
        if ml_15 is not None:
            # Price making higher high but MACD making lower high = bearish divergence
            price_higher = self.df15["c"].iloc[-1] > self.df15["c"].iloc[-10]
            macd_higher = hist_15.iloc[-1] > hist_15.iloc[-10]
            
            if price_higher and not macd_higher:
                self.stage2["divergence"] = "BEARISH"
            elif not price_higher and macd_higher:
                self.stage2["divergence"] = "BULLISH"
            else:
                self.stage2["divergence"] = "NONE"
        else:
            self.stage2["divergence"] = "NONE"
        
        # MOMENTUM AGENT: RSI + MACD combined (depends on regime)
        if regime == "TRENDING":
            # In trending, momentum matters more
            if macd_1h == "BULL" and rsi_1h > 50:
                self.stage2["momentum"] = "STRONG_BULL"
            elif macd_1h == "BEAR" and rsi_1h < 50:
                self.stage2["momentum"] = "STRONG_BEAR"
            else:
                self.stage2["momentum"] = "WEAK"
        else:
            # In ranging, momentum less reliable
            self.stage2["momentum"] = "NEUTRAL"
        
        # WHALE AGENT: volume + trend (depends on volume_profile from Stage 1)
        vol_profile = self.stage1.get("volume_profile", "NEUTRAL")
        
        if vol_profile == "ACCUMULATION" and trend_1h == "UP":
            self.stage2["whale"] = "BUYING"
        elif vol_profile == "DISTRIBUTION" and trend_1h == "DOWN":
            self.stage2["whale"] = "SELLING"
        elif vol_profile == "NO_INTEREST":
            self.stage2["whale"] = "ABSENT"
        else:
            self.stage2["whale"] = "NEUTRAL"
    
    # ═══════════════════════════════════════
    # STAGE 3: Decision Agents (depend on 0 + 1 + 2)
    # ═══════════════════════════════════════
    def stage_3_decision(self):
        """Final scoring — depends on ALL previous stages"""
        
        regime = self.stage1.get("regime", "RANGING")
        structure = self.stage1.get("structure", "NEUTRAL")
        sentiment = self.stage1.get("sentiment", "NEUTRAL")
        divergence = self.stage2.get("divergence", "NONE")
        momentum = self.stage2.get("momentum", "NEUTRAL")
        whale = self.stage2.get("whale", "NEUTRAL")
        trend_1h = self.stage0.get("1H", {}).get("trend", "NEUTRAL")
        macd_1h = self.stage0.get("1H", {}).get("macd", "NEUTRAL")
        vol_ratio = self.stage0.get("volume", 1)
        bull_candles = self.stage0.get("bull_candles", 2)
        
        bull = 0; bear = 0
        reasons_bull = []; reasons_bear = []
        
        # Trend (weight: 2)
        if trend_1h == "UP":
            bull += 2; reasons_bull.append("1H trend UP")
        else:
            bear += 2; reasons_bear.append("1H trend DOWN")
        
        # MACD 1H (weight: 2) — CRITICAL
        if macd_1h == "BULL":
            bull += 2; reasons_bull.append("1H MACD BULL")
        else:
            bear += 2; reasons_bear.append("1H MACD BEAR")
        
        # Regime adjustment (key DAG feature)
        if regime == "TRENDING":
            # In trending, boost trend signals
            if trend_1h == "UP": bull += 1; reasons_bull.append("TRENDING boosts trend")
            else: bear += 1; reasons_bear.append("TRENDING boosts trend")
        elif regime == "RANGING":
            # In ranging, reduce trend signals (mean reversion)
            if trend_1h == "UP": bear += 0.5; reasons_bear.append("RANGING weakens trend UP")
            else: bull += 0.5; reasons_bull.append("RANGING weakens trend DOWN")
        
        # Structure (depends on regime)
        if structure == "BULLISH":
            bull += 1; reasons_bull.append("Structure HH+HL")
        elif structure == "BEARISH":
            bear += 1; reasons_bear.append("Structure LH+LL")
        
        # Divergence
        if divergence == "BEARISH":
            bear += 1.5; reasons_bear.append("Bearish divergence")
        elif divergence == "BULLISH":
            bull += 1.5; reasons_bull.append("Bullish divergence")
        
        # Momentum
        if momentum == "STRONG_BULL":
            bull += 1; reasons_bull.append("Strong momentum")
        elif momentum == "STRONG_BEAR":
            bear += 1; reasons_bear.append("Strong momentum")
        
        # Whale
        if whale == "BUYING":
            bull += 1; reasons_bull.append("Whale buying")
        elif whale == "SELLING":
            bear += 1; reasons_bear.append("Whale selling")
        elif whale == "ABSENT":
            # No whale = no trade
            bear += 1; reasons_bear.append("Whale absent")
        
        # Sentiment (contrarian)
        if sentiment == "EXTREME_FEAR_OVERSOLD":
            bull += 1.5; reasons_bull.append("Extreme fear = contrarian buy")
        elif sentiment == "EXTREME_GREED_OVERBOUGHT":
            bear += 1.5; reasons_bear.append("Extreme greed = contrarian sell")
        elif sentiment == "FEAR":
            bull += 0.5; reasons_bull.append("Fear")
        elif sentiment == "GREED":
            bear += 0.5; reasons_bear.append("Greed")
        
        # Candles
        if bull_candles >= 4:
            bull += 0.5; reasons_bull.append("4/5 green")
        elif bull_candles <= 1:
            bear += 0.5; reasons_bear.append("4/5 red")
        
        self.stage3["bull"] = bull
        self.stage3["bear"] = bear
        self.stage3["reasons_bull"] = reasons_bull
        self.stage3["reasons_bear"] = reasons_bear
    
    # ═══════════════════════════════════════
    # STAGE 4: Risk Gate (depends on all)
    # ═══════════════════════════════════════
    def stage_4_risk(self):
        """Final decision with risk management"""
        
        bull = self.stage3["bull"]
        bear = self.stage3["bear"]
        vol_ratio = self.stage0.get("volume", 1)
        regime = self.stage1.get("regime", "RANGING")
        adx_1h = self.stage0.get("1H", {}).get("adx", 20)
        
        # GATE 1: Volume must be > 0.5x
        if vol_ratio < 0.5:
            return "WAIT", "Volume too low (%.1fx)" % vol_ratio, [], []
        
        # GATE 2: Need clear winner (difference > 2)
        if abs(bull - bear) < 2:
            return "WAIT", "Scores too close (%.1f vs %.1f)" % (bull, bear), [], []
        
        # GATE 3: ADX must confirm trend (if trending)
        if regime == "TRENDING" and adx_1h < 20:
            return "WAIT", "Regime says TRENDING but ADX=%.0f" % adx_1h, [], []
        
        # GATE 4: Divergence override
        divergence = self.stage2.get("divergence", "NONE")
        if divergence == "BEARISH" and bull > bear:
            return "WAIT", "Bearish divergence overrides bullish score", [], []
        if divergence == "BULLISH" and bear > bull:
            return "WAIT", "Bullish divergence overrides bearish score", [], []
        
        # All gates passed
        if bear > bull:
            return "SHORT", "Bear=%.1f > Bull=%.1f" % (bear, bull), self.stage3["reasons_bull"], self.stage3["reasons_bear"]
        else:
            return "LONG", "Bull=%.1f > Bear=%.1f" % (bull, bear), self.stage3["reasons_bull"], self.stage3["reasons_bear"]
    
    def run(self):
        """Execute full DAG pipeline"""
        self.stage_0_independent()
        self.stage_1_meta()
        self.stage_2_structure()
        self.stage_3_decision()
        direction, reason, reasons_bull, reasons_bear = self.stage_4_risk()
        
        return {
            "direction": direction,
            "reason": reason,
            "reasons_bull": reasons_bull,
            "reasons_bear": reasons_bear,
            "stage0": self.stage0,
            "stage1": self.stage1,
            "stage2": self.stage2,
            "bull_score": self.stage3["bull"],
            "bear_score": self.stage3["bear"],
        }


def analyze(symbol, fg_val=50):
    """Full analysis for one symbol"""
    coin = symbol.split("-")[0]
    
    df5 = fetch(symbol, "5m", 288)
    df15 = fetch(symbol, "15m", 200)
    df1h = fetch(symbol, "1H", 100)
    df4h = fetch(symbol, "4H", 100)
    
    if df5 is None or len(df5) < 100:
        print("%s: No data" % coin)
        return
    
    price = df5["c"].iloc[-1]
    now_utc = datetime.datetime.utcnow()
    now_tehran = now_utc + datetime.timedelta(hours=3, minutes=30)
    
    # Run DAG
    dag = DAGPipeline(df5, df15, df1h, df4h, fg_val)
    result = dag.run()
    
    # S/R for targets
    sr_list = []
    ds = df15.tail(200)
    for i in range(5, len(ds)-5):
        if ds["h"].iloc[i] >= ds["h"].iloc[i-5:i+6].max(): sr_list.append(("R", ds["h"].iloc[i]))
        if ds["l"].iloc[i] <= ds["l"].iloc[i-5:i+6].min(): sr_list.append(("S", ds["l"].iloc[i]))
    uniq = []
    for t, l in sr_list:
        if not any(abs(l-ul)/ul < 0.002 for _, ul in uniq): uniq.append((t, l))
    uniq.sort(key=lambda x: x[1])
    
    supports = [l for t, l in uniq if t == "S" and l < price]
    resistances = [l for t, l in uniq if t == "R" and l > price]
    ns1 = supports[-1] if supports else price * 0.99
    nr1 = resistances[0] if resistances else price * 1.01
    
    a_1h = atr(df1h).iloc[-1] if df1h is not None and len(df1h) > 14 else price * 0.01
    
    print()
    print("=" * 60)
    print("  %s — DAG PIPELINE SIGNAL" % coin)
    print("  NOW: %02d:%02d Tehran | Price: $%.2f" % (now_tehran.hour, now_tehran.minute, price))
    print("  Validity: 1 HOUR (until %02d:%02d)" % ((now_tehran.hour + 1) % 24, now_tehran.minute))
    print("=" * 60)
    
    # Stage 0
    print("\nSTAGE 0 (Indicators):")
    for tf in ["4H", "1H", "15m", "5m"]:
        if tf in result["stage0"]:
            s = result["stage0"][tf]
            print("  %s: %s RSI=%d MACD=%s ADX=%d" % (tf, s["trend"], s["rsi"], s["macd"], s["adx"]))
    print("  Volume: %.1fx | Candles: %d/5 green | FG=%d" % (
        result["stage0"].get("volume", 0), result["stage0"].get("bull_candles", 0), result["stage0"].get("fg", 50)))
    
    # Stage 1
    print("\nSTAGE 1 (Meta):")
    print("  Regime: %s" % result["stage1"].get("regime", "N/A"))
    print("  Structure: %s" % result["stage1"].get("structure", "N/A"))
    print("  Volume Profile: %s" % result["stage1"].get("volume_profile", "N/A"))
    print("  Sentiment: %s" % result["stage1"].get("sentiment", "N/A"))
    
    # Stage 2
    print("\nSTAGE 2 (Structure):")
    print("  S/R Weight: %.1f" % result["stage2"].get("sr_weight", 1.0))
    print("  Divergence: %s" % result["stage2"].get("divergence", "N/A"))
    print("  Momentum: %s" % result["stage2"].get("momentum", "N/A"))
    print("  Whale: %s" % result["stage2"].get("whale", "N/A"))
    
    # Stage 3
    print("\nSTAGE 3 (Decision):")
    print("  Bull: %.1f → %s" % (result["bull_score"], ", ".join(result["reasons_bull"])))
    print("  Bear: %.1f → %s" % (result["bear_score"], ", ".join(result["reasons_bear"])))
    
    # Stage 4
    print("\nSTAGE 4 (Risk Gate):")
    print("  Direction: %s" % result["direction"])
    print("  Reason: %s" % result["reason"])
    
    # Signal
    print()
    print("-" * 60)
    
    if result["direction"] == "WAIT":
        print("SIGNAL: WAIT")
        print("Reason: %s" % result["reason"])
    elif result["direction"] == "SHORT":
        entry = price
        stop = nr1 + a_1h * 0.3
        tp1 = entry - a_1h * 1.5
        tp2 = entry - a_1h * 2.5
        risk = (stop - entry) / entry * 100
        print("SIGNAL: SHORT (SELL)")
        print("Entry:  $%.2f" % entry)
        print("Stop:   $%.2f (+%.2f%%)" % (stop, risk))
        print("TP1:    $%.2f (-%.2f%%) R:R=1:%.1f" % (tp1, (entry-tp1)/entry*100, (entry-tp1)/(stop-entry)))
        print("TP2:    $%.2f (-%.2f%%)" % (tp2, (entry-tp2)/entry*100))
    else:
        entry = price
        stop = ns1 - a_1h * 0.3
        tp1 = entry + a_1h * 1.5
        tp2 = entry + a_1h * 2.5
        risk = (entry - stop) / entry * 100
        print("SIGNAL: LONG (BUY)")
        print("Entry:  $%.2f" % entry)
        print("Stop:   $%.2f (-%.2f%%)" % (stop, risk))
        print("TP1:    $%.2f (+%.2f%%) R:R=1:%.1f" % (tp1, (tp1-entry)/entry*100, (tp1-entry)/(entry-stop)))
        print("TP2:    $%.2f (+%.2f%%)" % (tp2, (tp2-entry)/entry*100))
    
    print("=" * 60)


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
try:
    fg = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=5).json()
    fg_val = int(fg["data"][0]["value"])
except:
    fg_val = 50

for sym in ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]:
    analyze(sym, fg_val)
