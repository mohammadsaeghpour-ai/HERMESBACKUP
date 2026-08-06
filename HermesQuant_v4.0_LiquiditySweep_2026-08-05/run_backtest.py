#!/usr/bin/env python3
"""Run Liquidity Sweep backtest on BTC and ETH"""
import sys, os
sys.path.insert(0, '/data/workspace/HermesQuant_v4')
import warnings; warnings.filterwarnings("ignore")

import requests, time, numpy as np, pandas as pd
from liquidity_sweep import LiquidityDetector, LiquiditySweepDetector, LiquiditySweepBacktest

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

print("="*60)
print("  LIQUIDITY SWEEP ENGINE — BACKTEST")
print("="*60)

for sym in ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]:
    for tf in ["1h", "15m"]:
        print("\n--- %s %s ---" % (sym.split("-")[0], tf.upper()))
        
        # Fetch data
        limit = 1440 if tf == "1h" else 2880
        df = fetch(sym, tf, limit)
        
        if df is None or len(df) < 200:
            print("Not enough data: %d candles" % (len(df) if df is not None else 0))
            continue
        
        days = len(df) * (60 if tf == "1h" else 15) / 1440
        print("Data: %d candles (%.0f days)" % (len(df), days))
        
        # Split: 70% train, 30% test
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        
        print("Train: %d candles | Test: %d candles" % (len(train_df), len(test_df)))
        
        # Build liquidity map on TRAIN
        detector = LiquidityDetector(cluster_pct=0.002, min_strength=2)
        liq_map = detector.build_liquidity_map(train_df)
        
        print("\nLiquidity Map:")
        print("  Resistance levels: %d" % len(liq_map["resistance"]))
        print("  Support levels: %d" % len(liq_map["support"]))
        print("  Equal Highs: %d" % len(liq_map["equal_highs"]))
        print("  Equal Lows: %d" % len(liq_map["equal_lows"]))
        print("  PDH: %s | PDL: %s" % (
            "$%.2f" % liq_map["pdh"] if liq_map["pdh"] else "N/A",
            "$%.2f" % liq_map["pdl"] if liq_map["pdl"] else "N/A"))
        
        # Top 5 levels
        all_levels = []
        for h in liq_map["resistance"][:5]:
            all_levels.append(("R", h["price"], h["strength"]))
        for l in liq_map["support"][:5]:
            all_levels.append(("S", l["price"], l["strength"]))
        all_levels.sort(key=lambda x: x[2], reverse=True)
        
        print("\n  Top Levels:")
        for t, p, s in all_levels[:5]:
            print("    %s $%.2f (strength=%d)" % (t, p, s))
        
        # Detect sweeps on TEST
        sweep_detector = LiquiditySweepDetector(
            lookback_fresh=10, confirm_candles=2,
            wick_ratio=1.0, stop_buffer=0.002
        )
        
        signals = sweep_detector.detect_sweeps(test_df, liq_map)
        print("\nSweeps detected: %d" % len(signals))
        
        if signals:
            # Backtest
            bt = LiquiditySweepBacktest(
                capital=10.0, risk_per_trade=0.01,
                max_trades=50, min_candle_gap=5, time_exit=50
            )
            
            result = bt.run(test_df, signals)
            
            if result["total"] > 0:
                print()
                print("=== BACKTEST RESULTS ===")
                print("Total Trades: %d" % result["total"])
                print("Winners: %d | Losers: %d" % (result["winners"], result["losers"]))
                print()
                print("Accuracy: %.1f%%" % result["accuracy"])
                print("Profit Factor: %.2f" % result["profit_factor"])
                print("Avg R: %.2f" % result["avg_r"])
                print("Avg Win: +$%.4f | Avg Loss: -$%.4f" % (result["avg_win"], abs(result["avg_loss"])))
                print()
                print("Total P&L: +$%.4f" % result["total_pnl"])
                print("Total Return: +%.1f%%" % result["total_return"])
                print("Max Drawdown: %.1f%%" % result["max_drawdown"])
                print("Final Equity: $%.2f" % result["final_equity"])
                print()
                print("Long Trades: %d (WR: %.1f%%)" % (result["long_trades"], result["long_wr"]))
                print("Short Trades: %d (WR: %.1f%%)" % (result["short_trades"], result["short_wr"]))
                
                # Last 5 trades
                print()
                print("=== LAST 5 TRADES ===")
                for trade in result["trades"][-5:]:
                    print("  %s @ $%.2f -> $%.2f | %+.2fR | %s" % (
                        trade["type"], trade["entry"], trade["exit"],
                        trade["risk_reward"], trade["exit_reason"]))
            else:
                print("No trades executed")
        else:
            print("No sweep signals found")

print()
print("="*60)
print("  BACKTEST COMPLETE")
print("="*60)
