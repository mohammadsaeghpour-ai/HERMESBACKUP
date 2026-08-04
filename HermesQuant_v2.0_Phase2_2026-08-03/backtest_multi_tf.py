
"""Test on multiple timeframes"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")
sys.path.insert(0, "/data/workspace")

# Import backtest function
exec(open("/data/workspace/HermesQuant/backtest_v2.py").read().split("if __name__")[0])

print("\n" + "#"*70)
print("#  MULTI-TIMEFRAME TEST")
print("#"*70)

for tf in ["15m", "30m", "1H"]:
    print("\n\n" + "="*70)
    print("  TIMEFRAME: %s" % tf)
    print("="*70)
    
    r_eth = backtest_v2("ETH-USDT-SWAP", tf, vote_thresh=0.72)
    print()
    r_btc = backtest_v2("BTC-USDT-SWAP", tf, vote_thresh=0.77)
    
    print("\n  SUMMARY %s: ETH=%.1f%% BTC=%.1f%%" % (
        tf, r_eth.get("accuracy", 0), r_btc.get("accuracy", 0)))
