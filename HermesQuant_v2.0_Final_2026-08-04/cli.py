"""HermesQuant CLI"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import argparse
from orchestrator import MasterOrchestrator

def main():
    parser = argparse.ArgumentParser(description="HermesQuant v2.0 Trading Signal")
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--capital", type=float, default=10)
    parser.add_argument("--leverage", type=int, default=20)
    args = parser.parse_args()
    
    orch = MasterOrchestrator(capital=args.capital, leverage=args.leverage)
    signal = orch.run(args.symbol, args.timeframe)
    
    if signal.direction == "WAIT":
        print("\n  NO SIGNAL — WAIT")
    else:
        print("\n  === TRADE ===")
        print("  %s %s @ $%.2f" % (signal.direction, args.symbol, signal.entry))
        print("  SL: $%.2f | TP1: $%.2f | TP2: $%.2f | TP3: $%.2f" % (signal.sl, signal.tp1, signal.tp2, signal.tp3))

if __name__ == "__main__":
    main()
