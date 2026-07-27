"""
HQIP CLI Runner
===============
python -m hqip.run [--symbol BTCUSDT] [--capital 10000] [--leverage 5]
"""
import sys, json, argparse
from hqip.orchestrator import Orchestrator
from hqip.config import SYMBOLS

def main():
    parser = argparse.ArgumentParser(description="HQIP - Crypto Trading Intelligence")
    parser.add_argument("--symbol", default=None, help="Scan single symbol")
    parser.add_argument("--capital", type=float, default=10000, help="Account capital in USD")
    parser.add_argument("--max-loss", type=float, default=100, help="Max dollar loss per trade")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage multiplier")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print("🧠 HQIP — Hermes Quant Intelligence Platform")
    print(f"   Capital: ${args.capital:,.0f} | Leverage: {args.leverage}x | Max Loss: ${args.max_loss:.0f}")
    print()

    orch = Orchestrator(capital=args.capital, max_loss=args.max_loss, leverage=args.leverage)

    if args.symbol:
        result = orch.scan_symbol(args.symbol)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(orch.format_signal(result))
    else:
        results = orch.scan_all()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for r in results:
                print(orch.format_signal(r))
                print()
            print(orch.format_summary())

if __name__ == "__main__":
    main()
