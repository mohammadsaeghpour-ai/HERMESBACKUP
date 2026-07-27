"""
HQIP CLI Runner
===============
Entry point for the HQIP scanning pipeline.

Usage
-----
    python -m hqip.run                     # scan all symbols (default)
    python -m hqip.run --symbol BTCUSDT    # single symbol
    python -m hqip.run --scan-all          # explicit scan-all (same as default)
    python -m hqip.run --json              # JSON output for programmatic use
    python -m hqip.run --symbol ETHUSDT --json
"""

import argparse
import logging
import sys
import time

from hqip.config import SYMBOLS, TIMEFRAMES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hqip.run",
        description="HQIP — Hermes Quant Intelligence Platform  •  Crypto Trading Brain",
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default=None,
        help="Scan a single symbol (e.g. BTCUSDT). Default: scan all configured symbols.",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        default=False,
        help="Scan all configured symbols (this is the default).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        default=None,
        nargs="+",
        help="Override timeframes (e.g. --timeframe 4h 1h). Default: all configured.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Override trading capital (default from config).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress output — only print final results.",
    )
    return parser


def _print_banner():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  HQIP — Quant Intelligence Platform  v1.0   ║")
    print("  ║  Multi-Agent Crypto Trading Brain            ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()


def _print_progress(symbol: str, timeframe: str, idx: int, total: int):
    pct = (idx / total) * 100 if total else 0
    print(f"  ⏳ [{idx}/{total}] {symbol} {timeframe} …", end="", flush=True)


def _print_done(symbol: str, timeframe: str, direction: str, grade: str, confidence: float):
    dir_icons = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}
    icon = dir_icons.get(direction, "❓")
    conf_str = f"{confidence:.0%}"
    print(f"  ✓  {icon} {direction:<8} {grade:<8} {conf_str:<6}")


def _print_error(symbol: str, timeframe: str, msg: str):
    print(f"  ✗  {symbol} {timeframe}: {msg}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # ── Logging setup ───────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Import orchestrator (triggers lazy deps) ────────
    from hqip.orchestrator import Orchestrator
    from hqip.display import (
        format_signal_console,
        format_signal_telegram,
        format_summary_table_console,
        format_summary_table_telegram,
        format_signals_json,
    )

    # ── Build orchestrator ──────────────────────────────
    timeframes = args.timeframe if args.timeframe else None
    orchestrator = Orchestrator(
        timeframes=timeframes,
        capital=args.capital,
    )

    # ── Run scan ────────────────────────────────────────
    if not args.quiet:
        _print_banner()

    t0 = time.time()
    all_signals = []

    if args.symbol:
        # Single symbol
        symbol = args.symbol.upper()
        tf_list = orchestrator.timeframes
        total_scans = len(tf_list)
        scan_idx = 0

        if not args.quiet:
            print(f"  🔍 Scanning {symbol} across {total_scans} timeframe(s)…\n")

        for tf in tf_list:
            scan_idx += 1
            if not args.quiet:
                _print_progress(symbol, tf, scan_idx, total_scans)

            try:
                signal = orchestrator.scan_single(symbol, tf)
                all_signals.append(signal)

                if not args.quiet:
                    print()  # newline after progress
                    _print_done(symbol, tf, signal.direction, signal.grade, signal.confidence)
            except Exception as e:
                all_signals.append(
                    orchestrator._error_signal(symbol, tf, str(e))
                )
                if not args.quiet:
                    print()
                    _print_error(symbol, tf, str(e))
    else:
        # Scan all symbols
        total_symbols = len(orchestrator.symbols)
        total_tfs = len(orchestrator.timeframes)
        total_scans = total_symbols * total_tfs
        scan_idx = 0

        if not args.quiet:
            print(
                f"  🔍 Scanning {total_symbols} symbols × {total_tfs} timeframes "
                f"= {total_scans} analyses\n"
            )

        for sym_idx, symbol in enumerate(orchestrator.symbols, 1):
            if not args.quiet:
                print(f"  ── [{sym_idx}/{total_symbols}] {symbol} {'─' * 30}")

            try:
                symbol_signals = orchestrator.scan_symbol(symbol)
                all_signals.extend(symbol_signals)

                for sig in symbol_signals:
                    scan_idx += 1
                    if not args.quiet:
                        _print_done(
                            sig.symbol, sig.timeframe,
                            sig.direction, sig.grade, sig.confidence,
                        )
            except Exception as e:
                scan_idx += total_tfs
                if not args.quiet:
                    _print_error(symbol, "*", str(e))

            if not args.quiet:
                print()

    elapsed = time.time() - t0

    # ── Output ──────────────────────────────────────────
    if args.json:
        print(format_signals_json(all_signals))
    else:
        # Detailed signals
        print()
        print(f"{'═' * 56}")
        print(f"  📊 DETAILED SIGNALS")
        print(f"{'═' * 56}")
        for sig in all_signals:
            print(format_signal_console(sig))
            print()

        # Summary table
        print()
        print(format_summary_table_console(all_signals))

        # Telegram version (copy-pasteable)
        print()
        print("  📱 Telegram-ready output:")
        print("  " + "─" * 40)
        print(format_summary_table_telegram(all_signals))
        print("  " + "─" * 40)

        # Timing
        print()
        print(f"  ⏱  Completed in {elapsed:.1f}s")

    # Exit code
    actionable = sum(1 for s in all_signals if s.direction != "NO_TRADE")
    if actionable == 0:
        sys.exit(1)  # no actionable signals → non-zero exit
    sys.exit(0)


if __name__ == "__main__":
    main()
