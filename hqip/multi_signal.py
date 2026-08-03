#!/usr/bin/env python3
"""
HQIP Multi-Signal Scanner — Per-Timeframe Signals
Each timeframe gets its OWN signal with optimized strategy.
"""
import sys
sys.path.insert(0, "/data/workspace")

from hqip.strategies import get_all_strategies
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators

# ── Config ─────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
CAPITAL = 10.0
LEVERAGE = 20
MAX_LOSS = 1.0  # $1 max loss per trade

def scan_symbol(symbol):
    """Scan one symbol with all timeframe strategies."""
    dp = DataPlatform()
    
    results = {}
    strategies = get_all_strategies()
    
    for tf, strategy in strategies.items():
        try:
            df = dp.fetch_ohlcv(symbol, tf, limit=300)
            if df is None or df.empty or len(df) < 30:
                results[tf] = {"direction": "NO_TRADE", "confidence": 0, "evidence": ["No data"]}
                continue
            
            df = calculate_all_indicators(df)
            result = strategy.analyze(df, symbol)
            results[tf] = result
        except Exception as e:
            results[tf] = {"direction": "NO_TRADE", "confidence": 0, "evidence": [str(e)[:50]]}
    
    return results


def format_signal(tf, result):
    """Format a single timeframe signal."""
    d = result.get("direction", "NO_TRADE")
    conf = result.get("confidence", 0)
    entry = result.get("entry", 0)
    sl = result.get("sl", 0)
    tp1 = result.get("tp1", 0)
    tp2 = result.get("tp2", 0)
    tp3 = result.get("tp3", 0)
    strategy = result.get("strategy", "")
    
    if d == "NO_TRADE":
        return f"  ⏸️ [{tf}] — NO TRADE ({conf:.0f}%)"
    
    emoji = "🟢" if d == "BUY" else "🔴"
    
    # Calculate potential $ profit/loss
    sl_pct = result.get("sl_pct", 0.5)
    tp_pct = result.get("tp_pct", 1.0)
    
    lines = []
    lines.append(f"  {emoji} [{tf}] {d} — {strategy} ({conf:.0f}%)")
    lines.append(f"     📍 Entry: ${entry:,.2f}")
    lines.append(f"     🛑 SL: ${sl:,.2f} ({sl_pct:.2f}%)")
    lines.append(f"     🎯 TP1: ${tp1:,.2f} ({tp_pct:.2f}%)")
    lines.append(f"     🎯 TP2: ${tp2:,.2f}")
    lines.append(f"     🎯 TP3: ${tp3:,.2f}")
    
    # Evidence
    evidence = result.get("evidence", [])
    for e in evidence[:3]:
        lines.append(f"     {e}")
    
    return "\n".join(lines)


def scan_all():
    """Scan all symbols with multi-timeframe strategies."""
    lines = []
    lines.append("=" * 50)
    lines.append("  🎯 HQIP — Multi-Signal Scanner")
    lines.append(f"  💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ ${MAX_LOSS} max")
    lines.append("=" * 50)
    
    for symbol in SYMBOLS:
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  🔍 {symbol}")
        lines.append(f"{'─' * 50}")
        
        results = scan_symbol(symbol)
        
        # Count signals
        buy_count = sum(1 for r in results.values() if r.get("direction") == "BUY")
        sell_count = sum(1 for r in results.values() if r.get("direction") == "SELL")
        
        lines.append(f"  📊 Signals: {buy_count} BUY | {sell_count} SELL")
        
        for tf in ["15m", "1h", "4h", "1d"]:
            if tf in results:
                lines.append(format_signal(tf, results[tf]))
                lines.append("")
    
    lines.append("=" * 50)
    lines.append("  ⚠️ فقط توصیه — نه ربات خودکار")
    lines.append("=" * 50)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(scan_all())
