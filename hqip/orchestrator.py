"""
HQIP Orchestrator
=================
Main entry point. Coordinates the Manager Agent.
"""
import json
from hqip.agents.manager_agent import ManagerAgent
from hqip.config import SYMBOLS

class Orchestrator:
    def __init__(self, capital=10000, max_loss=100, leverage=5):
        self.manager = ManagerAgent()
        self.capital = capital
        self.max_loss = max_loss
        self.leverage = leverage
        self.results = []

    def scan_all(self):
        self.results = []
        for symbol in SYMBOLS:
            try:
                r = self.manager.scan(symbol, self.capital, self.max_loss, self.leverage)
                self.results.append(r)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                self.results.append({"symbol": symbol, "direction": "ERROR", "error": str(e)})
        return self.results

    def scan_symbol(self, symbol):
        return self.manager.scan(symbol, self.capital, self.max_loss, self.leverage)

    def format_signal(self, r):
        if r.get("direction") == "ERROR":
            return f"❌ {r['symbol']}: Error - {r.get('error', 'unknown')}"

        d = r["direction"]
        emoji = "🟢" if d == "BUY" else "🔴" if d == "SELL" else "⚪"
        grade = r.get("grade", "?")
        conf = r.get("confidence", 0)

        lines = [
            f"{emoji} *{r['symbol']}* — {d} (Grade: {grade})",
            f"Confidence: {conf:.0f}%",
        ]

        if d != "NO_TRADE" and r.get("entry"):
            lines.extend([
                f"Entry: {r['entry']:.2f}",
                f"Stop Loss: {r['sl']:.2f}",
                f"TP1: {r['tp1']:.2f} (RR {r.get('risk_reward', 0):.1f})",
                f"TP2: {r['tp2']:.2f}",
                f"TP3: {r['tp3']:.2f}",
                f"Position: {r.get('position_size', 0):.6f} (${r.get('position_value', 0):.0f})",
                f"Leverage: {r.get('leverage', 5)}x",
            ])

        # Top evidence
        if r.get("explanation"):
            lines.append("")
            lines.append("*Reasons:*")
            for line in r["explanation"][:10]:
                lines.append(f"  {line}")

        return "\n".join(lines)

    def format_summary(self):
        lines = ["="*50, "  HQIP SCAN SUMMARY", "="*50, ""]
        for r in self.results:
            d = r.get("direction", "?")
            emoji = "🟢" if d == "BUY" else "🔴" if d == "SELL" else "⚪"
            lines.append(f"{emoji} {r['symbol']}: {d} | Grade: {r.get('grade', '?')} | Conf: {r.get('confidence', 0):.0f}%")
        lines.append("")
        actionable = [r for r in self.results if r.get("direction") in ("BUY", "SELL")]
        lines.append(f"Actionable signals: {len(actionable)}/{len(self.results)}")
        return "\n".join(lines)
