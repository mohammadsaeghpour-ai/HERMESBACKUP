"""
HQIP Display Formatting
=======================
Console (ANSI-colored) and Telegram (Markdown) formatters for signals
and summary tables.  All public functions accept a Signal (or list thereof)
and return a string ready for printing / sending.
"""

import json
from typing import List, Optional

from hqip.orchestrator import Signal

# ── ANSI colours ───────────────────────────────────────────

_C_RESET   = "\033[0m"
_C_BOLD    = "\033[1m"
_C_DIM     = "\033[2m"
_C_RED     = "\033[31m"
_C_GREEN   = "\033[32m"
_C_YELLOW  = "\033[33m"
_C_BLUE    = "\033[34m"
_C_MAGENTA = "\033[35m"
_C_CYAN    = "\033[36m"
_C_WHITE   = "\033[37m"
_C_BG_RED  = "\033[41m"
_C_BG_GRN  = "\033[42m"
_C_BG_YEL  = "\033[43m"

# ── Emoji / glyph maps ────────────────────────────────────

_DIR_EMOJI = {
    "LONG":     "🟢",
    "SHORT":    "🔴",
    "NO_TRADE": "⚪",
}

_GRADE_COLOR = {
    "A+": _C_GREEN + _C_BOLD,
    "A":  _C_GREEN,
    "B+": _C_CYAN,
    "B":  _C_CYAN + _C_DIM,
    "C":  _C_YELLOW,
}

_GRADE_EMOJI = {
    "A+": "🏆",
    "A":  "✅",
    "B+": "👍",
    "B":  "➡️",
    "C":  "⚠️",
    "NO_GRADE": "❌",
}

_REGIME_EMOJI = {
    "trending_up":   "📈",
    "trending_down": "📉",
    "ranging":       "↔️",
    "volatile":      "🌊",
    "low_volatility":"😴",
    "unknown":       "❓",
}


# ── Helpers ────────────────────────────────────────────────

def _fmt_price(price: float) -> str:
    if price == 0:
        return "—"
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


def _grade_display(grade: str) -> str:
    emoji = _GRADE_EMOJI.get(grade, "❓")
    return f"{emoji} {grade}"


def _direction_display(direction: str) -> str:
    emoji = _DIR_EMOJI.get(direction, "❓")
    return f"{emoji} {direction}"


def _regime_display(regime: str) -> str:
    return _REGIME_EMOJI.get(regime, "❓") + " " + regime.replace("_", " ").title()


def _confidence_bar(conf: float, width: int = 10) -> str:
    filled = int(conf * width)
    return "█" * filled + "░" * (width - filled)


# ── Telegram (Markdown) ───────────────────────────────────

def format_signal_telegram(signal: Signal) -> str:
    """
    Format a single signal for Telegram (MarkdownV2-friendly plain text).
    """
    lines: List[str] = []

    # Header
    lines.append(f"{'═' * 36}")
    d_emoji = _DIR_EMOJI.get(signal.direction, "❓")
    g_emoji = _GRADE_EMOJI.get(signal.grade, "❓")
    lines.append(f"{d_emoji}  {signal.symbol}  |  {signal.timeframe}")
    lines.append(f"{'─' * 36}")

    if signal.direction == "NO_TRADE":
        lines.append("")
        lines.append("  ⛔  NO TRADE SIGNAL")
        if signal.contributing_factors:
            lines.append("")
            lines.append("Reason:")
            for f in signal.contributing_factors[:5]:
                lines.append(f"  • {f}")
        lines.append(f"{'═' * 36}")
        return "\n".join(lines)

    # Core metrics
    lines.append("")
    lines.append(f"  Direction : {_direction_display(signal.direction)}")
    lines.append(f"  Grade     : {g_emoji}  {signal.grade}")
    lines.append(f"  Confidence: {signal.confidence:.0%}  {_confidence_bar(signal.confidence)}")
    lines.append(f"  Regime    : {_regime_display(signal.regime_type)}")
    lines.append("")

    # Entry / SL / TP
    lines.append("  ── Levels ──────────────────")
    lines.append(f"  Entry : {_fmt_price(signal.entry_price)}")
    lines.append(f"  SL    : {_fmt_price(signal.stop_loss)}")
    lines.append(f"  TP1   : {_fmt_price(signal.take_profit_1)}")
    lines.append(f"  TP2   : {_fmt_price(signal.take_profit_2)}")
    lines.append(f"  TP3   : {_fmt_price(signal.take_profit_3)}")
    lines.append("")

    # Risk
    lines.append("  ── Risk ────────────────────")
    lines.append(f"  Size   : ${signal.position_size_usd:,.2f}  (×{signal.leverage})")
    lines.append(f"  R:R    : {signal.risk_reward:.2f}")
    lines.append("")

    # Contributing factors (WHY)
    if signal.contributing_factors:
        lines.append("  ── Why ─────────────────────")
        for f in signal.contributing_factors[:8]:
            lines.append(f"  • {f}")

    # Failed agents
    if signal.failed_agents:
        lines.append("")
        lines.append(f"  ⚠️  Agents with errors: {', '.join(signal.failed_agents)}")

    lines.append(f"{'═' * 36}")
    return "\n".join(lines)


# ── Console (ANSI) ────────────────────────────────────────

def format_signal_console(signal: Signal) -> str:
    """Return a colored terminal string for a single signal."""
    lines: List[str] = []

    # Header
    lines.append(f"{_C_BOLD}{'═' * 52}{_C_RESET}")
    d_emoji = _DIR_EMOJI.get(signal.direction, "❓")
    if signal.direction == "LONG":
        d_color = _C_GREEN
    elif signal.direction == "SHORT":
        d_color = _C_RED
    else:
        d_color = _C_DIM

    lines.append(
        f"  {d_emoji}  {_C_BOLD}{signal.symbol}{_C_RESET}"
        f"  {_C_DIM}|{_C_RESET}  {signal.timeframe}"
    )
    lines.append(f"{'─' * 52}")

    if signal.direction == "NO_TRADE":
        lines.append("")
        lines.append(f"  ⛔  {_C_YELLOW}NO TRADE SIGNAL{_C_RESET}")
        if signal.contributing_factors:
            lines.append("")
            lines.append(f"  {_C_DIM}Reason:{_C_RESET}")
            for f in signal.contributing_factors[:5]:
                lines.append(f"    • {f}")
        lines.append(f"{_C_BOLD}{'═' * 52}{_C_RESET}")
        return "\n".join(lines)

    # Metrics
    lines.append("")
    lines.append(
        f"  Direction : {d_color}{signal.direction}{_C_RESET}"
    )
    grade_color = _GRADE_COLOR.get(signal.grade, _C_WHITE)
    lines.append(
        f"  Grade     : {grade_color}{signal.grade}{_C_RESET}"
    )
    conf_color = (
        _C_GREEN if signal.confidence >= 0.7
        else _C_YELLOW if signal.confidence >= 0.5
        else _C_RED
    )
    lines.append(
        f"  Confidence: {conf_color}{signal.confidence:.0%}{_C_RESET}"
        f"  {_C_DIM}{_confidence_bar(signal.confidence)}{_C_RESET}"
    )
    lines.append(
        f"  Regime    : {_regime_display(signal.regime_type)}"
    )
    lines.append("")

    # Levels
    lines.append(f"  {_C_CYAN}── Levels {'─' * 38}{_C_RESET}")
    lines.append(f"  Entry : {_C_BOLD}{_fmt_price(signal.entry_price)}{_C_RESET}")
    lines.append(f"  SL    : {_C_RED}{_fmt_price(signal.stop_loss)}{_C_RESET}")
    lines.append(f"  TP1   : {_C_GREEN}{_fmt_price(signal.take_profit_1)}{_C_RESET}")
    lines.append(f"  TP2   : {_C_GREEN}{_fmt_price(signal.take_profit_2)}{_C_RESET}")
    lines.append(f"  TP3   : {_C_GREEN}{_fmt_price(signal.take_profit_3)}{_C_RESET}")
    lines.append("")

    # Risk
    lines.append(f"  {_C_CYAN}── Risk {'─' * 40}{_C_RESET}")
    lines.append(f"  Size   : ${signal.position_size_usd:,.2f}  (×{signal.leverage})")
    rr_color = _C_GREEN if signal.risk_reward >= 2 else _C_YELLOW if signal.risk_reward >= 1 else _C_RED
    lines.append(f"  R:R    : {rr_color}{signal.risk_reward:.2f}{_C_RESET}")
    lines.append("")

    # Why
    if signal.contributing_factors:
        lines.append(f"  {_C_CYAN}── Why {'─' * 41}{_C_RESET}")
        for f in signal.contributing_factors[:8]:
            lines.append(f"    • {f}")

    if signal.failed_agents:
        lines.append("")
        lines.append(
            f"  {_C_YELLOW}⚠️  Agents with errors: "
            f"{', '.join(signal.failed_agents)}{_C_RESET}"
        )

    lines.append(f"{_C_BOLD}{'═' * 52}{_C_RESET}")
    return "\n".join(lines)


# ── Summary table (console) ──────────────────────────────

def format_summary_table_console(signals: List[Signal]) -> str:
    """
    Render a compact summary table for a list of signals (console/ANSI).
    """
    if not signals:
        return "  (no signals)\n"

    # Header
    hdr = (
        f"  {'Symbol':<10} {'TF':<5} {'Dir':<9} {'Grade':<9} "
        f"{'Conf':<6} {'Entry':>12} {'SL':>12} {'TP1':>12} "
        f"{'R:R':>6} {'Size':>10}"
    )
    sep = "  " + "─" * (len(hdr) - 2)

    lines = [sep, hdr, sep]

    for s in signals:
        d_emoji = _DIR_EMOJI.get(s.direction, "?")
        d_short = f"{d_emoji} {s.direction[:5]}"
        g_short = f"{_GRADE_EMOJI.get(s.grade, '?')} {s.grade}"
        row = (
            f"  {s.symbol:<10} {s.timeframe:<5} {d_short:<9} {g_short:<9} "
            f"{s.confidence:>5.0%} "
            f"{_fmt_price(s.entry_price):>12} "
            f"{_fmt_price(s.stop_loss):>12} "
            f"{_fmt_price(s.take_profit_1):>12} "
            f"{s.risk_reward:>6.2f} "
            f"${s.position_size_usd:>8,.0f}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append(f"  Total signals: {len(signals)}")
    no_trade = sum(1 for s in signals if s.direction == "NO_TRADE")
    if no_trade:
        lines.append(f"  No-trade: {no_trade}  |  Actionable: {len(signals) - no_trade}")

    return "\n".join(lines)


def format_summary_table_telegram(signals: List[Signal]) -> str:
    """
    Render a compact summary table for Telegram (no ANSI).
    """
    if not signals:
        return "(no signals)"

    lines: List[str] = []
    lines.append("═══ SUMMARY ═══════════════════")
    lines.append("")

    for s in signals:
        d_emoji = _DIR_EMOJI.get(s.direction, "❓")
        g_emoji = _GRADE_EMOJI.get(s.grade, "❓")
        if s.direction == "NO_TRADE":
            lines.append(
                f"  {d_emoji} {s.symbol} ({s.timeframe})  —  NO TRADE"
            )
        else:
            lines.append(
                f"  {d_emoji} {s.symbol} ({s.timeframe})  |  "
                f"{g_emoji} {s.grade}  |  "
                f"{s.confidence:.0%}  |  "
                f"R:R {s.risk_reward:.1f}"
            )

    lines.append("")
    lines.append(f"Total: {len(signals)} signals")
    no_trade = sum(1 for s in signals if s.direction == "NO_TRADE")
    if no_trade:
        lines.append(f"No-trade: {no_trade}  |  Actionable: {len(signals) - no_trade}")

    return "\n".join(lines)


# ── JSON output ────────────────────────────────────────────

def format_signals_json(signals: List[Signal]) -> str:
    """Return pretty-printed JSON for programmatic consumption."""
    return json.dumps(
        [s.to_dict() for s in signals],
        indent=2,
        default=str,
    )


# ── Top contributing factors ───────────────────────────────

def format_agent_breakdown(signal: Signal) -> str:
    """
    Detailed breakdown of every agent's contribution.
    Useful for debugging / Telegram deep-dive.
    """
    if not signal.agent_results:
        return "  (no agent data)"

    lines: List[str] = []
    lines.append(f"── Agent Breakdown: {signal.symbol} {signal.timeframe} ──")

    for r in signal.agent_results:
        if r.failed:
            icon = "❌"
            status = f"ERROR: {r.error}"
        else:
            d_emoji = _DIR_EMOJI.get(r.direction, "❓")
            icon = d_emoji
            status = f"{r.direction.upper()} @ {r.confidence:.0%}"

        lines.append(f"  {icon}  {r.agent_name:<18}  {status}")
        if r.reasoning:
            lines.append(f"       └─ {r.reasoning}")

    return "\n".join(lines)
