"""
ICT Kill Zones — Institutional trading windows.

Kill Zones are specific time periods during the trading day when institutional
activity is highest. ICT identifies four key zones:

- Asian Kill Zone (20:00–00:00 UTC): Accumulation phase. Smart money builds
  positions. Typically ranges sideways.
- London Kill Zone (02:00–05:00 UTC): Manipulation phase. Stop hunts and
  fake-outs occur. Price sweeps liquidity.
- New York Kill Zone (07:00–10:00 UTC): Distribution phase. The real move
  happens. Price trends aggressively.
- London Close (10:00–12:00 UTC): Reversal window. Profit-taking and new
  setups for the next session.

These are general guides — actual sessions may shift with DST (not applied
for crypto markets which trade 24/7).
"""

from typing import Optional, Tuple
from datetime import datetime, time, timezone, timedelta
import pandas as pd


# Kill zone definitions as (start_hour, start_min, end_hour, end_min) in UTC
KILL_ZONES = {
    "asian": {
        "start": time(20, 0, tzinfo=timezone.utc),
        "end": time(0, 0, tzinfo=timezone.utc),
        "bias": "accumulation",
        "description": "Accumulation — smart money builds positions in ranging conditions",
    },
    "london": {
        "start": time(2, 0, tzinfo=timezone.utc),
        "end": time(5, 0, tzinfo=timezone.utc),
        "bias": "manipulation",
        "description": "Manipulation — stop hunts, fake breakouts, liquidity sweeps",
    },
    "new_york": {
        "start": time(7, 0, tzinfo=timezone.utc),
        "end": time(10, 0, tzinfo=timezone.utc),
        "bias": "distribution",
        "description": "Distribution — the real move, institutional order flow",
    },
    "london_close": {
        "start": time(10, 0, tzinfo=timezone.utc),
        "end": time(12, 0, tzinfo=timezone.utc),
        "bias": "distribution",
        "description": "Reversal window — profit-taking, new setups form",
    },
}


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _time_in_range(t: time, start: time, end: time) -> bool:
    """
    Check if a time is within a range, handling midnight wraparound.

    Args:
        t: Time to check.
        start: Range start time.
        end: Range end time.

    Returns:
        True if t is in [start, end) accounting for midnight crossing.
    """
    t_val = t.hour * 60 + t.minute
    s_val = start.hour * 60 + start.minute
    e_val = end.hour * 60 + end.minute

    if s_val <= e_val:
        # Normal range (no midnight wrap)
        return s_val <= t_val < e_val
    else:
        # Wraps midnight (e.g., 20:00–00:00)
        return t_val >= s_val or t_val < e_val


def get_killzone(dt: datetime) -> Optional[str]:
    """
    Get the active Kill Zone name for a given datetime.

    Args:
        dt: Datetime to check (timezone-aware recommended, assumed UTC if naive).

    Returns:
        Kill zone name ('asian', 'london', 'new_york', 'london_close') or None
        if not within any kill zone.
    """
    dt = _ensure_utc(dt)
    t = dt.time()

    # Check in order of specificity (London Close is subset of others)
    if _time_in_range(t, KILL_ZONES["london_close"]["start"], KILL_ZONES["london_close"]["end"]):
        return "london_close"
    if _time_in_range(t, KILL_ZONES["new_york"]["start"], KILL_ZONES["new_york"]["end"]):
        return "new_york"
    if _time_in_range(t, KILL_ZONES["london"]["start"], KILL_ZONES["london"]["end"]):
        return "london"
    if _time_in_range(t, KILL_ZONES["asian"]["start"], KILL_ZONES["asian"]["end"]):
        return "asian"

    return None


def is_killzone_active(dt: datetime) -> bool:
    """
    Check if any Kill Zone is active at the given datetime.

    Args:
        dt: Datetime to check.

    Returns:
        True if within any kill zone.
    """
    return get_killzone(dt) is not None


def killzone_bias(killzone: str) -> str:
    """
    Get the ICT bias/phase for a kill zone.

    Args:
        killzone: Kill zone name ('asian', 'london', 'new_york', 'london_close').

    Returns:
        Bias string: 'accumulation', 'manipulation', or 'distribution'.
    """
    if killzone in KILL_ZONES:
        return KILL_ZONES[killzone]["bias"]
    raise ValueError(f"Unknown kill zone: {killzone}. Valid: {list(KILL_ZONES.keys())}")


def get_killzone_info(killzone: str) -> dict:
    """
    Get full information about a kill zone.

    Args:
        killzone: Kill zone name.

    Returns:
        Dict with 'start', 'end', 'bias', 'description'.
    """
    if killzone in KILL_ZONES:
        return dict(KILL_ZONES[killzone])
    raise ValueError(f"Unknown kill zone: {killzone}. Valid: {list(KILL_ZONES.keys())}")


def get_next_killzone(dt: datetime) -> Tuple[str, datetime]:
    """
    Get the next kill zone after the given datetime.

    Args:
        dt: Current datetime (UTC).

    Returns:
        Tuple of (zone_name, zone_start_datetime).
    """
    dt = _ensure_utc(dt)
    candidates = []

    for name, info in KILL_ZONES.items():
        start = info["start"]
        # Build a datetime for the next occurrence of this start time
        next_start = dt.replace(
            hour=start.hour, minute=start.minute, second=0, microsecond=0
        )
        if next_start <= dt:
            next_start += timedelta(days=1)
        candidates.append((next_start, name))

    candidates.sort()
    return candidates[0][1], candidates[0][0]


def time_until_killzone(dt: datetime) -> Tuple[Optional[str], float]:
    """
    Calculate time until next kill zone and which zone it is.

    Args:
        dt: Current datetime.

    Returns:
        Tuple of (next_zone_name, seconds_until_start).
        If currently in a kill zone, returns (current_zone, 0.0).
    """
    dt = _ensure_utc(dt)
    current = get_killzone(dt)
    if current is not None:
        return current, 0.0

    zone_name, zone_start = get_next_killzone(dt)
    delta = (zone_start - dt).total_seconds()
    return zone_name, delta
