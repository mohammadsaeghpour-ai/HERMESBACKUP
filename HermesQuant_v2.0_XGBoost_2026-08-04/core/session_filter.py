"""
Session / Kill Zone Filter
Best trading hours based on volume data:
- Europe Open: 08:00-10:00 UTC (12:30-14:30 Tehran)
- US Open: 14:00-16:00 UTC (18:30-20:30 Tehran)
- Europe/US Overlap: 14:00-16:00 UTC (best)
"""
from datetime import datetime, timezone, timedelta

tz = timezone(timedelta(hours=3, minutes=30))

# Kill Zones in Tehran time
KILL_ZONES = {
    "europe_open": (12, 14),     # 12:00-14:00 Tehran
    "us_open": (18, 20),         # 18:00-20:00 Tehran
    "overlap": (18, 20),         # Best session
}

# Volatility multipliers by hour (Tehran time)
HOURLY_VOL = {
    0: 0.3, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4,
    6: 0.5, 7: 0.6, 8: 0.7, 9: 0.8, 10: 0.9, 11: 1.0,
    12: 1.1, 13: 1.2, 14: 1.1, 15: 1.0, 16: 0.9, 17: 1.0,
    18: 1.3, 19: 1.4, 20: 1.2, 21: 1.0, 22: 0.7, 23: 0.5,
}

def in_kill_zone():
    """Check if current time is in a kill zone"""
    now = datetime.now(tz)
    hour = now.hour
    return (12 <= hour <= 14) or (18 <= hour <= 20)

def session_multiplier():
    """Return volume multiplier for current hour"""
    now = datetime.now(tz)
    return HOURLY_VOL.get(now.hour, 0.5)

def is_best_session():
    """Check if in best trading session"""
    now = datetime.now(tz)
    return 18 <= now.hour <= 20
