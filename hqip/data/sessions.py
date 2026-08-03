"""
HQIP v3 — Session & Kill-Zone Manager
=======================================

Maps UTC timestamps to the correct trading session and ICT kill zones.
Sessions determine position-sizing multipliers; kill zones narrow
high-probability entry windows.

Session Definitions (UTC)
-------------------------
- **Asia**     00:00 – 08:00  (low volatility)
- **Europe**   07:00 – 16:00  (medium)
- **America**  13:00 – 22:00  (high)
- **Overlap**  13:00 – 16:00  (highest — Europe ∩ America)
- **Off-Hours** 22:00 – 00:00 (lowest)

ICT Kill Zones
--------------
- **London Open**  07:00 – 10:00
- **NY Open**      12:00 – 15:00
- **London Close** 15:00 – 17:00

Usage::

    from hqip.data.sessions import SessionManager

    sm = SessionManager()
    session = sm.get_session(datetime.now(timezone.utc))
    multiplier = sm.get_volatility_multiplier(session)
    in_kz = sm.is_killzone(datetime.now(timezone.utc))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Volatility profiles
# ---------------------------------------------------------------------------

# Maps session name → position-sizing multiplier.
_VOLATILITY_MULTIPLIER: dict[str, float] = {
    "asian":    0.75,
    "asia":     0.75,
    "european": 1.0,
    "europe":   1.0,
    "american": 1.25,
    "america":  1.25,
    "overlap":  1.5,
    "off-hours": 0.5,
}


# ---------------------------------------------------------------------------
# Session definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """A named time window with a volatility profile."""

    name: str
    display: str
    start: time          # UTC
    end: time            # UTC
    volatility: str      # human label
    multiplier: float    # position-sizing multiplier


_SESSIONS: list[Session] = [
    Session(
        name="asia", display="Asian Session",
        start=time(0, 0), end=time(8, 0),
        volatility="low", multiplier=0.75,
    ),
    Session(
        name="europe", display="European Session",
        start=time(7, 0), end=time(16, 0),
        volatility="medium", multiplier=1.0,
    ),
    Session(
        name="america", display="American Session",
        start=time(13, 0), end=time(22, 0),
        volatility="high", multiplier=1.25,
    ),
    Session(
        name="overlap", display="Europe–America Overlap",
        start=time(13, 0), end=time(16, 0),
        volatility="highest", multiplier=1.5,
    ),
    Session(
        name="off-hours", display="Off-Hours",
        start=time(22, 0), end=time(0, 0),
        volatility="lowest", multiplier=0.5,
    ),
]


# ---------------------------------------------------------------------------
# Kill-zone definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KillZone:
    """ICT kill zone — a narrow high-probability entry window."""

    name: str
    display: str
    start: time
    end: time


_KILL_ZONES: list[KillZone] = [
    KillZone(
        name="london_open", display="ICT London Open Kill Zone",
        start=time(7, 0), end=time(10, 0),
    ),
    KillZone(
        name="ny_open", display="ICT New York Open Kill Zone",
        start=time(12, 0), end=time(15, 0),
    ),
    KillZone(
        name="london_close", display="ICT London Close Kill Zone",
        start=time(15, 0), end=time(17, 0),
    ),
]


# ---------------------------------------------------------------------------
# Helper: time-range membership (handles midnight wrap)
# ---------------------------------------------------------------------------

def _time_in_range(t: time, start: time, end: time) -> bool:
    """Return True if *t* is within [start, end), handling midnight wrap.

    When ``start >= end`` (e.g. 22:00 → 00:00), the range wraps past
    midnight.
    """
    if start <= end:
        return start <= t < end
    # Wraps past midnight
    return t >= start or t < end


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Resolve UTC timestamps to trading sessions and kill zones.

    Parameters
    ----------
    sessions : list[Session] | None
        Override the default session definitions.
    kill_zones : list[KillZone] | None
        Override the default kill-zone definitions.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> sm = SessionManager()
    >>> now = datetime.now(timezone.utc)
    >>> sm.get_session(now)
    'europe'
    >>> sm.is_killzone(now)
    False
    """

    def __init__(
        self,
        sessions: Optional[list[Session]] = None,
        kill_zones: Optional[list[KillZone]] = None,
    ) -> None:
        self._sessions = sessions or list(_SESSIONS)
        self._kill_zones = kill_zones or list(_KILL_ZONES)
        # Lookup dicts for O(1) access
        self._session_map: dict[str, Session] = {s.name: s for s in self._sessions}
        self._kz_map: dict[str, KillZone] = {kz.name: kz for kz in self._kill_zones}

    # ------------------------------------------------------------------
    # Session resolution
    # ------------------------------------------------------------------
    def get_session(self, dt: datetime) -> str:
        """Return the primary session name for a UTC datetime.

        When multiple sessions overlap, priority order is:
        ``overlap`` > ``america`` > ``europe`` > ``asia`` > ``off-hours``.

        Parameters
        ----------
        dt : datetime
            A UTC-aware datetime.

        Returns
        -------
        str
            Session name (e.g. ``"europe"``).
        """
        t = dt.astimezone(timezone.utc).time()
        # Check in priority order (overlap first)
        priority = ["overlap", "america", "europe", "asia", "off-hours"]
        for name in priority:
            sess = self._session_map.get(name)
            if sess and _time_in_range(t, sess.start, sess.end):
                return sess.name
        # Fallback (should never reach here if sessions cover 24 h)
        return "off-hours"

    def get_session_detail(self, dt: datetime) -> Session:
        """Return the full :class:`Session` dataclass for *dt*."""
        name = self.get_session(dt)
        return self._session_map[name]

    def get_volatility_multiplier(self, session_name: str) -> float:
        """Return the position-sizing multiplier for *session_name*.

        Parameters
        ----------
        session_name : str
            E.g. ``"europe"`` or ``"overlap"``.

        Returns
        -------
        float
            Multiplier (0.5 – 1.5).
        """
        sess = self._session_map.get(session_name)
        if sess is not None:
            return sess.multiplier
        return _VOLATILITY_MULTIPLIER.get(session_name, 1.0)

    # ------------------------------------------------------------------
    # Kill-zone resolution
    # ------------------------------------------------------------------
    def is_killzone(self, dt: datetime) -> bool:
        """Return True if *dt* falls inside any ICT kill zone.

        Parameters
        ----------
        dt : datetime
            A UTC-aware datetime.
        """
        t = dt.astimezone(timezone.utc).time()
        return any(
            _time_in_range(t, kz.start, kz.end) for kz in self._kill_zones
        )

    def get_killzone(self, dt: datetime) -> Optional[KillZone]:
        """Return the active :class:`KillZone` for *dt*, or ``None``."""
        t = dt.astimezone(timezone.utc).time()
        for kz in self._kill_zones:
            if _time_in_range(t, kz.start, kz.end):
                return kz
        return None

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------
    def list_sessions(self) -> list[Session]:
        """Return all configured sessions."""
        return list(self._sessions)

    def list_kill_zones(self) -> list[KillZone]:
        """Return all configured kill zones."""
        return list(self._kill_zones)

    def session_schedule(self) -> dict[str, dict[str, str]]:
        """Return a human-readable schedule dict.

        Returns
        -------
        dict
            ``{"asia": {"start": "00:00", "end": "08:00", ...}, ...}``
        """
        return {
            s.name: {
                "display": s.display,
                "start": s.start.strftime("%H:%M"),
                "end": s.end.strftime("%H:%M"),
                "volatility": s.volatility,
                "multiplier": str(s.multiplier),
            }
            for s in self._sessions
        }

    def __repr__(self) -> str:
        return (
            f"SessionManager(sessions={len(self._sessions)}, "
            f"kill_zones={len(self._kill_zones)})"
        )
