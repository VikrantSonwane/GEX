"""
realized_vol.py
------------------
Tracks spot readings during a live session and computes trailing
realized volatility over a few short windows (5/15/30 minutes by
default). This is the intraday counterpart to the GARCH forecast,
which only updates once per day — comparing ATM IV against THIS
(what's actually happening in the last few minutes) answers "is
volatility expanding right now," which a once-daily forecast can't.

No new data source needed: main.py's live_loop() already gets a fresh
spot reading every cycle. This just keeps a short rolling buffer of
those readings in memory for the current session (cleared each time
you restart main.py — it's a same-day tool, not a persisted history).
"""

from __future__ import annotations
from collections import deque
import datetime as dt
import numpy as np

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_TRADING_DAY = 375  # NSE session length


class RealizedVolTracker:

    def __init__(self, max_window_minutes: int = 60):
        """
        max_window_minutes: how far back to keep readings for — must be
        at least as large as the longest window you'll ever ask for
        (default windows go up to 30 min; 60 min of buffer gives margin).
        """
        self._max_window = dt.timedelta(minutes=max_window_minutes)
        self._readings: deque[tuple[dt.datetime, float]] = deque()

    def update(self, timestamp: dt.datetime, spot: float) -> None:
        """Call once per live_loop() cycle with the current timestamp and spot."""
        self._readings.append((timestamp, spot))
        cutoff = timestamp - self._max_window
        while self._readings and self._readings[0][0] < cutoff:
            self._readings.popleft()

    def realized_vol(self, window_minutes: int, annualize: bool = True) -> float | None:
        """
        Annualized realized volatility computed from log returns between
        consecutive readings within the trailing `window_minutes`.
        Returns None if there aren't at least 3 readings in that window
        yet (too little data for a meaningful number — same "don't fake
        it with insufficient data" rule as everywhere else in this
        project).
        """
        if not self._readings:
            return None
        latest_time = self._readings[-1][0]
        cutoff = latest_time - dt.timedelta(minutes=window_minutes)
        window = [spot for (t, spot) in self._readings if t >= cutoff]
        if len(window) < 3:
            return None

        log_returns = np.diff(np.log(window))
        if len(log_returns) == 0 or np.std(log_returns) == 0:
            return 0.0

        # Scale from "per-observation" to annualized: observations are
        # roughly 1 per minute (matches SIGNAL_POLL_SECONDS=60), so treat
        # each return as one "tick" out of MINUTES_PER_TRADING_DAY*TRADING_DAYS_PER_YEAR
        # ticks in a year. If your poll interval differs, this scaling
        # implicitly assumes ~1-minute spacing; a materially different
        # interval would need this factor adjusted.
        periods_per_year = MINUTES_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR
        vol = np.std(log_returns, ddof=1)
        return float(vol * np.sqrt(periods_per_year)) if annualize else float(vol)

    def snapshot(self, windows_minutes: list[int] = (5, 15, 30)) -> dict:
        """Convenience: realized vol at each of several windows in one call."""
        return {f"realized_vol_{m}min": self.realized_vol(m) for m in windows_minutes}
