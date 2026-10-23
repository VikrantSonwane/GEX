"""
futures_positioning.py
--------------------------
The classic Price x OI quadrant framework, applied to NIFTY futures'
own OI (real, live, from Kotak — no external data source needed):

    Futures price   Futures OI    Label
    up              up            LONG_BUILDUP    (fresh bullish positioning)
    down            up            SHORT_BUILDUP   (fresh bearish positioning)
    up              down          SHORT_COVERING  (bearish move, but exhaustible)
    down            down          LONG_UNWINDING  (bearish move, also less durable)

This is deliberately the INTRADAY-NATIVE version of this framework —
compared against the session's own opening OI/price, not yesterday's
close (the textbook daily version) and not the NSE FII participant
report (which is end-of-day data, published once, and cannot refresh
during market hours — a fundamentally different, slower-moving thing,
not built here).

Like gamma exposure, this is a MEASUREMENT, not a signal by itself —
main.py logs and displays it, signal_engine.py does not consume it.
No hand-tuned threshold anywhere: buildup/unwinding is a pure sign
classification (did price/OI rise or fall vs session open), not a
magnitude-based rule that would need calibrating.
"""

from __future__ import annotations


class FuturesPositioningTracker:

    def __init__(self):
        self._session_start_price: float | None = None
        self._session_start_oi: float | None = None

    def update(self, futures_price: float, futures_oi: float | None) -> dict:
        """
        Call once per live_loop() cycle with the current futures LTP and
        OI. Returns the classification dict every time; the FIRST call
        of a session establishes the baseline and returns a "no baseline
        yet" state rather than guessing a direction from nothing.
        """
        if futures_oi is None:
            return self._result(None, None, None, "unavailable",
                                 "futures OI not available this cycle")

        if self._session_start_price is None:
            self._session_start_price = futures_price
            self._session_start_oi = futures_oi
            return self._result(0.0, 0.0, futures_oi, "insufficient_data",
                                 "session baseline just set — no change to classify yet")

        price_change = futures_price - self._session_start_price
        oi_change = futures_oi - self._session_start_oi

        if price_change == 0 or oi_change == 0:
            label = "neutral"
            note = "no net price or OI change vs session open yet"
        elif price_change > 0 and oi_change > 0:
            label = "long_buildup"
            note = "price up, OI up vs session open — fresh bullish positioning"
        elif price_change < 0 and oi_change > 0:
            label = "short_buildup"
            note = "price down, OI up vs session open — fresh bearish positioning"
        elif price_change > 0 and oi_change < 0:
            label = "short_covering"
            note = "price up, OI down vs session open — bearish positions closing, may be exhaustible"
        else:  # price_change < 0 and oi_change < 0
            label = "long_unwinding"
            note = "price down, OI down vs session open — bullish positions closing, may be exhaustible"

        return self._result(price_change, oi_change, futures_oi, label, note)

    @staticmethod
    def _result(price_change, oi_change, current_oi, label, note) -> dict:
        return {
            "futures_price_change_vs_open": price_change,
            "futures_oi_change_vs_open": oi_change,
            "futures_oi_current": current_oi,
            "positioning_label": label,
            "positioning_note": note,
        }
