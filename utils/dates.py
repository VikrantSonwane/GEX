"""
dates.py
----------
Time-to-expiry calculation for options pricing. Replaces the
`t_years = 7/365.0` placeholder that was in main.py — every IV/greeks
calculation needs a REAL time-to-expiry, not a stand-in constant.

Counts to the exchange's actual settlement cut-off (config.MARKET_CLOSE_TIME,
default 15:30 IST for NSE) on the expiry date, not midnight — a few hours'
difference matters for very short-dated (0-2 day) options, where theta is
large and t_years close to zero.
"""

from __future__ import annotations
import datetime as dt

TRADING_DAYS_PER_YEAR = 365.0  # calendar-day count convention, matching
                                # how Black-76 t_years is used elsewhere
                                # in this project (see stats_utils.py's
                                # separate TRADING_DAYS_PER_YEAR=252,
                                # which is the *trading*-day convention
                                # used for annualizing volatility — these
                                # are deliberately different conventions
                                # for different purposes, not a mismatch)


def parse_expiry(expiry_str: str, expiry_format: str | None = None) -> dt.date:
    """
    expiry_format: an explicit strptime format string (e.g. '%d-%b-%Y')
    matching your scrip master's exact expiry format — get this by
    running inspect_scrip_master.py once and reading off a real value.
    If None, tries a handful of common formats before giving up; prefer
    passing the explicit format once you know it, since silent format
    guessing is exactly the kind of "artificial value" this project
    avoids elsewhere — this fallback exists only to get you unblocked
    before your first inspect_scrip_master.py run.
    """
    if expiry_format:
        return dt.datetime.strptime(expiry_str, expiry_format).date()

    candidate_formats = ["%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d%b%Y", "%Y%m%d"]
    for fmt in candidate_formats:
        try:
            return dt.datetime.strptime(expiry_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Could not parse expiry string '{expiry_str}' with any known format. "
        f"Run inspect_scrip_master.py, read the exact format off a real row, "
        f"and pass it explicitly as expiry_format."
    )


def time_to_expiry_years(expiry_str: str, expiry_format: str | None = None,
                          market_close_time: dt.time = dt.time(15, 30),
                          now: dt.datetime | None = None) -> float:
    """
    Real T (in years) from now until the expiry date's market close —
    the value every function in core/options_math.py needs as t_years.
    Returns 0.0 (not negative) if expiry has already passed today, so
    callers get intrinsic-value pricing rather than a nonsensical
    negative time.
    """
    now = now or dt.datetime.now()
    expiry_date = parse_expiry(expiry_str, expiry_format)
    expiry_dt = dt.datetime.combine(expiry_date, market_close_time)

    seconds_remaining = (expiry_dt - now).total_seconds()
    if seconds_remaining <= 0:
        return 0.0
    return seconds_remaining / (TRADING_DAYS_PER_YEAR * 24 * 3600)
