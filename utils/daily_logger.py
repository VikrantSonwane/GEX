"""
daily_logger.py
-----------------
Appends one row per live_loop() cycle to a CSV file named by today's
date, inside daily_data/. Purpose: build a REAL historical record of
every metric this engine computes (IV, GEX, gamma flip, walls, trend,
regime, signal) so that in a few months there's actual data to
validate against — hit rate, false-breakout rate, whether GEX regime
correlates with next-session realized vol, etc.

This is deliberately NOT a backtest and makes NO claims about what the
data will show — it's the raw material a future validation pass will
need. Nothing here interprets or scores anything.

One file per day (daily_data/2026-09-08.csv) rather than one giant
file, so a single day's data is easy to open/inspect/delete on its own
without touching the rest of the archive.
"""

from __future__ import annotations
import os
import datetime as dt
import pandas as pd

DAILY_DATA_DIR = "daily_data"


def log_snapshot(row: dict) -> None:
    """
    Appends `row` (a flat dict of metric_name -> value) to today's CSV,
    creating daily_data/ and the file (with header) if they don't exist
    yet. A 'logged_at' timestamp is added automatically — don't pass
    your own under that key.

    Column set is allowed to grow over time (e.g. if you add a new
    metric later): new columns just show up as blank in older rows
    when the file is eventually read back with pandas, since each
    day's file only has the columns that existed when IT was written.
    Keep that in mind when concatenating multiple days for analysis —
    use pd.concat(..., ignore_index=True) which handles the differing
    columns by filling NaN, rather than assuming every file matches.
    """
    os.makedirs(DAILY_DATA_DIR, exist_ok=True)
    today_str = dt.date.today().isoformat()
    path = os.path.join(DAILY_DATA_DIR, f"{today_str}.csv")

    full_row = {"logged_at": dt.datetime.now().isoformat(timespec="seconds"), **row}
    df_row = pd.DataFrame([full_row])

    header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=header, index=False)


def log_strike_breakdown(rows_df: pd.DataFrame) -> None:
    """
    Appends the FULL per-strike detail for one cycle (strike, option_type,
    oi, iv, gamma, strike_gex — whatever columns rows_df has) to a
    SEPARATE file, daily_data/strike_breakdown_YYYY-MM-DD.csv — one file
    per day, many rows per cycle (one per strike/option_type).

    Why this exists: log_snapshot() only stores the AGGREGATE total_gex
    per cycle — if a spike ever shows up in that aggregate (like the one
    diagnosed from a live spike), there's no way to know afterward WHICH
    strike caused it without this. With this, you can filter by
    timestamp and see exactly which row was the outlier, instead of
    inferring it indirectly.
    """
    os.makedirs(DAILY_DATA_DIR, exist_ok=True)
    today_str = dt.date.today().isoformat()
    path = os.path.join(DAILY_DATA_DIR, f"strike_breakdown_{today_str}.csv")

    out = rows_df.copy()
    out.insert(0, "logged_at", dt.datetime.now().isoformat(timespec="seconds"))

    header = not os.path.exists(path)
    out.to_csv(path, mode="a", header=header, index=False)
