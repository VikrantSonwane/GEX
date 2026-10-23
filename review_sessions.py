"""
review_sessions.py
---------------------
Run this manually, once you've accumulated daily_data/*.csv files across
enough sessions to be worth looking at (your own guidance: ~20+). This is
NOT part of main.py's live loop — it only reads what's already been
logged, never touches anything live.

What it does:
  1. Loads every daily_data/*.csv (one file per trading day) and combines
     them into one table.
  2. For each logged cycle, computes REALIZED forward price moves at a
     few horizons (15/30/60 minutes ahead, and the move to that day's
     final logged reading) — using that SAME day's own later rows as
     ground truth. Since spot was already logged every ~60 seconds all
     day, this is a lookup, not a guess or an external data pull.
  3. Writes the combined, outcome-annotated table to
     daily_data_with_outcomes.csv, so you can open it in Excel and slice
     by dealer_regime / signal_strategy / vol_regime / trend_direction
     yourself — pivot tables, filters, whatever you're comfortable with.
  4. Prints a few honest descriptive summaries — explicitly NOT a
     verdict. With a handful of sessions, group averages are barely more
     than noise; the script says so out loud rather than implying a
     conclusion the sample size can't support.

Usage:
    python review_sessions.py
"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd

DAILY_DATA_DIR = "daily_data"
FORWARD_HORIZONS_MINUTES = [15, 30, 60]


def load_all_sessions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{DAILY_DATA_DIR}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {DAILY_DATA_DIR}/ — run main.py "
                                  f"for at least one full session first.")
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["logged_at"])
        df["session_date"] = df["logged_at"].dt.date
        frames.append(df)
    # pd.concat fills missing columns with NaN automatically if an older
    # log file predates a metric that got added later — nothing special
    # needed here, this is exactly the scenario daily_logger.py's own
    # docstring flags.
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("logged_at").reset_index(drop=True)


def add_forward_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, finds the spot price at T+N minutes WITHIN THE SAME
    TRADING DAY only (never crosses a day boundary — an overnight gap is
    a completely different phenomenon from intraday continuation, and
    mixing them would corrupt the comparison) and records the realized
    point move. Uses binary search (np.searchsorted) against that day's
    own sorted timestamps — fast, and no room for the off-by-one errors
    a naive loop invites.
    """
    df = df.copy()
    for minutes in FORWARD_HORIZONS_MINUTES:
        df[f"realized_move_{minutes}min"] = np.nan
    df["realized_move_to_session_close"] = np.nan

    for date, day_idx in df.groupby("session_date").groups.items():
        day_df = df.loc[day_idx].sort_values("logged_at")
        times = day_df["logged_at"].values
        spots = day_df["spot"].values
        idxs = day_df.index.values

        for minutes in FORWARD_HORIZONS_MINUTES:
            target_times = times + np.timedelta64(minutes, "m")
            # searchsorted finds, for each target time, the first index
            # in `times` that is >= that target — i.e. the first real
            # logged reading at or after "now + N minutes".
            j = np.searchsorted(times, target_times, side="left")
            valid = j < len(times)
            col = f"realized_move_{minutes}min"
            df.loc[idxs[valid], col] = spots[j[valid]] - spots[valid]

        last_spot = spots[-1]
        df.loc[idxs, "realized_move_to_session_close"] = last_spot - spots

    return df


def print_summary(df: pd.DataFrame):
    n_sessions = df["session_date"].nunique()
    print(f"\nLoaded {len(df)} logged cycles across {n_sessions} session(s).")
    if n_sessions < 20:
        print(f"  [note] fewer than 20 sessions so far — treat everything below as "
              f"exploratory, not conclusive. Small samples are noisy; a pattern "
              f"here could easily reverse with 10 more sessions of data.")

    outcome_cols = [c for c in df.columns if c.startswith("realized_move_")]

    print("\n--- Average forward move (points), by dealer_regime ---")
    print(df.groupby("dealer_regime")[outcome_cols].mean().round(1))

    print("\n--- Row count per dealer_regime (sample size — read the averages above WITH this) ---")
    print(df["dealer_regime"].value_counts())

    print("\n--- Average forward move (points), by signal_strategy ---")
    print(df.groupby("signal_strategy")[outcome_cols].mean().round(1))

    print("\n--- Row count per signal_strategy ---")
    print(df["signal_strategy"].value_counts())

    print("\n--- Std deviation of forward move (a crude realized-vol proxy), by dealer_regime ---")
    print("(if positive_gamma really dampens moves the way the theory predicts, "
          "this should come out LOWER for positive_gamma than negative_gamma — "
          "but only trust this once each regime has a meaningful row count above)")
    print(df.groupby("dealer_regime")[outcome_cols].std().round(1))


if __name__ == "__main__":
    combined = load_all_sessions()
    combined = add_forward_outcomes(combined)
    out_path = f"{DAILY_DATA_DIR}_with_outcomes.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows, {len(combined.columns)} columns)")
    print_summary(combined)
