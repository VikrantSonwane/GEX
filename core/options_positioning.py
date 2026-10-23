"""
options_positioning.py
--------------------------
The same Price x OI quadrant framework as futures_positioning.py, but
applied PER STRIKE, PER OPTION TYPE, using the weekly options chain
itself — not the monthly future. This matters because the future only
rolls monthly, while the options you actually trade expire weekly; the
future's OI can't tell you what's happening in THIS week's specific
puts and calls. This uses data already fetched every cycle (each
strike's real OI and real LTP, already used to solve IV) — no new API
calls.

    OI vs session open   Price(LTP) vs session open   Label
    up                   up                           LONG_BUILDUP
    up                   down                          SHORT_BUILDUP
    down                 up                            SHORT_COVERING
    down                 down                           LONG_UNWINDING

Tracked independently for every (strike, option_type) pair, then
aggregated separately for calls and puts — "is short building up in
puts overall" is the aggregate question; the per-strike detail is
there so you can see exactly which strikes are driving that.

Same honest limitation as the IV outlier guard's "new strike" gap: a
strike needs a session-opening baseline before it can be classified.
A strike entering the spot-narrowed chain window partway through the
day has no baseline yet — it's correctly labeled "insufficient_data",
not guessed at.

Pure sign classification, no tunable threshold, no overfitting surface.
"""

from __future__ import annotations
import pandas as pd


class OptionsPositioningTracker:

    def __init__(self):
        self._session_start: dict[tuple, dict] = {}  # (strike, option_type) -> {'oi':, 'ltp':}

    def update(self, chain_df: pd.DataFrame) -> pd.DataFrame:
        """
        chain_df needs columns: strike, option_type, oi, ltp.
        Returns a new DataFrame (one row per input row) with
        oi_change_vs_open, ltp_change_vs_open, positioning_label added.
        """
        records = []
        for _, row in chain_df.iterrows():
            key = (row["strike"], row["option_type"])
            if key not in self._session_start:
                self._session_start[key] = {"oi": row["oi"], "ltp": row["ltp"]}
                label = "insufficient_data"
                oi_change, ltp_change = 0.0, 0.0
            else:
                base = self._session_start[key]
                oi_change = row["oi"] - base["oi"]
                ltp_change = row["ltp"] - base["ltp"]
                if oi_change == 0 or ltp_change == 0:
                    label = "neutral"
                elif oi_change > 0 and ltp_change > 0:
                    label = "long_buildup"
                elif oi_change > 0 and ltp_change < 0:
                    label = "short_buildup"
                elif oi_change < 0 and ltp_change > 0:
                    label = "short_covering"
                else:
                    label = "long_unwinding"

            records.append({
                "strike": row["strike"], "option_type": row["option_type"],
                "oi_change_vs_open": oi_change, "ltp_change_vs_open": ltp_change,
                "positioning_label": label,
            })
        return pd.DataFrame(records)

    @staticmethod
    def summarize(detail_df: pd.DataFrame) -> dict:
        """
        Aggregates the per-row detail into the "is short building up
        overall" answer: total OI change and a count of strikes in each
        bucket, separately for calls (CE) and puts (PE).
        """
        summary = {}
        for opt_type, prefix in (("CE", "call"), ("PE", "put")):
            sub = detail_df[detail_df["option_type"] == opt_type]
            summary[f"{prefix}_total_oi_change"] = float(sub["oi_change_vs_open"].sum())
            counts = sub["positioning_label"].value_counts().to_dict()
            summary[f"{prefix}_positioning_counts"] = counts
            # a simple, honest majority read -- NOT a score, just "which
            # bucket has the most strikes in it right now"
            summary[f"{prefix}_dominant_label"] = (
                max(counts, key=counts.get) if counts else "insufficient_data")
        return summary
