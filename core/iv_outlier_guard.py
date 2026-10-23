"""
iv_outlier_guard.py
----------------------
Catches the exact failure mode diagnosed from a live GEX spike: a single
strike's LTP goes stale/delayed for one polling cycle, the IV solver
dutifully backs out whatever IV makes that stale price "correct," and if
that strike carries real OI, its distorted gamma briefly dominates
total_gex — a sharp, self-correcting spike with no matching move in
ATM IV (since ATM IV only reflects one strike, not the whole chain).

How it decides "implausible," without a hardcoded percentage:
Each cycle, for every strike that also appeared last cycle, computes how
much its IV changed. Most strikes' IV changes by a small amount cycle to
cycle (real, smooth) — a genuinely bad quote changes by a LOT relative to
that cycle's own typical change. The threshold is derived from THIS
cycle's own distribution of changes (median + a multiple of the median
absolute deviation, a robust "how unusual is this outlier" measure) —
not a fixed number that would need retuning as volatility regimes shift.

This is the PROVEN, SIMPLE version — deliberately time-series-only. An
earlier attempt to also add a cross-sectional check (for strikes with no
prior-cycle history, e.g. a strike newly entering the spot-narrowed
chain window) introduced real bugs of its own (false positives from
neighbor-comparison edge effects) that were not fully resolved. Rather
than ship a half-fixed extension, this reverts to the simple version
that passed all 6 of its tests cleanly. The "new strike has no history
to check" gap is real and still open — a known, accepted limitation,
not silently pretended away.
"""

from __future__ import annotations
import pandas as pd


class IVOutlierGuard:

    def __init__(self, mad_multiplier: float = 6.0, min_comparable_strikes: int = 5):
        """
        mad_multiplier: how many median-absolute-deviations above the
        typical cycle-to-cycle IV change counts as an outlier. 6 is
        deliberately generous (real IV can legitimately jump on fast
        moves) — the goal is catching clearly-broken single-strike
        quotes, not routine volatility.
        min_comparable_strikes: need at least this many strikes with a
        previous reading before trusting a MAD computed from them —
        early in a session (or right after this guard is first created)
        there's nothing to compare against yet, so nothing gets flagged.
        """
        self.mad_multiplier = mad_multiplier
        self.min_comparable_strikes = min_comparable_strikes
        self._previous_iv: dict[tuple, float] = {}

    def filter(self, chain_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (clean_chain, flagged_rows). clean_chain excludes any
        row judged an outlier this cycle; flagged_rows is exactly those
        excluded rows (with their IV change amount attached), for
        printing/logging — never silently dropped without a trace.
        """
        df = chain_df.copy()
        keys = list(zip(df["strike"], df["option_type"]))
        df["prev_iv"] = [self._previous_iv.get(k) for k in keys]
        df["iv_change"] = (df["iv"] - df["prev_iv"]).abs()

        has_prev = df["iv_change"].notna()
        if has_prev.sum() < self.min_comparable_strikes:
            flagged = df.iloc[0:0].copy()
            clean = df.copy()
        else:
            changes = df.loc[has_prev, "iv_change"]
            median = changes.median()
            mad = (changes - median).abs().median()
            threshold = median + self.mad_multiplier * max(mad, 1e-6)
            is_outlier = has_prev & (df["iv_change"] > threshold)
            flagged = df[is_outlier].copy()
            clean = df[~is_outlier].copy()

        # Update state with EVERY reading this cycle (flagged or not) —
        # next cycle compares against what was actually quoted, not a
        # guess. A flagged reading isn't a permanent judgment; if the
        # same strike shows a similarly-shifted IV next cycle too, that
        # stops looking like a one-off bad quote and starts looking like
        # a real move, and it will no longer be flagged then.
        for k, iv in zip(keys, df["iv"]):
            self._previous_iv[k] = iv

        return clean.drop(columns=["prev_iv", "iv_change"]), flagged.drop(columns=["prev_iv"])
