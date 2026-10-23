"""
iv_skew.py
------------
Volatility skew: compares implied vol at a few different points along
the chain (roughly 25-delta put, ATM, roughly 25-delta call) instead of
just the single ATM number VRP already uses. Skew tells you whether the
market is pricing downside moves richer than upside ones (or vice versa)
— a real, standard piece of options-market information that's free to
compute here since every strike's IV is already solved each cycle
(core/options_math.py's implied_volatility()).

Convention used: skew = put_iv - call_iv. Positive skew (puts pricier
than calls) is the normal, expected shape for equity/index options —
crashes are feared more than rallies. A skew that's unusually flat or
inverted relative to its own recent history is the informative signal,
not the raw sign (which is almost always positive for NIFTY).
"""

from __future__ import annotations
import pandas as pd


def compute_skew(chain_df: pd.DataFrame, target_abs_delta: float = 0.25) -> dict:
    """
    chain_df expects columns: ['strike', 'option_type', 'iv', 'delta'].

    Finds the call whose delta is closest to +target_abs_delta, the put
    whose delta is closest to -target_abs_delta, and the strike closest
    to spot's own IV (using the smallest |delta| as the ATM proxy —
    consistent with how the rest of this project already treats "ATM").

    Returns None values (not a guess) if the chain doesn't have a call
    or put in a sane range to match against — better to say "couldn't
    find one" than silently picking something far off.
    """
    calls = chain_df[chain_df["option_type"].str.upper() == "CE"]
    puts = chain_df[chain_df["option_type"].str.upper() == "PE"]

    if calls.empty or puts.empty:
        return {"atm_iv": None, "call_25d_iv": None, "put_25d_iv": None, "skew": None}

    call_row = calls.iloc[(calls["delta"] - target_abs_delta).abs().argmin()]
    put_row = puts.iloc[(puts["delta"] + target_abs_delta).abs().argmin()]

    all_deltas_abs = chain_df["delta"].abs()
    atm_row = chain_df.iloc[all_deltas_abs.sub(0.5).abs().argmin()]

    call_iv = float(call_row["iv"])
    put_iv = float(put_row["iv"])
    atm_iv = float(atm_row["iv"])
    skew = put_iv - call_iv

    return {
        "atm_iv": atm_iv,
        "call_25d_iv": call_iv,
        "call_25d_strike": float(call_row["strike"]),
        "put_25d_iv": put_iv,
        "put_25d_strike": float(put_row["strike"]),
        "skew": float(skew),
    }
