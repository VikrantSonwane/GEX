"""
test_iv_outlier_guard.py
----------------------------
Standalone, re-runnable test suite for core/iv_outlier_guard.py. Run
this yourself, anytime, to independently verify the guard behaves
correctly — don't just take a chat transcript's word for it.

Covers all 6 scenarios from the test plan discussed:
  1. Normal market      -> nothing incorrectly removed
  2. Genuine broad move -> keep it (don't mass-flag a real event)
  3. One bad quote      -> catch and exclude it
  4. Several bad quotes at once -> still catch all of them
  5. A real, isolated single-strike move -> HONEST LIMITATION: this
     gets flagged the same as a bad quote (IV alone can't tell them
     apart). This test doesn't "pass" in the sense of avoiding the
     flag — it documents that the flag fires, so you know exactly
     what the guard can and can't distinguish.
  6. Expiry-day convexity (a real spot move very close to expiry,
     which can swing raw OPTION PRICES enormously) -> must NOT be
     mistaken for bad data, because the guard checks IV, not price,
     and a real move driven by unchanged volatility doesn't move
     solved IV at all.

Usage:
    python test_iv_outlier_guard.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from core.iv_outlier_guard import IVOutlierGuard

STRIKES = list(range(23000, 24500, 100))


def make_chain(iv_map: dict) -> pd.DataFrame:
    rows = []
    for k in STRIKES:
        for opt in ("CE", "PE"):
            rows.append({"strike": k, "option_type": opt, "oi": 100000,
                         "iv": iv_map[k], "gamma": 0.001, "delta": 0.5})
    return pd.DataFrame(rows)


def base_iv_curve() -> dict:
    return {k: 0.18 + 0.00001 * (k - 23700) for k in STRIKES}


def test_1_normal_market():
    print("\n=== Test 1: normal market (no history yet) -> nothing flagged ===")
    guard = IVOutlierGuard(mad_multiplier=6.0)
    _, flagged = guard.filter(make_chain(base_iv_curve()))
    print(f"flagged: {len(flagged)} (expect 0, first cycle has no prior data to compare against)")
    assert len(flagged) == 0
    print("PASSED")


def test_2_genuine_broad_move():
    print("\n=== Test 2: genuine, uniform broad IV move across the whole chain -> keep it ===")
    guard = IVOutlierGuard(mad_multiplier=6.0)
    base = base_iv_curve()
    guard.filter(make_chain(base))
    moved = {k: v + 0.03 for k, v in base.items()}  # every strike shifts by the same real amount
    _, flagged = guard.filter(make_chain(moved))
    print(f"flagged: {len(flagged)} (expect 0)")
    assert len(flagged) == 0
    print("PASSED")


def test_3_one_bad_quote():
    print("\n=== Test 3: one strike gets a wild, implausible stale-quote-style spike ===")
    guard = IVOutlierGuard(mad_multiplier=6.0)
    base = base_iv_curve()
    guard.filter(make_chain(base))
    bad = dict(base)
    bad[23800] = 0.45  # was ~0.15, jumps to 0.45 -- nothing else moves
    _, flagged = guard.filter(make_chain(bad))
    flagged_strikes = sorted(set(flagged["strike"]))
    print(f"flagged strikes: {flagged_strikes} (expect exactly [23800])")
    assert flagged_strikes == [23800]
    print("PASSED")


def test_4_several_bad_quotes():
    print("\n=== Test 4: several strikes (5 of 15) corrupted simultaneously ===")
    guard = IVOutlierGuard(mad_multiplier=6.0)
    base = base_iv_curve()
    guard.filter(make_chain(base))
    bad = dict(base)
    bad_strikes = [23200, 23400, 23800, 24100, 24300]
    for k in bad_strikes:
        bad[k] = 0.60
    _, flagged = guard.filter(make_chain(bad))
    flagged_strikes = sorted(set(flagged["strike"]))
    print(f"injected bad: {bad_strikes}")
    print(f"flagged:      {flagged_strikes}")
    assert set(bad_strikes).issubset(set(flagged_strikes))
    print("PASSED — caught all simultaneously-bad strikes")


def test_5_real_isolated_move_honest_limitation():
    print("\n=== Test 5: a REAL, isolated, large move at one strike (not a bad quote) ===")
    guard = IVOutlierGuard(mad_multiplier=6.0)
    base = base_iv_curve()
    guard.filter(make_chain(base))
    real_move = dict(base)
    real_move[23800] = 0.30  # a genuine, large, isolated repricing
    _, flagged = guard.filter(make_chain(real_move))
    flagged_strikes = sorted(set(flagged["strike"]))
    print(f"flagged: {flagged_strikes}")
    print("This IS expected to flag [23800] — documenting the guard's honest, irreducible")
    print("limitation: IV alone cannot distinguish a real isolated move from a bad quote.")
    assert flagged_strikes == [23800]
    print("DOCUMENTED (not a failure — this is the known, acknowledged tradeoff)")


def _black76_price(F, K, T, iv, r=0.065, opt="C"):
    if T <= 0 or iv <= 0:
        return max(0.0, F - K) if opt == "C" else max(0.0, K - F)
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    discount = np.exp(-r * T)
    if opt == "C":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _solve_iv(F, K, T, price, opt, r=0.065):
    try:
        return brentq(lambda iv: _black76_price(F, K, T, iv, r, opt) - price, 1e-4, 5.0)
    except (ValueError, RuntimeError):
        return None


def test_6_expiry_day_convexity():
    print("\n=== Test 6: a REAL spot move, 1 day to expiry -> must NOT look like bad data ===")
    strikes = list(range(23600, 24500, 100))
    t_years = 1 / 365
    true_iv = 0.18
    r = 0.065
    f1, f2 = 23900.0, 23935.0  # a real, modest 35-point move

    def build(forward):
        rows = []
        for k in strikes:
            for opt_char, opt_label in (("C", "CE"), ("P", "PE")):
                price = _black76_price(forward, k, t_years, true_iv, r, opt_char)
                iv = _solve_iv(forward, k, t_years, price, opt_char, r)
                rows.append({"strike": k, "option_type": opt_label, "oi": 100000,
                             "iv": iv, "gamma": 0.001, "delta": 0.5})
        return pd.DataFrame(rows)

    guard = IVOutlierGuard(mad_multiplier=6.0)
    guard.filter(build(f1))
    _, flagged = guard.filter(build(f2))
    print(f"spot moved {f1} -> {f2}, 1 day to expiry, TRUE IV held constant at {true_iv}")
    print(f"flagged: {len(flagged)} (expect 0 — a real move with unchanged true vol")
    print("produces unchanged SOLVED IV too, since the guard checks IV, not raw price)")
    assert len(flagged) == 0
    print("PASSED")


if __name__ == "__main__":
    test_1_normal_market()
    test_2_genuine_broad_move()
    test_3_one_bad_quote()
    test_4_several_bad_quotes()
    test_5_real_isolated_move_honest_limitation()
    test_6_expiry_day_convexity()
    print("\nAll 6 scenarios ran. Tests 1-4 and 6 confirm correct behavior.")
    print("Test 5 documents the guard's one honest, acknowledged limitation.")
