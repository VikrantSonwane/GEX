"""
gamma_exposure.py
--------------------
Dealer gamma exposure (GEX) and net delta positioning, built from the
option chain's open interest. This is a PROXY, not observed truth: it
assumes the standard convention used by SpotGamma / SqueezeMetrics-style
GEX dashboards — that dealers are net long the calls customers sell them
and net short the puts customers buy from them, i.e. dealer gamma tracks
+call_OI and -put_OI. That assumption is usually reasonable in aggregate
for liquid index chains (NIFTY/BANKNIFTY) but can be wrong for any single
strike or session — treat the output as a probabilistic tilt, not a fact.

What it gives you:
  - Net GEX (total): sign tells you the dominant hedging regime.
      positive -> dealers hedge AGAINST the move -> dampened vol, mean-reversion
      negative -> dealers hedge WITH the move -> amplified vol, trend acceleration
  - Gamma flip point: the HYPOTHETICAL SPOT LEVEL at which total dealer
      GEX itself would cross zero, holding today's OI and each strike's
      already-solved IV fixed. This is the standard "zero gamma level"
      definition used by SpotGamma-style tools — NOT the strike where a
      running sum across strikes happens to cross zero (an earlier
      version of this file computed that instead; the two are genuinely
      different quantities and were verified to diverge on synthetic
      data before this fix — see _find_gamma_flip()'s docstring).
  - Call wall / put wall: strikes with the largest positive call-side and
      put-side gamma*OI — common magnets / support-resistance in OI-based
      analysis (the same idea NSE option-chain traders already use with
      max-OI strikes, just gamma-weighted instead of OI-weighted).
  - Net delta exposure: aggregate dealer delta, a proxy for which
      direction dealer hedging flow leans.

Nothing here is a hard threshold either — "gamma regime" is read directly
off the sign of GEX and the spot's position relative to the flip point,
not an arbitrary cutoff.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from core.options_math import black_scholes_gamma


class GammaExposureEngine:

    def __init__(self, contract_multiplier: int):
        """
        contract_multiplier: lot size of the underlying's option contract
        (e.g. NIFTY's lot size — check the CURRENT NSE-specified lot size
        at run time, since NSE revises these periodically; don't hard-code
        last year's value into config and forget it).
        """
        self.contract_multiplier = contract_multiplier

    def compute(self, chain_df: pd.DataFrame, spot: float,
                t_years: float | None = None, risk_free_rate: float = 0.065,
                flip_search_range_pct: float = 0.08, flip_search_points: int = 200,
                local_band_pct: float = 0.01, expected_daily_move: float | None = None) -> dict:
        """
        chain_df expects columns: ['strike', 'option_type', 'oi', 'gamma',
        'delta', 'iv']. 'iv' is required for the gamma-flip search below
        (t_years/risk_free_rate also required for it) — total_gex/walls/
        delta exposure at the ACTUAL current spot don't need them and
        still work if you omit t_years (flip_point just comes back None).

        Per-strike dealer GEX AT THE CURRENT SPOT, standard convention:
            call_GEX(strike) = +gamma * call_OI * spot^2 * 0.01 * multiplier
            put_GEX(strike)  = -gamma * put_OI  * spot^2 * 0.01 * multiplier
        (the spot^2 * 0.01 term converts gamma into a dollar-exposure-per-
        1%-move figure, the standard GEX scaling convention.) This part
        answers "what is dealer gamma exposure RIGHT NOW" — unaffected by
        the flip-point fix below, which answers a different question
        ("at what OTHER spot level would that exposure flip sign").

        local_band_pct: also computes "local GEX" — total GEX restricted
        to strikes within +/-local_band_pct of spot (default 1%). A huge
        OI position thousands of points away can distort the TOTAL GEX
        number without affecting today's actual tape near spot; local_gex
        is the more relevant number for "can price move right now."

        expected_daily_move: pass next_session_vol['daily_vol'] * spot
        (i.e. the GARCH model's forecasted 1-day point move) to also get
        distance-to-flip/walls expressed as a MULTIPLE of that expected
        move, not just raw points — "0.3x expected move away" is a more
        comparable number across different vol regimes than "140 points"
        is by itself. Omit it and those fields just come back None.
        """
        df = chain_df.copy()
        is_call = df["option_type"].str.lower().str.startswith("c")
        sign = np.where(is_call, 1.0, -1.0)

        df["strike_gex"] = sign * df["gamma"] * df["oi"] * (spot ** 2) * 0.01 * self.contract_multiplier
        df["strike_delta_exposure"] = df["delta"] * df["oi"] * self.contract_multiplier

        by_strike = df.groupby("strike")["strike_gex"].sum().sort_index()
        total_gex = float(by_strike.sum())
        net_delta_exposure = float(df["strike_delta_exposure"].sum())

        local_lo, local_hi = spot * (1 - local_band_pct), spot * (1 + local_band_pct)
        local_gex = float(by_strike[(by_strike.index >= local_lo) & (by_strike.index <= local_hi)].sum())

        gex_slope_near_spot = self._gex_slope_near_spot(by_strike, spot)
        max_pain = self._compute_max_pain(df, is_call)

        flip_point = None
        if t_years is not None and "iv" in df.columns:
            flip_point = self._find_gamma_flip(df, sign, spot, t_years, risk_free_rate,
                                                 flip_search_range_pct, flip_search_points)

        call_gex_by_strike = df[is_call].groupby("strike")["strike_gex"].sum()
        put_gex_by_strike = df[~is_call].groupby("strike")["strike_gex"].sum()
        call_wall = float(call_gex_by_strike.idxmax()) if not call_gex_by_strike.empty else None
        put_wall = float(put_gex_by_strike.idxmin()) if not put_gex_by_strike.empty else None

        if total_gex > 0:
            dealer_regime = "positive_gamma"
            regime_note = "Dealers net long gamma: hedging flow tends to dampen moves (mean-reversion, pinning near high-OI strikes more likely)."
        else:
            dealer_regime = "negative_gamma"
            regime_note = "Dealers net short gamma: hedging flow tends to amplify moves (trend acceleration, sharper breakouts more likely)."

        spot_vs_flip = None
        if flip_point is not None:
            spot_vs_flip = "above_flip" if spot > flip_point else "below_flip"

        def _dist(level):
            return (spot - level) if level is not None else None

        def _dist_in_move_units(level):
            if level is None or expected_daily_move in (None, 0):
                return None
            return (spot - level) / expected_daily_move

        return {
            "total_gex": total_gex,
            "local_gex": local_gex,
            "gex_slope_near_spot": gex_slope_near_spot,
            "max_pain": max_pain,
            "dealer_regime": dealer_regime,
            "regime_note": regime_note,
            "gamma_flip_point": flip_point,
            "spot_vs_flip": spot_vs_flip,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "distance_to_flip": _dist(flip_point),
            "distance_to_call_wall": _dist(call_wall),
            "distance_to_put_wall": _dist(put_wall),
            "distance_to_flip_in_expected_moves": _dist_in_move_units(flip_point),
            "distance_to_call_wall_in_expected_moves": _dist_in_move_units(call_wall),
            "distance_to_put_wall_in_expected_moves": _dist_in_move_units(put_wall),
            "net_delta_exposure": net_delta_exposure,
            "by_strike_gex": by_strike,
            "row_level_detail": df[["strike", "option_type", "oi", "gamma", "delta", "strike_gex"] +
                                     (["iv"] if "iv" in df.columns else [])].copy(),
        }

    @staticmethod
    def _gex_slope_near_spot(by_strike: pd.Series, spot: float):
        """
        Rate of change of GEX per point, between the listed strikes
        immediately below and above spot. A steep slope means a small
        move in either direction crosses a lot of gamma structure (a
        "wall" close by); a flat slope means spot is in open space
        with little nearby structure to react to. Returns None if
        there's no strike on one side (spot sits outside the chain).

        Uses STRICT < / > (not <=/>=) — if spot lands exactly on a
        listed strike, that strike would otherwise satisfy both "below"
        and "above" simultaneously, making k_lo == k_hi and returning
        None when a real slope was actually computable using the
        genuine neighboring strikes on each side.
        """
        strikes = by_strike.index.values
        below = strikes[strikes < spot]
        above = strikes[strikes > spot]
        if len(below) == 0 or len(above) == 0:
            return None
        k_lo, k_hi = below.max(), above.min()
        return float((by_strike[k_hi] - by_strike[k_lo]) / (k_hi - k_lo))

    @staticmethod
    def _compute_max_pain(df: pd.DataFrame, is_call: np.ndarray):
        """
        Max pain: the strike that, if NIFTY settled there today, would
        cost option WRITERS the least total payout across all open
        contracts (equivalently, gives option HOLDERS the least total
        payoff — "maximum pain" for holders). Standard definition:
        for each candidate settlement price S (every listed strike),
        total_payout(S) = sum_over_calls(max(S-K,0)*OI) +
                          sum_over_puts(max(K-S,0)*OI)
        Max pain = the S that minimizes total_payout(S).

        This is a DIFFERENT quantity from gamma flip or the walls —
        it's a pure OI-based magnet candidate, not gamma-weighted at
        all, included as a complementary, well-known reference point,
        not a replacement for the gamma-based levels.
        """
        strikes = df["strike"].unique()
        call_rows = df[is_call]
        put_rows = df[~is_call]

        payouts = []
        for S in strikes:
            call_payout = np.sum(np.maximum(S - call_rows["strike"].values, 0) * call_rows["oi"].values)
            put_payout = np.sum(np.maximum(put_rows["strike"].values - S, 0) * put_rows["oi"].values)
            payouts.append(call_payout + put_payout)

        if not payouts:
            return None
        return float(strikes[int(np.argmin(payouts))])

    @staticmethod
    def _find_gamma_flip(df: pd.DataFrame, sign: np.ndarray, spot: float, t_years: float,
                          risk_free_rate: float, search_range_pct: float, n_points: int):
        """
        The REAL gamma flip: sweeps a range of HYPOTHETICAL spot levels
        (default +/-8% around today's actual spot), and at each one,
        recomputes every strike's gamma AS IF the underlying were at
        that hypothetical level — using each strike's own already-solved
        IV held fixed (the standard simplification: this ignores how the
        vol smile itself would shift if spot actually moved there, which
        professional GEX tools also typically ignore for this calculation).
        Sums signed GEX across all strikes at each hypothetical spot to
        build a "total GEX vs hypothetical spot" curve, then finds where
        THAT curve crosses zero.

        This is NOT the same as summing GEX across strikes at today's
        ACTUAL spot and looking for a strike-order running-sum crossing
        (what an earlier version of this file did) — verified directly
        on synthetic data that the two methods diverge, sometimes with
        the strike-order method producing multiple spurious crossings
        where the spot-sweep method gives one clean pivot.
        """
        strikes = df["strike"].values
        oi = df["oi"].values
        iv = df["iv"].values
        valid = ~np.isnan(iv) & (iv > 0) & (oi > 0)
        if valid.sum() == 0:
            return None
        strikes, oi, iv, sign = strikes[valid], oi[valid], iv[valid], sign[valid]

        hyp_spots = np.linspace(spot * (1 - search_range_pct), spot * (1 + search_range_pct), n_points)

        # Vectorized Black-76 gamma across (hypothetical spot x strike) grid:
        # shape (n_points, n_strikes) via broadcasting.
        S = hyp_spots[:, None]          # (n_points, 1)
        K = strikes[None, :]            # (1, n_strikes)
        sigma = iv[None, :]
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * t_years) / (sigma * np.sqrt(t_years))
        discount = np.exp(-risk_free_rate * t_years)
        gamma_grid = discount * norm.pdf(d1) / (S * sigma * np.sqrt(t_years))

        total_gex_curve = np.sum(gamma_grid * (sign * oi)[None, :] * (S ** 2) * 0.01, axis=1)

        sign_changes = np.where(np.diff(np.sign(total_gex_curve)) != 0)[0]
        if len(sign_changes) == 0:
            return None

        idx = sign_changes[np.argmin(np.abs(hyp_spots[sign_changes] - spot))]
        x0, x1 = hyp_spots[idx], hyp_spots[idx + 1]
        y0, y1 = total_gex_curve[idx], total_gex_curve[idx + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
