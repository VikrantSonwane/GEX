"""
options_math.py
------------------
Black-76 (forward-based) option math — used instead of spot-based
Black-Scholes because NIFTY futures already embed the market's own
view of cost-of-carry (interest minus dividends). Pricing/IV off the
live futures LTP means we no longer need to ASSUME a dividend yield or
risk-free drift for the underlying — the forward price is a real,
live, exchange-traded number pulled straight from Kotak, not a guess.

The only assumed input left is a short-term discount rate `r`, used
purely to discount the option's future settlement value back to today
(a small, low-controversy adjustment — nowhere near as consequential
as guessing a dividend yield was). Use the current T-bill/repo rate
and update it periodically; it is NOT a drift assumption on the
underlying anymore.

Formulas (Black, 1976):
    d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = e^(-rT) * [F*N(d1) - K*N(d2)]
    Put  = e^(-rT) * [K*N(-d2) - F*N(-d1)]

All functions take `forward` (F) instead of `spot`.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black76_price(forward: float, strike: float, t_years: float,
                   iv: float, r: float, option_type: str) -> float:
    if t_years <= 0 or iv <= 0:
        intrinsic = max(0.0, forward - strike) if option_type.lower().startswith("c") else max(0.0, strike - forward)
        return intrinsic * np.exp(-r * t_years)

    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    d2 = d1 - iv * np.sqrt(t_years)
    discount = np.exp(-r * t_years)

    if option_type.lower().startswith("c"):
        return float(discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))
    return float(discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1)))


def black76_delta(forward: float, strike: float, t_years: float,
                   iv: float, r: float, option_type: str) -> float:
    """
    Delta with respect to the FORWARD price (the standard convention for
    futures-style options). Numerically very close to spot-delta for
    short-dated index options, and the more correct quantity to use
    here since we're anchored on the forward, not spot.
    """
    if t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    discount = np.exp(-r * t_years)
    if option_type.lower().startswith("c"):
        return float(discount * norm.cdf(d1))
    return float(discount * (norm.cdf(d1) - 1))


def black76_gamma(forward: float, strike: float, t_years: float,
                   iv: float, r: float) -> float:
    """Gamma is identical for calls and puts at the same strike (BS/Black-76 identity)."""
    if t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    discount = np.exp(-r * t_years)
    return float(discount * norm.pdf(d1) / (forward * iv * np.sqrt(t_years)))


def black76_theta(forward: float, strike: float, t_years: float,
                   iv: float, r: float, option_type: str) -> float:
    """
    Theta per YEAR (divide by 365 for a per-day figure, which is the
    more intuitive unit for a manual-execution trader checking decay
    risk before entering — main.py's live_loop() does that division
    when printing).
    """
    if t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    d2 = d1 - iv * np.sqrt(t_years)
    discount = np.exp(-r * t_years)
    term1 = -(forward * discount * norm.pdf(d1) * iv) / (2 * np.sqrt(t_years))

    if option_type.lower().startswith("c"):
        term2 = r * discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        return float(term1 - term2)
    term2 = r * discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return float(term1 + term2)


def black76_vega(forward: float, strike: float, t_years: float,
                  iv: float, r: float) -> float:
    """Vega per 1.00 (100 percentage points) change in IV — divide by 100
    for the conventional "per 1% IV move" quoting convention."""
    if t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    discount = np.exp(-r * t_years)
    return float(discount * forward * norm.pdf(d1) * np.sqrt(t_years))


def implied_volatility(forward: float, strike: float, t_years: float,
                        market_price: float, r: float, option_type: str,
                        iv_bounds: tuple[float, float] = (1e-4, 5.0)) -> float | None:
    """
    Backs out the IV that reprices the option to its real, exchange-quoted
    LTP, anchored on the real futures forward price rather than spot.
    Uses brentq (bisection-family root finder), robust and derivative-free.

    Returns None if no solution exists in iv_bounds — this legitimately
    happens for deep ITM/OTM or stale/illiquid LTPs. Callers should drop
    that row rather than force a number (see main.py's live_loop()).
    """
    if t_years <= 0 or market_price <= 0:
        return None

    intrinsic = (max(0.0, forward - strike) if option_type.lower().startswith("c")
                 else max(0.0, strike - forward)) * np.exp(-r * t_years)
    if market_price < intrinsic:
        return None  # price below discounted intrinsic value -> stale/bad quote

    def error(iv):
        return black76_price(forward, strike, t_years, iv, r, option_type) - market_price

    try:
        lo, hi = iv_bounds
        if error(lo) * error(hi) > 0:
            return None  # no sign change -> no root in range, don't force a fit
        return float(brentq(error, lo, hi, xtol=1e-6))
    except (ValueError, RuntimeError):
        return None


# --- backward-compatible names (spot-based callers now pass forward instead) ---
# Kept so signal_engine.py / gamma_exposure.py don't need renamed imports;
# they now receive the futures forward price wherever they used to receive spot.
black_scholes_delta = black76_delta
black_scholes_gamma = black76_gamma
