"""
Black-76 futures-option valuation and Greeks.

Inputs and outputs are in price points; volatility is decimal annualized
volatility; time is years using the caller's documented day-count
convention. Greeks are derivatives with respect to the forward.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _validate_inputs(forward: float, strike: float, t_years: float,
                     iv: float, r: float) -> None:
    vals = np.asarray([forward, strike, t_years, iv, r], dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Black-76 inputs must all be finite.")
    if forward <= 0 or strike <= 0:
        raise ValueError("Forward and strike must be strictly positive.")
    if t_years < 0:
        raise ValueError("Time to expiry cannot be negative.")
    if iv < 0:
        raise ValueError("Implied volatility cannot be negative.")


def _option_kind(option_type: str) -> str:
    kind = str(option_type).strip().lower()
    if kind in {"c", "ce", "call"}:
        return "call"
    if kind in {"p", "pe", "put"}:
        return "put"
    raise ValueError("option_type must be call/CE or put/PE.")


def black76_price(forward: float, strike: float, t_years: float,
                  iv: float, r: float, option_type: str) -> float:
    _validate_inputs(forward, strike, t_years, iv, r)
    kind = _option_kind(option_type)
    if t_years == 0:
        return float(max(forward - strike, 0.0) if kind == "call"
                     else max(strike - forward, 0.0))
    discount = float(np.exp(-r * t_years))
    if iv == 0:
        intrinsic = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        return float(discount * intrinsic)
    root_t = np.sqrt(t_years)
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * root_t)
    d2 = d1 - iv * root_t
    if kind == "call":
        value = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        value = discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return float(value)


def black76_delta(forward: float, strike: float, t_years: float,
                  iv: float, r: float, option_type: str) -> float:
    _validate_inputs(forward, strike, t_years, iv, r)
    kind = _option_kind(option_type)
    if t_years == 0 or iv == 0:
        if kind == "call":
            return 1.0 if forward > strike else (0.0 if forward < strike else 0.5)
        return -1.0 if forward < strike else (0.0 if forward > strike else -0.5)
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    discount = np.exp(-r * t_years)
    if kind == "call":
        return float(discount * norm.cdf(d1))
    return float(discount * (norm.cdf(d1) - 1.0))


def black76_gamma(forward: float, strike: float, t_years: float,
                  iv: float, r: float) -> float:
    _validate_inputs(forward, strike, t_years, iv, r)
    if t_years == 0 or iv == 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    return float(np.exp(-r * t_years) * norm.pdf(d1) /
                 (forward * iv * np.sqrt(t_years)))


def black76_theta(forward: float, strike: float, t_years: float,
                  iv: float, r: float, option_type: str) -> float:
    """
    Calendar-time theta per year: -dV/dT for time-to-expiry T.

    With Black-76's forward held fixed, this is identical for calls and
    puts at the same strike/IV: r*option_price - discounted diffusion
    term. Finite-difference checks are included in the test suite.
    """
    _validate_inputs(forward, strike, t_years, iv, r)
    kind = _option_kind(option_type)
    if t_years == 0 or iv == 0:
        return 0.0
    root_t = np.sqrt(t_years)
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * root_t)
    discount = np.exp(-r * t_years)
    diffusion = discount * forward * norm.pdf(d1) * iv / (2.0 * root_t)
    option_price = black76_price(forward, strike, t_years, iv, r, kind)
    return float(r * option_price - diffusion)


def black76_vega(forward: float, strike: float, t_years: float,
                 iv: float, r: float) -> float:
    """Vega per 1.00 (100 percentage points) change in decimal IV."""
    _validate_inputs(forward, strike, t_years, iv, r)
    if t_years == 0 or iv == 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    return float(np.exp(-r * t_years) * forward * norm.pdf(d1) * np.sqrt(t_years))


def implied_volatility(forward: float, strike: float, t_years: float,
                       market_price: float, r: float, option_type: str,
                       iv_bounds: tuple[float, float] = (1e-4, 5.0)) -> float | None:
    """Invert Black-76 with no extrapolation beyond theoretical price bounds."""
    kind = _option_kind(option_type)
    vals = np.asarray([forward, strike, t_years, market_price, r, *iv_bounds], dtype=float)
    if not np.isfinite(vals).all() or forward <= 0 or strike <= 0 or t_years <= 0 or market_price <= 0:
        return None
    lo, hi = map(float, iv_bounds)
    if lo <= 0 or hi <= lo:
        raise ValueError("iv_bounds must satisfy 0 < lower < upper.")
    discount = float(np.exp(-r * t_years))
    intrinsic = (max(forward - strike, 0.0) if kind == "call"
                 else max(strike - forward, 0.0)) * discount
    upper_bound = (forward if kind == "call" else strike) * discount
    tolerance = max(1e-10, upper_bound * 1e-10)
    if market_price < intrinsic - tolerance or market_price > upper_bound + tolerance:
        return None

    def error(vol):
        return black76_price(forward, strike, t_years, vol, r, kind) - market_price

    try:
        e_lo, e_hi = error(lo), error(hi)
        if e_lo == 0:
            return lo
        if e_hi == 0:
            return hi
        if e_lo * e_hi > 0:
            return None
        return float(brentq(error, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError, FloatingPointError):
        return None


# Backward-compatible aliases: callers supply a forward price, not spot.
black_scholes_delta = black76_delta
black_scholes_gamma = black76_gamma
