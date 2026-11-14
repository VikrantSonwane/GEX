"""
Statistical utilities shared by the research and signal layers.

The trend test is applied to log-price levels because the research
question is whether price has a monotonic path over the observation
window. Hamed-Rao variance correction adjusts for serial correlation;
it does not replace walk-forward validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk

TRADING_MINUTES_PER_DAY = 375
TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Validated log returns from strictly positive prices."""
    values = pd.to_numeric(prices, errors="coerce").astype(float)
    if (values.dropna() <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")
    return np.log(values / values.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def realized_volatility(intraday_returns: pd.Series, bars_per_day: int | None = None,
                        annualize: bool = True) -> float:
    """Close-to-close realized volatility from one clean intraday session."""
    values = pd.to_numeric(intraday_returns, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("At least one valid intraday return is required.")
    rv = float(np.sqrt(np.square(values).sum()))
    return rv * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else rv


def trend_test(closes: pd.Series, alpha: float = 0.05,
               method: str = "hamed_rao") -> dict:
    """
    Modified Mann-Kendall test on log prices.

    Log transformation preserves price ordering while reducing scale
    effects. Hamed-Rao modifies the test variance for serial correlation,
    a material concern for price-level time series. 'original' is exposed
    only for diagnostics/comparison, not as the default.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    prices = pd.to_numeric(closes, errors="coerce").dropna().astype(float)
    if (prices <= 0).any():
        raise ValueError("Trend prices must be strictly positive.")
    log_prices = np.log(prices.to_numpy(dtype=float))
    if len(log_prices) < 8:
        return {"direction": "insufficient_data", "p_value": None,
                "tau": None, "slope": None, "significant": False,
                "observations": int(len(log_prices)), "method": method,
                "input": "log_prices"}
    if not np.isfinite(log_prices).all():
        raise ValueError("Trend input contains non-finite prices.")
    if method == "hamed_rao":
        result = mk.hamed_rao_modification_test(log_prices, alpha=alpha)
    elif method == "original":
        result = mk.original_test(log_prices, alpha=alpha)
    else:
        raise ValueError("method must be 'hamed_rao' or 'original'")
    direction = "up" if result.trend == "increasing" else (
        "down" if result.trend == "decreasing" else "no_trend")
    if not result.h:
        direction = "no_trend"
    return {"direction": direction, "p_value": float(result.p),
            "tau": float(result.Tau), "slope": float(result.slope),
            "significant": bool(result.h), "observations": int(len(log_prices)),
            "method": method, "input": "log_prices"}


def distributional_regime(current_value: float, history: pd.Series,
                          min_history: int = 20) -> dict:
    """Classify current forecast against strictly prior forecast history."""
    value = float(current_value)
    hist = pd.to_numeric(history, errors="coerce").dropna()
    hist = hist[np.isfinite(hist.to_numpy())]
    if not np.isfinite(value) or value < 0:
        raise ValueError("current_value must be finite and non-negative.")
    if len(hist) < min_history:
        return {"regime": "insufficient_history", "p33": None, "p66": None,
                "percentile_rank": None, "history_observations": int(len(hist))}
    p33, p66 = np.percentile(hist.to_numpy(), [33, 66])
    regime = "low" if value <= p33 else ("high" if value >= p66 else "normal")
    rank = float((hist.to_numpy() < value).mean() * 100.0)
    return {"regime": regime, "p33": float(p33), "p66": float(p66),
            "percentile_rank": rank, "history_observations": int(len(hist))}


def select_best_by_information_criterion(fits: dict, criterion: str = "bic"):
    """Select only converged, finite-criterion fits; fail closed if none qualify."""
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")
    valid = {}
    for name, fit in fits.items():
        converged = getattr(getattr(fit, "optimization_result", None), "success", True)
        score = getattr(fit, criterion, np.nan)
        if converged and np.isfinite(score):
            valid[name] = float(score)
    if not valid:
        raise RuntimeError("No candidate volatility model converged with a finite information criterion.")
    best = min(valid, key=valid.get)
    return best, valid


def forecast_error_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """Out-of-sample volatility forecast metrics on aligned positive variance proxies."""
    paired = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).dropna()
    paired = paired[(paired["actual"] > 0) & (paired["forecast"] > 0)]
    if paired.empty:
        raise ValueError("No aligned positive actual/forecast observations.")
    y = paired["actual"].to_numpy(dtype=float)
    f = paired["forecast"].to_numpy(dtype=float)
    ratio = y / f
    qlike = np.mean(ratio - np.log(ratio) - 1.0)
    return {"n": int(len(paired)), "mae": float(np.mean(np.abs(y - f))),
            "rmse": float(np.sqrt(np.mean(np.square(y - f)))),
            "qlike": float(qlike), "mean_error": float(np.mean(f - y))}
