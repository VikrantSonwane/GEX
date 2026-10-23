"""
stats_utils.py
----------------
Pure statistical helper functions. Nothing in this file uses a fixed,
hand-picked magic number for a trading threshold. Every threshold is
either:
  (a) derived from the data's own distribution (percentiles/terciles), or
  (b) the output of a significance test (p-value), or
  (c) chosen by an information criterion (AIC/BIC) over a candidate grid.

This is the discipline the whole engine follows: config.py holds
*operational* choices (which symbol, which timeframe) — this file and
the engines never hard-code a "if vol > 20 -> high regime" style rule.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pymannkendall as mk


TRADING_MINUTES_PER_DAY = 375       # NSE cash/derivatives session length
TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns from a price series. Base unit for every model here."""
    r = np.log(prices / prices.shift(1))
    return r.dropna()


def realized_volatility(intraday_returns: pd.Series, bars_per_day: int,
                         annualize: bool = True) -> float:
    """
    Realized volatility estimator (sum of squared intraday log returns),
    the standard model-free volatility measure used to check a GARCH
    forecast against what *actually* happened.

    intraday_returns: log returns sampled at a fixed intraday interval,
                       for a SINGLE session.
    bars_per_day:     number of bars that make up one session at that
                       sampling interval (e.g. 75 for 5-min bars on NSE).
    """
    rv = np.sqrt(np.sum(intraday_returns.values ** 2))
    if annualize:
        rv *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(rv)


def trend_test(closes: pd.Series, alpha: float = 0.05) -> dict:
    """
    Mann-Kendall trend test on a higher-timeframe close series.

    This replaces arbitrary moving-average-crossover trend detection
    with a nonparametric significance test: it tells you whether the
    observed monotonic drift is statistically distinguishable from
    noise, and at what confidence, rather than reacting to any two
    MAs crossing.

    Returns direction ('up' / 'down' / 'no_trend'), the p-value, and
    Kendall's tau (strength of the trend, -1..1).
    """
    result = mk.original_test(closes.values, alpha=alpha)
    if result.trend == "increasing":
        direction = "up"
    elif result.trend == "decreasing":
        direction = "down"
    else:
        direction = "no_trend"

    return {
        "direction": direction,
        "p_value": float(result.p),
        "tau": float(result.Tau),
        "slope": float(result.slope),
        "significant": bool(result.h),
    }


def distributional_regime(current_value: float, history: pd.Series) -> dict:
    """
    Classify `current_value` (e.g. today's GARCH-forecasted vol) against
    the TERCILES of its own trailing history, instead of a hard-coded
    threshold like "vol > 18% = high". The 33rd/66th percentile cut
    points are recomputed from the data every time this runs, so the
    definition of "high vol" adapts as the market's own regime shifts.

    Requires at least 20 historical observations to be meaningful;
    below that it returns 'insufficient_history' rather than guessing.
    """
    hist = history.dropna()
    if len(hist) < 20:
        return {"regime": "insufficient_history", "p33": None, "p66": None,
                "percentile_rank": None}

    p33, p66 = np.percentile(hist.values, [33, 66])
    if current_value <= p33:
        regime = "low"
    elif current_value >= p66:
        regime = "high"
    else:
        regime = "normal"

    percentile_rank = float((hist.values < current_value).mean() * 100)

    return {"regime": regime, "p33": float(p33), "p66": float(p66),
            "percentile_rank": percentile_rank}


def select_best_by_information_criterion(fits: dict, criterion: str = "bic"):
    """
    Given {model_name: fitted_result} where each fitted_result exposes
    .aic and .bic (as arch's ARCHModelResult does), pick the model that
    minimizes the requested criterion. This is how the engine chooses
    between GJR-GARCH and EGARCH — never a fixed preference for one.
    """
    scores = {name: getattr(fit, criterion) for name, fit in fits.items()}
    best_name = min(scores, key=scores.get)
    return best_name, scores
