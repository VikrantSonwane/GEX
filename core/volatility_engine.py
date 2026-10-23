"""
volatility_engine.py
---------------------
Fits GJR-GARCH(1,1) and EGARCH(1,1) on daily log returns, lets BIC decide
which one better describes THIS instrument's volatility dynamics (some
names show strong leverage/asymmetry, some don't — we don't assume it),
and produces a forward volatility forecast used by every downstream
decision (regime, stop-loss sizing, VRP).

Both models use a Student-t innovation distribution. That is a
well-established, standard choice for equity/index returns (fat tails
are a documented empirical fact, not a discretionary trading parameter)
— so it's kept fixed rather than "selected," to avoid overfitting model
choice on top of distribution choice with the same small sample.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model

from utils.stats_utils import select_best_by_information_criterion, TRADING_DAYS_PER_YEAR


class VolatilityForecastEngine:

    def __init__(self):
        self.fitted_model = None
        self.model_name = None
        self.bic_scores = None
        self.last_fit_returns_len = None

    def fit(self, daily_log_returns: pd.Series, min_obs: int = 250):
        """
        daily_log_returns: log returns of DAILY closes (%, not decimal —
        arch_model wants percentage returns for numerical stability, so
        we scale by 100 internally).

        Requires at least `min_obs` daily observations (~1 trading year)
        so the (E)GARCH likelihood has enough data to identify persistence
        and asymmetry parameters reliably. Below that, refuses to fit
        rather than returning an unstable/garbage forecast.
        """
        r = daily_log_returns.dropna()
        if len(r) < min_obs:
            raise ValueError(
                f"Need >= {min_obs} daily returns to fit reliably, got {len(r)}. "
                f"Feeding a GARCH model too little history produces unstable, "
                f"overfit parameters — that's exactly the 'artificial value' "
                f"this engine is built to avoid."
            )

        scaled = r * 100.0

        gjr = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist="t")
        egarch = arch_model(scaled, vol="EGARCH", p=1, o=1, q=1, dist="t")

        gjr_fit = gjr.fit(disp="off")
        egarch_fit = egarch.fit(disp="off")

        fits = {"gjr_garch": gjr_fit, "egarch": egarch_fit}
        best_name, scores = select_best_by_information_criterion(fits, criterion="bic")

        self.fitted_model = fits[best_name]
        self.model_name = best_name
        self.bic_scores = scores
        self.last_fit_returns_len = len(r)
        return self

    def forecast_next_session_vol(self, annualize: bool = True) -> dict:
        """
        One-step-ahead conditional volatility forecast (i.e. tomorrow's /
        next session's expected volatility), converted back out of the
        x100 scaling used for fitting.
        """
        if self.fitted_model is None:
            raise RuntimeError("Call .fit() before forecasting.")

        fc = self.fitted_model.forecast(horizon=1, reindex=False)
        variance_pct2 = fc.variance.values[-1, 0]        # variance in (pct-return)^2 units
        daily_vol = np.sqrt(variance_pct2) / 100.0        # back to decimal daily vol

        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else None

        return {
            "model_used": self.model_name,
            "bic_scores": self.bic_scores,
            "daily_vol": float(daily_vol),
            "annualized_vol": float(annualized_vol) if annualized_vol is not None else None,
            "fitted_on_n_obs": self.last_fit_returns_len,
            "leverage_param": self._extract_leverage_param(),
        }

    def _extract_leverage_param(self):
        """
        Surfaces the fitted asymmetry/leverage coefficient (gamma for
        GJR-GARCH, the 'o' term for EGARCH) so you can see WHY a model
        was chosen, not just take the forecast on faith.
        """
        params = self.fitted_model.params
        for key in params.index:
            if key.lower() in ("gamma[1]", "o1"):
                return float(params[key])
        return None

    def volatility_risk_premium(self, implied_vol_annualized: float) -> dict:
        """
        VRP = implied vol (market's option-priced expectation) vs the
        GARCH-forecasted realized vol (statistical expectation).

        VRP > 0  -> options priced "expensive" relative to the model's
                    expectation of realized volatility (favors premium
                    selling, all else equal).
        VRP < 0  -> options priced "cheap" relative to model expectation
                    (favors premium buying / long gamma).

        This is a diagnostic ratio, not a signal by itself — the
        SignalEngine combines it with trend/regime before suggesting
        anything.
        """
        fc = self.forecast_next_session_vol(annualize=True)
        forecast_vol = fc["annualized_vol"]
        vrp_ratio = (implied_vol_annualized - forecast_vol) / forecast_vol
        return {
            "implied_vol": implied_vol_annualized,
            "forecast_vol": forecast_vol,
            "vrp_ratio": float(vrp_ratio),
            "model_used": fc["model_used"],
        }
