"""
Conditional volatility forecasting with GJR-GARCH(1,1) and EGARCH(1,1).

BIC remains a parsimonious in-sample model-selection criterion. It is
not represented as proof of forecast superiority: production research
must also compare one-step forecasts out of sample using QLIKE/MAE/RMSE.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

from utils.stats_utils import select_best_by_information_criterion, TRADING_DAYS_PER_YEAR


class VolatilityForecastEngine:
    def __init__(self):
        self.fitted_model = None
        self.model_name = None
        self.bic_scores = None
        self.fit_diagnostics = None
        self.last_fit_returns_len = None

    def fit(self, daily_log_returns: pd.Series, min_obs: int = 250):
        r = pd.to_numeric(daily_log_returns, errors="coerce").dropna().astype(float)
        if not np.isfinite(r.to_numpy()).all() or (r.abs() > 1).any():
            raise ValueError("Returns must be finite decimal log returns with absolute value <= 1.")
        if len(r) < min_obs:
            raise ValueError(f"Need >= {min_obs} daily returns; received {len(r)}.")
        scaled = r * 100.0
        specifications = {
            "gjr_garch": arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"),
            "egarch": arch_model(scaled, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t"),
        }
        fits, failures = {}, {}
        for name, model in specifications.items():
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = model.fit(disp="off", show_warning=False)
                fits[name] = result
                if caught:
                    failures[name] = [str(w.message) for w in caught]
            except Exception as exc:
                failures[name] = [f"{type(exc).__name__}: {exc}"]
        best_name, scores = select_best_by_information_criterion(fits, "bic")
        self.fitted_model = fits[best_name]
        self.model_name = best_name
        self.bic_scores = scores
        self.fit_diagnostics = {
            "candidate_bic": scores,
            "candidate_warnings_or_failures": failures,
            "selected_convergence_flag": int(getattr(self.fitted_model, "convergence_flag", -1)),
            "selected_optimizer_message": str(getattr(getattr(self.fitted_model, "optimization_result", None), "message", "")),
            "selected_log_likelihood": float(self.fitted_model.loglikelihood),
            "selected_aic": float(self.fitted_model.aic),
            "selected_bic": float(self.fitted_model.bic),
            "selection_note": "BIC is in-sample parsimony; validate forecast quality out of sample.",
        }
        self.last_fit_returns_len = int(len(r))
        return self

    def forecast_next_session_vol(self, annualize: bool = True) -> dict:
        if self.fitted_model is None:
            raise RuntimeError("Call fit() before forecasting.")
        fc = self.fitted_model.forecast(horizon=1, reindex=False)
        variance_pct2 = float(fc.variance.values[-1, 0])
        if not np.isfinite(variance_pct2) or variance_pct2 <= 0:
            raise RuntimeError(f"Invalid GARCH variance forecast: {variance_pct2}")
        daily_vol = float(np.sqrt(variance_pct2) / 100.0)
        annualized = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else None
        return {
            "model_used": self.model_name, "bic_scores": self.bic_scores,
            "fit_diagnostics": self.fit_diagnostics, "daily_vol": daily_vol,
            "annualized_vol": float(annualized) if annualized is not None else None,
            "fitted_on_n_obs": self.last_fit_returns_len,
            "leverage_param": self._extract_leverage_param(),
        }

    def _extract_leverage_param(self):
        if self.fitted_model is None:
            return None
        params = self.fitted_model.params
        for key in params.index:
            if key.lower() in {"gamma[1]", "gamma[1]", "o[1]"}:
                return float(params[key])
        return None

    def volatility_risk_premium(self, implied_vol_annualized: float) -> dict:
        implied = float(implied_vol_annualized)
        fc = self.forecast_next_session_vol(annualize=True)
        forecast_vol = fc["annualized_vol"]
        if not np.isfinite(implied) or implied <= 0 or forecast_vol is None or forecast_vol <= 0:
            raise ValueError("Implied and forecast volatility must be finite and positive.")
        return {
            "implied_vol": implied, "forecast_vol": forecast_vol,
            "vrp_ratio": float((implied - forecast_vol) / forecast_vol),
            "model_used": fc["model_used"],
        }
