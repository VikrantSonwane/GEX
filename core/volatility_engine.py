"""
Conditional volatility forecasting with GJR-GARCH(1,1) and EGARCH(1,1).

BIC is an in-sample parsimony criterion, not proof of better forecasts.
Every candidate fit must converge; forecast quality should be compared
out of sample using QLIKE/MAE/RMSE.
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
        if not np.isfinite(r.to_numpy()).all():
            raise ValueError("Returns must be finite decimal log returns.")
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
        opt = getattr(self.fitted_model, "optimization_result", None)
        self.fit_diagnostics = {
            "candidate_bic": scores,
            "candidate_warnings_or_failures": failures,
            "selected_convergence_flag": int(getattr(self.fitted_model, "convergence_flag", -1)),
            "selected_optimizer_message": str(getattr(opt, "message", "")),
            "selected_log_likelihood": float(self.fitted_model.loglikelihood),
            "selected_aic": float(self.fitted_model.aic),
            "selected_bic": float(self.fitted_model.bic),
            "selection_note": "BIC is in-sample parsimony; validate forecasts out of sample.",
        }
        self.last_fit_returns_len = int(len(r))
        return self

    def forecast_next_session_vol(self, annualize: bool = True) -> dict:
        forecast = self.forecast_horizon_vol(horizon_sessions=1, annualize=annualize)
        forecast["daily_vol"] = forecast["horizon_daily_vol"]
        forecast["annualized_vol"] = forecast["horizon_annualized_vol"]
        return forecast

    def forecast_horizon_vol(self, horizon_sessions: int,
                             annualize: bool = True) -> dict:
        """Forecast average annualized volatility over H future trading sessions."""
        if self.fitted_model is None:
            raise RuntimeError("Call fit() before forecasting.")
        if not isinstance(horizon_sessions, int) or horizon_sessions < 1:
            raise ValueError("horizon_sessions must be a positive integer.")
        if horizon_sessions == 1:
            fc = self.fitted_model.forecast(horizon=1, reindex=False, method="analytic")
        else:
            # EGARCH does not have a generally available analytic multi-step
            # variance recursion in arch; simulation supports both candidate
            # specifications. A fixed seed keeps research runs reproducible.
            rng = np.random.default_rng(20261114)
            fc = self.fitted_model.forecast(
                horizon=horizon_sessions,
                reindex=False,
                method="simulation",
                simulations=10000,
                rng=rng.standard_normal,
            )
        variances_pct2 = np.asarray(fc.variance.values[-1, :horizon_sessions], dtype=float)
        if variances_pct2.size != horizon_sessions or not np.isfinite(variances_pct2).all() or (variances_pct2 <= 0).any():
            raise RuntimeError("Invalid or incomplete GARCH variance term-structure forecast.")
        daily_variances = variances_pct2 / 10000.0
        mean_daily_variance = float(daily_variances.mean())
        horizon_daily_vol = float(np.sqrt(mean_daily_variance))
        annualized = horizon_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else None
        return {
            "model_used": self.model_name,
            "bic_scores": self.bic_scores,
            "fit_diagnostics": self.fit_diagnostics,
            "horizon_sessions": horizon_sessions,
            "horizon_daily_vol": horizon_daily_vol,
            "horizon_annualized_vol": float(annualized) if annualized is not None else None,
            "fitted_on_n_obs": self.last_fit_returns_len,
            "leverage_param": self._extract_leverage_param(),
        }

    def _extract_leverage_param(self):
        if self.fitted_model is None:
            return None
        params = self.fitted_model.params
        for key in params.index:
            if key.lower() in {"gamma[1]", "o[1]"}:
                return float(params[key])
        return None

    def volatility_risk_premium(self, implied_vol_annualized: float,
                                horizon_sessions: int = 1) -> dict:
        """
        Compare market IV with GARCH forecast over the same approximate
        expiry horizon. VRP is positive when implied volatility exceeds
        forecast realized volatility; it is a valuation diagnostic only.
        """
        implied = float(implied_vol_annualized)
        forecast = self.forecast_horizon_vol(horizon_sessions, annualize=True)
        forecast_vol = forecast["horizon_annualized_vol"]
        if not np.isfinite(implied) or implied <= 0 or forecast_vol is None or forecast_vol <= 0:
            raise ValueError("Implied and forecast volatility must be finite and positive.")
        return {
            "implied_vol": implied,
            "forecast_vol": forecast_vol,
            "vrp_ratio": float((implied - forecast_vol) / forecast_vol),
            "model_used": forecast["model_used"],
            "horizon_sessions": int(horizon_sessions),
            "comparison_note": "ATM IV and GARCH forecast use the same approximate trading-session horizon.",
        }
