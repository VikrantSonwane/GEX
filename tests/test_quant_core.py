import numpy as np
import pandas as pd
import pytest

from core.options_math import (
    black76_price, black76_delta, black76_gamma, black76_theta,
    black76_vega, implied_volatility,
)
from core.signal_engine import OptionsSignalEngine
from core.gamma_exposure import GammaExposureEngine
from utils.stats_utils import (
    trend_test, distributional_regime, select_best_by_information_criterion,
    forecast_error_metrics,
)


def test_black76_put_call_parity():
    forward, strike, t, iv, rate = 22500.0, 22400.0, 21 / 365, 0.19, 0.065
    call = black76_price(forward, strike, t, iv, rate, "call")
    put = black76_price(forward, strike, t, iv, rate, "put")
    expected = np.exp(-rate * t) * (forward - strike)
    assert call - put == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_black76_iv_round_trip(kind):
    forward, strike, t, iv, rate = 22500.0, 22600.0, 14 / 365, 0.235, 0.065
    premium = black76_price(forward, strike, t, iv, rate, kind)
    recovered = implied_volatility(forward, strike, t, premium, rate, kind)
    assert recovered == pytest.approx(iv, abs=1e-7)


def test_black76_greeks_match_finite_differences():
    forward, strike, t, iv, rate = 22500.0, 22500.0, 18 / 365, 0.21, 0.065
    eps_f, eps_t, eps_iv = 0.1, 1e-6, 1e-5
    price_up = black76_price(forward + eps_f, strike, t, iv, rate, "call")
    price_dn = black76_price(forward - eps_f, strike, t, iv, rate, "call")
    numeric_delta = (price_up - price_dn) / (2 * eps_f)
    assert black76_delta(forward, strike, t, iv, rate, "call") == pytest.approx(numeric_delta, rel=1e-5)

    delta_up = black76_delta(forward + eps_f, strike, t, iv, rate, "call")
    delta_dn = black76_delta(forward - eps_f, strike, t, iv, rate, "call")
    numeric_gamma = (delta_up - delta_dn) / (2 * eps_f)
    assert black76_gamma(forward, strike, t, iv, rate) == pytest.approx(numeric_gamma, rel=2e-4)

    price_now = black76_price(forward, strike, t, iv, rate, "call")
    price_less_t = black76_price(forward, strike, t - eps_t, iv, rate, "call")
    numeric_theta = (price_less_t - price_now) / eps_t
    assert black76_theta(forward, strike, t, iv, rate, "call") == pytest.approx(numeric_theta, rel=2e-4)

    price_more_iv = black76_price(forward, strike, t, iv + eps_iv, rate, "call")
    numeric_vega = (price_more_iv - price_now) / eps_iv
    assert black76_vega(forward, strike, t, iv, rate) == pytest.approx(numeric_vega, rel=2e-4)


def test_black76_rejects_negative_time_and_invalid_option_kind():
    with pytest.raises(ValueError):
        black76_price(100, 100, -0.01, 0.2, 0.05, "call")
    with pytest.raises(ValueError):
        black76_price(100, 100, 0.1, 0.2, 0.05, "straddle")


def test_iv_rejects_prices_outside_theoretical_bounds():
    assert implied_volatility(100, 100, 0.1, 200, 0.05, "call") is None
    assert implied_volatility(100, 100, 0.1, -1, 0.05, "put") is None


def test_hamed_rao_trend_runs_on_log_price_levels():
    closes = pd.Series(100 * np.exp(np.linspace(0, 0.15, 80)))
    result = trend_test(closes, method="hamed_rao")
    assert result["direction"] == "up"
    assert result["significant"] is True
    assert result["input"] == "log_prices"
    assert result["method"] == "hamed_rao"


def test_regime_uses_supplied_prior_distribution_and_fails_closed():
    insufficient = distributional_regime(0.2, pd.Series([0.1, 0.2]))
    assert insufficient["regime"] == "insufficient_history"
    hist = pd.Series(np.linspace(0.1, 0.3, 30))
    assert distributional_regime(0.05, hist)["regime"] == "low"
    assert distributional_regime(0.5, hist)["regime"] == "high"


def test_information_criterion_ignores_nonconverged_candidates():
    class Optimizer:
        def __init__(self, success):
            self.success = success

    class Fit:
        def __init__(self, bic, success=True):
            self.bic = bic
            self.optimization_result = Optimizer(success)

    name, scores = select_best_by_information_criterion({
        "bad": Fit(1.0, success=False),
        "valid": Fit(3.0, success=True),
    })
    assert name == "valid"
    assert scores == {"valid": 3.0}


def test_forecast_error_metrics_are_aligned_and_finite():
    actual = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2026-01-01", periods=3))
    forecast = pd.Series([0.012, 0.018, 0.029], index=pd.date_range("2026-01-01", periods=3))
    metrics = forecast_error_metrics(actual, forecast)
    assert metrics["n"] == 3
    assert metrics["mae"] > 0
    assert np.isfinite(metrics["qlike"])


def test_signal_engine_never_recommends_short_premium():
    engine = OptionsSignalEngine()
    base = {
        "trend": {"direction": "up", "significant": True, "tau": 0.25, "p_value": 0.01},
        "volatility": {"regime": "low"},
    }
    signal = engine.generate(base, {"vrp_ratio": 0.80},
                             {"daily_vol": 0.01}, spot=22000)
    assert signal.strategy == "LONG_CALL"
    assert signal.direction == "bullish"

    base["trend"] = {"direction": "down", "significant": True, "tau": -0.3, "p_value": 0.01}
    signal = engine.generate(base, {"vrp_ratio": -0.2},
                             {"daily_vol": 0.01}, spot=22000)
    assert signal.strategy == "LONG_PUT"

    base["trend"] = {"direction": "no_trend", "significant": False, "tau": 0.0, "p_value": 0.8}
    signal = engine.generate(base, {"vrp_ratio": -0.2},
                             {"daily_vol": 0.01}, spot=22000)
    assert signal.strategy == "NO_SIGNAL"


def test_gex_is_explicitly_an_assumption_not_observed_dealer_inventory():
    chain = pd.DataFrame([
        {"strike": 100.0, "option_type": "CE", "oi": 1000, "gamma": 0.02, "delta": 0.5},
        {"strike": 100.0, "option_type": "PE", "oi": 1000, "gamma": 0.02, "delta": -0.5},
    ])
    result = GammaExposureEngine(contract_multiplier=10).compute(chain, spot=100.0)
    assert result["total_gex"] == pytest.approx(0.0)
    assert result["dealer_regime"] == "neutral_gamma"
    assert result["positioning_assumption"] == "dealer_long_calls_short_puts"
    assert result["dealer_positions_observed"] is False
    assert result["open_interest_unit_assumption"] == "contracts"
    assert result["contract_multiplier_applied"] == 10
    assert result["net_delta_exposure"] == pytest.approx(10000.0)


def test_gex_requires_verified_contract_multiplier():
    chain = pd.DataFrame([
        {"strike": 100.0, "option_type": "CE", "oi": 1000, "gamma": 0.02, "delta": 0.5},
    ])
    with pytest.raises(ValueError):
        GammaExposureEngine(contract_multiplier=None).compute(chain, spot=100.0)
