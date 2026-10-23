"""
test_pipeline.py
------------------
Validates the statistical pipeline end-to-end using SYNTHETIC data
(no Kotak API needed). This proves the math runs and produces sane
output before you ever point it at live/real data. Run:

    python test_pipeline.py
"""

import numpy as np
import pandas as pd

from core.volatility_engine import VolatilityForecastEngine
from core.trend_engine import TrendEngine
from core.regime_classifier import RegimeClassifier
from core.signal_engine import OptionsSignalEngine
from core.gamma_exposure import GammaExposureEngine
from core.options_math import (black76_price, black_scholes_delta, black_scholes_gamma,
                                black76_theta, black76_vega, implied_volatility)
from utils.stats_utils import log_returns

np.random.seed(42)

# --- synthetic daily price series with a volatility cluster + drift -------
n = 500
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
n = len(dates)  # pandas versions can differ by one at the "today" boundary —
                 # derive n from the actual date range rather than assuming
shocks = np.random.standard_t(df=5, size=n) * 0.01
# inject a vol cluster in the back half + mild upward drift
shocks[300:350] *= 3.0
drift = 0.0004
prices = 20000 * np.exp(np.cumsum(drift + shocks))
price_series = pd.Series(prices, index=dates)

daily_returns = log_returns(price_series)

print("=== 1. Volatility Engine ===")
vol_engine = VolatilityForecastEngine()
vol_engine.fit(daily_returns)
forecast = vol_engine.forecast_next_session_vol()
print(forecast)
assert forecast["annualized_vol"] > 0
assert forecast["model_used"] in ("gjr_garch", "egarch")

print("\n=== 2. VRP ===")
vrp = vol_engine.volatility_risk_premium(implied_vol_annualized=forecast["annualized_vol"] * 1.2)
print(vrp)
assert abs(vrp["vrp_ratio"] - 0.2) < 1e-6

print("\n=== 3. Trend Engine (higher timeframe proxy: last 60 sessions) ===")
trend_engine = TrendEngine()
trend_result = trend_engine.analyze(price_series.tail(60))
print(trend_result)

print("\n=== 4. Regime Classifier ===")
# simulate a trailing forecast history so terciles are computable
synthetic_history = pd.Series(np.random.uniform(0.10, 0.30, 40))
regime_classifier = RegimeClassifier()
regime = regime_classifier.classify(forecast["annualized_vol"], synthetic_history, trend_result)
print(regime)

print("\n=== 5. Gamma Exposure Engine (starting only from strike/OI/LTP, forward-anchored) ===")
spot = float(price_series.iloc[-1])
forward = spot * 1.0015  # a plausible small cost-of-carry premium over spot, for the test only —
                          # in main.py this comes from a REAL futures LTP, never assumed like this
strikes = np.arange(spot - 1000, spot + 1000, 50)
rng = np.random.default_rng(1)
t_years = 7 / 365.0
r_rate = 0.065
rows = []
for k in strikes:
    for opt_type in ("call", "put"):
        moneyness = abs(k - forward)
        oi = max(rng.integers(500, 5000) - int(moneyness), 100)  # more OI near ATM
        true_iv = 0.18 + 0.00005 * moneyness  # "true" market IV, unknown to us in real life
        ltp = black76_price(forward, k, t_years, true_iv, r_rate, opt_type)  # this is
        # the only thing Kotak actually gives us: a real traded price. Everything else
        # below is RECOVERED from it, exactly like main.py's live_loop() does.
        rows.append({"strike": k, "option_type": opt_type, "oi": oi, "ltp": ltp})
raw_chain = pd.DataFrame(rows)

# --- recover IV from the real LTP, anchored on the real futures forward ---
raw_chain["iv"] = raw_chain.apply(
    lambda row: implied_volatility(forward, row["strike"], t_years, row["ltp"], r_rate, row["option_type"]),
    axis=1)
n_dropped = raw_chain["iv"].isna().sum()
raw_chain = raw_chain.dropna(subset=["iv"]).copy()
print(f"IV solved for {len(raw_chain)}/{len(rows)} strikes ({n_dropped} dropped as unsolvable)")

raw_chain["delta"] = raw_chain.apply(
    lambda row: black_scholes_delta(forward, row["strike"], t_years, row["iv"], r_rate, row["option_type"]),
    axis=1)
raw_chain["gamma"] = raw_chain.apply(
    lambda row: black_scholes_gamma(forward, row["strike"], t_years, row["iv"], r_rate),
    axis=1)
raw_chain["theta_per_day"] = raw_chain.apply(
    lambda row: black76_theta(forward, row["strike"], t_years, row["iv"], r_rate, row["option_type"]) / 365.0,
    axis=1)
raw_chain["vega_per_1pct"] = raw_chain.apply(
    lambda row: black76_vega(forward, row["strike"], t_years, row["iv"], r_rate) / 100.0,
    axis=1)

# sanity check: recovered IV should match the true IV we generated the price from
recovered_vs_true_error = (raw_chain["iv"] - (0.18 + 0.00005 * (raw_chain["strike"] - forward).abs())).abs().max()
print(f"max |recovered_iv - true_iv| = {recovered_vs_true_error:.6f} (should be ~0, confirms Black-76 solver correctness)")
assert recovered_vs_true_error < 1e-4
print(f"sample theta/day: {raw_chain['theta_per_day'].iloc[0]:.3f}  "
      f"sample vega/1%iv: {raw_chain['vega_per_1pct'].iloc[0]:.3f}")

gamma_engine = GammaExposureEngine(contract_multiplier=25)
gamma_context = gamma_engine.compute(raw_chain, spot, t_years=t_years, risk_free_rate=r_rate)   # GEX scaling convention uses spot, not forward
print({k: v for k, v in gamma_context.items() if k != "by_strike_gex"})
assert gamma_context["dealer_regime"] in ("positive_gamma", "negative_gamma")

print("\n=== 6. Signal Engine (with gamma overlay) ===")
signal_engine = OptionsSignalEngine()
signal = signal_engine.generate(regime, vrp, forecast, spot=spot, gamma_context=gamma_context)
print(signal)

print("\n=== 7. Expiry auto-resolution (current vs next, rollover-safe) ===")
import datetime as _dt
from main import resolve_option_expiry
import config as _config

class _FakeFeed:
    """Stub feed for testing resolve_option_expiry()'s logic in isolation —
    no real API needed, this is pure date logic."""
    def get_available_expiries(self, symbol, exchange_segment):
        # deliberately out of chronological order and in DD-MMM-YYYY style,
        # to prove the date-aware sort (not alphabetical) works
        return ["25-Sep-2026", "04-Sep-2026", "11-Sep-2026", "02-Oct-2026"]

_config.OPTION_EXPIRY_OVERRIDE = None
_config.EXPIRY_DATE_FORMAT = "%d-%b-%Y"

# pretend "now" is 04-Sep-2026 10:00 — before that day's own market close,
# so 04-Sep-2026 should still count as "current"
_fixed_now = _dt.datetime(2026, 9, 4, 10, 0, 0)
import utils.dates as _dates_mod
_orig_now_default = _dates_mod.time_to_expiry_years

def _patched(expiry_str, expiry_format=None, market_close_time=_dt.time(15, 30), now=None):
    return _orig_now_default(expiry_str, expiry_format, market_close_time, now=_fixed_now)
_dates_mod.time_to_expiry_years = _patched
import main as _main_mod
_main_mod.time_to_expiry_years = _patched

_config.EXPIRY_SELECTION = "current"
current_expiry = resolve_option_expiry(_FakeFeed())
print(f"'current' resolved to: {current_expiry}")
assert current_expiry == "04-Sep-2026", f"expected 04-Sep-2026, got {current_expiry}"

_config.EXPIRY_SELECTION = "next"
next_expiry = resolve_option_expiry(_FakeFeed())
print(f"'next' resolved to: {next_expiry}")
assert next_expiry == "11-Sep-2026", f"expected 11-Sep-2026, got {next_expiry}"

# pretend "now" is 04-Sep-2026 16:00 — AFTER that day's market close, so
# 04-Sep-2026 should have rolled off and 11-Sep-2026 becomes "current"
_fixed_now = _dt.datetime(2026, 9, 4, 16, 0, 0)
_config.EXPIRY_SELECTION = "current"
rolled_expiry = resolve_option_expiry(_FakeFeed())
print(f"post-close rollover -> 'current' resolved to: {rolled_expiry}")
assert rolled_expiry == "11-Sep-2026", f"expected rollover to 11-Sep-2026, got {rolled_expiry}"

print("Expiry auto-resolution behaves correctly (chronological sort, "
      "same-day-before-close still current, rolls forward after close).")

print("\nAll pipeline stages executed without error.")
