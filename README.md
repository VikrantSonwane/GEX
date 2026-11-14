# GEX — Quantitative Intraday Options Research Engine

A quantitative, **long-options-only**, manual-execution decision-support system for NSE index and stock options. It produces a structured research signal; it does not place, modify, or cancel orders.

The design goal is to preserve explicit mathematics, expose assumptions, and fail closed when required market data or contract specifications are unavailable. A signal is not a claim of profitability.

## Research pipeline

1. **Historical data and validation** — CSV by default, with optional Yahoo Finance and jugaad-data sources.
2. **Volatility** — fit GJR-GARCH(1,1) and EGARCH(1,1) with Student-t innovations; reject non-converged fits and select among valid candidates by BIC.
3. **Trend** — autocorrelation-adjusted Hamed-Rao Mann-Kendall test on log-price levels.
4. **Volatility regime** — compare the current forecast with the distribution of strictly prior session forecasts.
5. **Option valuation** — Black-76 prices and Greeks using a futures forward, plus implied-volatility inversion from broker option prices.
6. **VRP** — compare ATM implied volatility with a GARCH/EGARCH forecast over approximately the same number of trading sessions to expiry.
7. **GEX context** — compute OI-weighted gamma scenarios, local GEX, strike concentrations and a modelled gamma-flip level.
8. **Signal** — return LONG_CALL, LONG_PUT or NO_SIGNAL. No short-premium recommendation is emitted.
9. **Dashboard and event logs** — preserve inputs and outputs for later research; the dashboard is observational, not an execution terminal.

## Mathematical choices and their limits

### Volatility model

Daily log returns are converted to percentage units for fitting. GJR-GARCH and EGARCH candidates use Student-t innovations. BIC is an **in-sample parsimony criterion**, not evidence that a model forecasts better. Fit convergence, optimizer details, likelihood, AIC and BIC are surfaced in diagnostics.

For multi-session VRP, the selected model's forward variance term structure is forecast over the approximate remaining trading-session horizon. Multi-step forecasts use a reproducible 10,000-path simulation so EGARCH can be handled as well as GJR-GARCH. Out-of-sample forecast evaluation should use QLIKE, MAE and RMSE; helper functions and CI tests are included, but no historical forecast score is represented as a proven trading edge.

### Trend and regime

The trend test is run on log-price levels because the question is monotonic price direction over the window. Hamed-Rao modifies the Mann-Kendall variance for serial correlation. The standard 5% test level is a statistical convention, not a tuned entry threshold. A non-significant trend produces NO_SIGNAL.

Volatility labels use the instrument's own historical forecast terciles. Today's forecast is excluded from the historical reference sample, and repeated same-day restarts do not append multiple observations for the same session. At least 20 prior forecasts are required; before that the classifier returns insufficient_history.

### Black-76 options valuation

The system uses the observed futures price as the forward anchor rather than adding an assumed dividend yield to spot. Prices, Greeks and IV inversion are validated against finite-difference and put-call-parity identities.

- Delta and gamma are with respect to the futures forward.
- Vega is per 1.00 (100 percentage-point) change in decimal IV; divide by 100 for per-1%-IV.
- Theta is calendar-time theta per year, i.e. negative derivative with respect to time-to-expiry.
- The time-to-expiry input follows the expiry date's exchange close and a calendar-year day count; volatility annualization uses 252 trading days.

The weekly-option forward proxy can still differ from the true weekly implied forward when only monthly futures are listed. That basis risk remains an explicit approximation.

### Gamma exposure is a scenario, not dealer inventory

The current GEX scenario uses the sign convention dealer_long_calls_short_puts:

- Call contribution: positive gamma × call OI.
- Put contribution: negative gamma × put OI.
- Both are scaled by spot², 1% and the live contract lot size.

Open interest does not identify who is long or short. The returned object therefore labels the assumed positioning convention and sets dealer_positions_observed = false. Positive/negative GEX, the gamma flip, call wall and put wall are conditional on this assumption and should not be read as verified dealer books. The flip holds current strike IVs fixed while sweeping hypothetical spot; it does not dynamically reshape the volatility smile.

### Long-only signal policy

The engine emits only LONG_CALL, LONG_PUT or NO_SIGNAL. A positive VRP is treated as a premium-cost diagnostic; it does not trigger a credit spread, straddle sale or other short-premium structure. The current 0.40 target-delta convention and 1x stop / 2x target underlying-volatility multiples are operational starting conventions, not statistically estimated parameters.

The stop/target fields are **underlying-point moves**, not option-premium stop and target prices. Position sizing, portfolio-level Kelly sizing, broker execution, transaction-cost modeling and a validated options backtest are separate workstreams and are not claimed as implemented here.

## Data adapters and Kotak Neo

The repository now tracks Python source under data/; .gitignore excludes local datasets and credentials, not the source package.

- data/kotak_client.py normalizes Kotak Neo expiries, option_chain, search_scrip and quotes responses. It is read-only and contains no order endpoint calls.
- data/historical_source.py provides validated CSV loading, daily resampling, optional Yahoo Finance data and optional jugaad-data NSE index/stock history.
- config.py defaults to the user's own CSV history to preserve data provenance.

The old project notes claiming that Kotak had no historical-candle endpoint were based on an older SDK. The current Kotak Neo SDK documentation includes historical_data(neosymbol, interval, from_date, to_date) with broker-enforced lookback limits. This endpoint is **not yet wired into the default historical loader**; use CSV/Yahoo/jugaad as configured until that adapter path is explicitly added and tested. The option-chain adapter in this repository does use the newer SDK interface.

Install the current SDK version required by this project:

    pip install -r requirements.txt

Create .env from .env.example, then fill in credentials privately. TOTP is time-sensitive and must be current at login. Never commit .env.

## Setup

    python -m venv .venv

Windows PowerShell:

    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Linux/macOS:

    source .venv/bin/activate
    pip install -r requirements.txt

Check local historical configuration in config.py:

- HISTORICAL_DATA_SOURCE = "csv" by default.
- DAILY_HISTORY_CSV_PATH must exist.
- DAILY_HISTORY_CSV_DATE_COLUMN and DAILY_HISTORY_CSV_CLOSE_COLUMN must match the CSV.
- At least 250 daily returns are required by the volatility model.

Run offline tests:

    pytest -q tests test_iv_outlier_guard.py

Run the synthetic pipeline smoke test:

    python run_pipeline_smoke.py

Run the manual live decision-support engine only after validating the current market-data responses and credentials:

    python main.py

The live dashboard is local-only and opens from the main program. No order is sent.

## CI

GitHub Actions runs offline unit tests on Python 3.11 and 3.12. The tests mock broker responses and do not access credentials or call the live broker. They cover Black-76 identities/Greeks, IV inversion, trend inference, volatility-regime classification, forecast metrics, long-only signal behavior, GEX assumptions, and market-data adapter normalization.

## Current known limitations before live reliance

1. The Kotak adapter has mocked tests only; a real authenticated read-only smoke check is still required for the user's account, exact symbols and current scrip-master response.
2. Kotak's historical_data endpoint is not yet integrated into the historical-source dispatcher.
3. The GEX dealer-side sign convention is unobservable from OI alone.
4. The monthly-futures proxy for weekly options can create forward-basis error.
5. Trend significance and BIC are not a substitute for walk-forward testing. The next research milestone is a timestamp-safe options backtester with bid/ask fills, brokerage, slippage, contract expiry settlement and per-trade path attribution.
6. No performance claim is made until historical options quotes and realistic execution costs have been validated.

## Source map

| Path | Responsibility |
|---|---|
| core/options_math.py | Black-76 price, Greeks and IV solver |
| core/volatility_engine.py | GJR-GARCH / EGARCH fitting and expiry-horizon forecast |
| core/trend_engine.py | Hamed-Rao trend inference |
| core/regime_classifier.py | Distribution-derived volatility regime |
| core/signal_engine.py | Long-only signal qualification |
| core/gamma_exposure.py | OI-based GEX scenario, walls and flip |
| core/iv_skew.py | 25-delta put/call skew |
| core/iv_outlier_guard.py | Cycle-over-cycle IV quote anomaly guard |
| data/kotak_client.py | Read-only Kotak Neo market-data adapter |
| data/historical_source.py | CSV, Yahoo and NSE history loaders |
| utils/stats_utils.py | Statistical tests, distributions and forecast losses |
| main.py | Session orchestration and live polling |
| dashboard_server.py | Local observational dashboard |
| tests/ | Offline unit and mocked adapter tests |
| run_pipeline_smoke.py | Synthetic end-to-end smoke test |
