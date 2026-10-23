"""
main.py
--------
Orchestrator. Flow, once per trading session:

  1. Log in to Kotak Neo, pull daily history -> fit GJR-GARCH/EGARCH,
     pick the better model by BIC -> forecast next-session volatility.
  2. Append that forecast to a persisted history file (this is what lets
     the regime classifier eventually judge "high vs low vol" from the
     instrument's OWN distribution instead of a guessed number).
  3. Trend read: by default, reuses those same daily closes over the
     last HIGHER_TF_LOOKBACK_SESSIONS -> Mann-Kendall trend test. (Set
     config.TREND_TIMEFRAME_SOURCE="intraday" only if you have your own
     genuine intraday historical data — see historical_source.py.)
  4. Every SIGNAL_POLL_SECONDS during the session: pull the live futures
     LTP (forward price anchor), the live option chain, solve IV/greeks
     off the real forward, compute VRP, classify regime, generate a
     signal, print it.

This script NEVER places an order — it is a decision-support feed for
manual execution, per your requirement.
"""

from __future__ import annotations
import time
import datetime as dt
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the SAME FOLDER as this script, regardless of what the
# current working directory happens to be (PyCharm/terminal CWD can
# differ from the script's own folder, which is the #1 reason a .env
# file that "should" be picked up silently isn't).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_env_loaded = load_dotenv(dotenv_path=_ENV_PATH, override=True)
if _env_loaded:
    print(f"[env] loaded .env from {_ENV_PATH}")
else:
    print(f"[env] WARNING: no .env file found at {_ENV_PATH} — "
          f"relying on already-set environment variables, if any. "
          f"If you meant to use a .env file, check it's saved EXACTLY "
          f"as '.env' (not '.env.txt') in this exact folder.")

import config
from data.kotak_client import KotakDataFeed
from data.historical_source import (get_daily_closes, get_intraday_closes, get_daily_closes_jugaad,
                                     get_daily_ohlc_jugaad, sync_ohlc_to_csv,
                                     load_closes_from_csv, resample_intraday_to_daily_closes)
from core.volatility_engine import VolatilityForecastEngine
from core.trend_engine import TrendEngine
from core.regime_classifier import RegimeClassifier
from core.signal_engine import OptionsSignalEngine, black_scholes_delta
from core.gamma_exposure import GammaExposureEngine, black_scholes_gamma
from core.realized_vol import RealizedVolTracker
from core.iv_skew import compute_skew
from core.iv_outlier_guard import IVOutlierGuard
from core.options_math import implied_volatility, black76_theta, black76_vega
from utils.stats_utils import log_returns
from utils.dates import time_to_expiry_years
from utils.daily_logger import log_snapshot, log_strike_breakdown
import dashboard_server


def load_forecast_history() -> pd.Series:
    if not os.path.exists(config.VOL_FORECAST_HISTORY_PATH):
        return pd.Series(dtype=float)
    df = pd.read_csv(config.VOL_FORECAST_HISTORY_PATH, parse_dates=["date"])
    return df.set_index("date")["annualized_vol"]


def append_forecast_history(annualized_vol: float, model_used: str):
    row = pd.DataFrame([{
        "date": dt.datetime.now().date(),
        "annualized_vol": annualized_vol,
        "model_used": model_used,
    }])
    header = not os.path.exists(config.VOL_FORECAST_HISTORY_PATH)
    row.to_csv(config.VOL_FORECAST_HISTORY_PATH, mode="a", header=header, index=False)


def resolve_option_expiry(feed: KotakDataFeed) -> str:
    """
    Auto-picks the expiry to trade, run fresh at the start of every
    session — this is what makes the engine survive expiry rollover
    without you editing config.py every week. Logic:

      1. If config.OPTION_EXPIRY_OVERRIDE is set, use it verbatim (manual pin).
      2. Otherwise, pull every live expiry from the scrip master, drop any
         that have already passed market close TODAY (so on expiry day
         itself, before close, today's own expiry still counts as
         "current" — only after close does it roll to next), sort
         chronologically, and pick index 0 ("current") or index 1 ("next")
         per config.EXPIRY_SELECTION.
    """
    if config.OPTION_EXPIRY_OVERRIDE:
        print(f"[expiry] using manual override: {config.OPTION_EXPIRY_OVERRIDE}")
        return config.OPTION_EXPIRY_OVERRIDE

    all_expiries = feed.get_available_expiries(config.UNDERLYING_SYMBOL, config.EXCHANGE_SEGMENT)
    live_expiries = [e for e in all_expiries
                      if time_to_expiry_years(e, expiry_format=config.EXPIRY_DATE_FORMAT) > 0]
    # sort chronologically by remaining time — don't trust the caller's
    # ordering (the real KotakDataFeed.get_available_expiries() already
    # sorts, but this function shouldn't silently depend on that; a test
    # feeding an out-of-order list is exactly what caught this before it
    # became a live bug).
    live_expiries.sort(key=lambda e: time_to_expiry_years(e, expiry_format=config.EXPIRY_DATE_FORMAT))

    if not live_expiries:
        raise RuntimeError(f"No live (non-expired) expiries found for "
                            f"{config.UNDERLYING_SYMBOL} — scrip master may be stale "
                            f"or the symbol/segment is wrong.")

    idx = {"current": 0, "next": 1}.get(config.EXPIRY_SELECTION)
    if idx is None:
        raise ValueError(f"config.EXPIRY_SELECTION must be 'current' or 'next', "
                          f"got '{config.EXPIRY_SELECTION}'")
    if idx >= len(live_expiries):
        raise RuntimeError(f"Requested '{config.EXPIRY_SELECTION}' expiry but only "
                            f"{len(live_expiries)} live expiry(ies) available: {live_expiries}")

    chosen = live_expiries[idx]
    days_out = time_to_expiry_years(chosen, expiry_format=config.EXPIRY_DATE_FORMAT) * 365
    print(f"[expiry] auto-selected '{config.EXPIRY_SELECTION}' expiry: {chosen} "
          f"(~{days_out:.1f} days out). Other live expiries seen: {live_expiries}")
    return chosen


def _load_daily_closes() -> pd.Series:
    """Dispatches to your own CSV, jugaad-data, or Yahoo Finance per
    config.HISTORICAL_DATA_SOURCE."""
    if config.HISTORICAL_DATA_SOURCE == "csv":
        if config.TREND_TIMEFRAME_SOURCE == "intraday" and config.DERIVE_DAILY_FROM_INTRADAY:
            intraday = load_closes_from_csv(config.INTRADAY_HISTORY_CSV_PATH,
                                             config.INTRADAY_HISTORY_CSV_DATE_COLUMN,
                                             config.INTRADAY_HISTORY_CSV_CLOSE_COLUMN)
            return resample_intraday_to_daily_closes(intraday)
        return load_closes_from_csv(config.DAILY_HISTORY_CSV_PATH,
                                     config.DAILY_HISTORY_CSV_DATE_COLUMN,
                                     config.DAILY_HISTORY_CSV_CLOSE_COLUMN,
                                     lookback_days=None)  # keep everything in the file —
                                                            # more history only helps the
                                                            # GARCH fit, no need to trim
    elif config.HISTORICAL_DATA_SOURCE == "jugaad":
        ohlc = get_daily_ohlc_jugaad(config.JUGAAD_INDEX_SYMBOL, config.DAILY_HISTORY_LOOKBACK_DAYS)
        if config.JUGAAD_SYNC_TO_CSV:
            added = sync_ohlc_to_csv(ohlc, config.DAILY_HISTORY_CSV_PATH, date_column="Date")
            print(f"[jugaad] synced {added} new day(s) into {config.DAILY_HISTORY_CSV_PATH}")
        closes = ohlc.set_index("Date")["Close"]
        closes.name = "close"
        return closes
    elif config.HISTORICAL_DATA_SOURCE == "yfinance":
        return get_daily_closes(config.HISTORICAL_DATA_TICKER, config.DAILY_HISTORY_LOOKBACK_DAYS)
    raise ValueError(f"config.HISTORICAL_DATA_SOURCE must be 'csv', 'jugaad', or "
                      f"'yfinance', got '{config.HISTORICAL_DATA_SOURCE}'")


def _load_higher_tf_closes(daily_closes: pd.Series) -> pd.Series:
    """
    Feeds the trend test. Default (config.TREND_TIMEFRAME_SOURCE ==
    "daily"): reuses the SAME daily closes already fetched for the vol
    model, over the last HIGHER_TF_LOOKBACK_SESSIONS — no separate
    historical intraday data needed (see historical_source.py's
    module docstring for why: reliable free intraday historical NIFTY
    data doesn't really exist publicly). Only fetches separately if
    you've explicitly opted into TREND_TIMEFRAME_SOURCE == "intraday"
    with your own genuine intraday historical data.
    """
    if config.TREND_TIMEFRAME_SOURCE == "daily":
        return daily_closes.tail(config.HIGHER_TF_LOOKBACK_SESSIONS)
    elif config.TREND_TIMEFRAME_SOURCE == "intraday":
        if config.HISTORICAL_DATA_SOURCE == "csv":
            return load_closes_from_csv(config.INTRADAY_HISTORY_CSV_PATH,
                                         config.INTRADAY_HISTORY_CSV_DATE_COLUMN,
                                         config.INTRADAY_HISTORY_CSV_CLOSE_COLUMN,
                                         lookback_days=config.HIGHER_TF_LOOKBACK_SESSIONS)
        elif config.HISTORICAL_DATA_SOURCE == "yfinance":
            return get_intraday_closes(config.HISTORICAL_DATA_TICKER,
                                        interval=config.HIGHER_TF_INTERVAL,
                                        lookback_days=config.HIGHER_TF_LOOKBACK_SESSIONS)
        raise ValueError(f"TREND_TIMEFRAME_SOURCE='intraday' needs "
                          f"HISTORICAL_DATA_SOURCE to be 'csv' or 'yfinance' "
                          f"(jugaad-data doesn't offer intraday), got "
                          f"'{config.HISTORICAL_DATA_SOURCE}'")
    raise ValueError(f"config.TREND_TIMEFRAME_SOURCE must be 'daily' or "
                      f"'intraday', got '{config.TREND_TIMEFRAME_SOURCE}'")


def session_start_fit(feed: KotakDataFeed):
    """
    Steps 1-2: fit the vol model on daily history, persist the forecast.

    Daily closes come from your own CSV, jugaad-data, or Yahoo Finance
    (see config.HISTORICAL_DATA_SOURCE / data/historical_source.py),
    NOT Kotak — Kotak Neo's Trade API has no historical-candle endpoint
    (confirmed by introspecting a live client's real method list: no
    'historical_data' or equivalent exists). `feed` isn't used here for
    that reason; it's still passed in for symmetry with the rest of
    main() and in case a future SDK version adds one.

    Returns the fitted engine, the forecast, AND the daily closes
    themselves — session_trend() reuses those closes by default rather
    than re-fetching, see _load_higher_tf_closes().
    """
    daily_closes = _load_daily_closes()
    daily_returns = log_returns(daily_closes)

    engine = VolatilityForecastEngine()
    engine.fit(daily_returns)
    forecast = engine.forecast_next_session_vol()

    append_forecast_history(forecast["annualized_vol"], forecast["model_used"])
    print(f"[fit] model={forecast['model_used']} "
          f"annualized_vol={forecast['annualized_vol']:.2%} "
          f"(fit on {forecast['fitted_on_n_obs']} sessions, "
          f"leverage_param={forecast['leverage_param']})")

    return engine, forecast, daily_closes


def get_contract_multiplier(feed: KotakDataFeed) -> int:
    """Reads lot size live from the scrip master; falls back to the
    config constant only if that lookup fails, and says so loudly since
    a fallback here means the live source didn't work as expected."""
    try:
        multiplier = feed.get_lot_size(config.UNDERLYING_SYMBOL, config.EXCHANGE_SEGMENT)
        print(f"[contract] lot size read live from scrip master: {multiplier}")
        return multiplier
    except Exception as e:
        print(f"[warn] live lot size lookup failed ({e}); falling back to "
              f"config.CONTRACT_MULTIPLIER_FALLBACK={config.CONTRACT_MULTIPLIER_FALLBACK}. "
              f"Verify this is still correct before trusting any GEX numbers.")
        return config.CONTRACT_MULTIPLIER_FALLBACK


def session_trend(feed: KotakDataFeed, daily_closes: pd.Series):
    """
    Step 3: higher-timeframe trend read.

    Default (config.TREND_TIMEFRAME_SOURCE == "daily"): reuses the
    SAME daily closes already fetched in session_start_fit() — no
    separate historical intraday data needed (see
    data/historical_source.py's module docstring for why). `feed` is
    kept as a parameter for symmetry/future use, unused here for that
    reason regardless of source.
    """
    higher_tf_closes = _load_higher_tf_closes(daily_closes)
    trend_engine = TrendEngine()
    trend_result = trend_engine.analyze(higher_tf_closes)
    print(f"[trend] direction={trend_result['direction']} "
          f"tau={trend_result.get('tau')} p={trend_result.get('p_value')}")
    return trend_result


def estimate_atm_iv(chain_df: pd.DataFrame, spot: float) -> float:
    """
    Picks the strike closest to spot and averages its call/put IV as a
    simple ATM IV proxy. If your chain response already tags an 'atm'
    flag, prefer that instead — this is a fallback, not a preference.
    """
    chain_df = chain_df.copy()
    chain_df["dist_to_spot"] = (chain_df["strike"] - spot).abs()
    atm_strike = chain_df.loc[chain_df["dist_to_spot"].idxmin(), "strike"]
    atm_rows = chain_df[chain_df["strike"] == atm_strike]
    return float(atm_rows["iv"].mean())


def live_loop(feed: KotakDataFeed, vol_engine: VolatilityForecastEngine,
              trend_result: dict, forecast_history: pd.Series, contract_multiplier: int,
              option_expiry: str):
    regime_classifier = RegimeClassifier()
    signal_engine = OptionsSignalEngine()
    gamma_engine = GammaExposureEngine(contract_multiplier=contract_multiplier)
    realized_vol_tracker = RealizedVolTracker()
    iv_guard = IVOutlierGuard()

    while True:
        try:
            # Re-check the resolved expiry hasn't rolled over (e.g. this
            # process has been left running across an expiry day). If it
            # has, re-resolve automatically instead of silently querying
            # an expired contract forever.
            t_years = time_to_expiry_years(option_expiry, expiry_format=config.EXPIRY_DATE_FORMAT)
            if t_years <= 0:
                print(f"  [expiry] {option_expiry} has passed market close — rolling to next.")
                option_expiry = resolve_option_expiry(feed)
                t_years = time_to_expiry_years(option_expiry, expiry_format=config.EXPIRY_DATE_FORMAT)

            quote = feed.get_live_quote(config.UNDERLYING_SYMBOL, config.EXCHANGE_SEGMENT)
            spot = float(quote["ltp"])
            realized_vol_tracker.update(dt.datetime.now(), spot)

            futures_info = feed.get_futures_ltp(config.UNDERLYING_SYMBOL, option_expiry)
            forward = futures_info["futures_ltp"]
            if not futures_info["is_exact_expiry_match"]:
                print(f"  [forward] using {futures_info['futures_expiry']} future as proxy "
                      f"forward (no exact-expiry future available for this option's expiry)")

            chain = feed.get_option_chain(config.UNDERLYING_SYMBOL, option_expiry,
                                           spot=spot, strike_range_pct=config.OPTION_CHAIN_STRIKE_RANGE_PCT)
            if chain.empty:
                print(f"  [warn] option chain came back with zero rows for {option_expiry} — "
                      f"skipping this cycle (check for a quotes() batch failure above).")
                time.sleep(config.SIGNAL_POLL_SECONDS)
                continue
            risk_free_rate = 0.065  # short-term discount rate ONLY now (forward already
                                     # embeds cost-of-carry) — still a market input to
                                     # keep current, just a much smaller lever than before

            # chain has REAL strike/oi/ltp from Kotak. IV and greeks aren't
            # provided by the broker, so back IV out of the real LTP using the
            # REAL FUTURES FORWARD as the anchor (not spot + an assumed dividend
            # yield), then get delta/gamma/theta/vega from that IV. Rows where
            # no IV solves (stale/illiquid quotes) are dropped, never guessed.
            chain["iv"] = chain.apply(
                lambda r: implied_volatility(forward, r["strike"], t_years, r["ltp"],
                                              risk_free_rate, r["option_type"]),
                axis=1)
            dropped = chain["iv"].isna().sum()
            if dropped:
                print(f"  [chain] dropped {dropped} strikes with no solvable IV "
                      f"(stale/illiquid LTP)")
            chain = chain.dropna(subset=["iv"]).copy()
            if chain.empty:
                print(f"  [warn] every strike was dropped (no solvable IV anywhere) — "
                      f"skipping this cycle. If this keeps happening, check whether "
                      f"quotes() batches are actually returning real LTP data.")
                time.sleep(config.SIGNAL_POLL_SECONDS)
                continue

            # Catches the specific failure mode diagnosed from a live GEX
            # spike: one strike's stale/delayed LTP produces an implausible
            # IV jump vs its own previous cycle, and if that strike carries
            # real OI, its distorted gamma briefly dominates total_gex with
            # no matching move in ATM IV. Excluded here, not silently —
            # flagged rows are printed so you can see exactly what happened.
            chain, iv_outliers = iv_guard.filter(chain)
            if not iv_outliers.empty:
                print(f"  [iv-guard] excluded {len(iv_outliers)} strike(s) with implausible "
                      f"cycle-over-cycle IV jumps: "
                      f"{iv_outliers[['strike','option_type','iv_change']].to_dict('records')}")

            chain["delta"] = chain.apply(
                lambda r: black_scholes_delta(forward, r["strike"], t_years, r["iv"],
                                                risk_free_rate, r["option_type"]),
                axis=1)
            chain["gamma"] = chain.apply(
                lambda r: black_scholes_gamma(forward, r["strike"], t_years, r["iv"], risk_free_rate),
                axis=1)
            chain["theta_per_day"] = chain.apply(
                lambda r: black76_theta(forward, r["strike"], t_years, r["iv"],
                                          risk_free_rate, r["option_type"]) / 365.0,
                axis=1)
            chain["vega_per_1pct"] = chain.apply(
                lambda r: black76_vega(forward, r["strike"], t_years, r["iv"], risk_free_rate) / 100.0,
                axis=1)

            atm_iv = estimate_atm_iv(chain, spot)
            next_vol = vol_engine.forecast_next_session_vol()
            vrp = vol_engine.volatility_risk_premium(implied_vol_annualized=atm_iv)

            regime = regime_classifier.classify(
                current_vol_forecast=next_vol["annualized_vol"],
                vol_forecast_history=forecast_history,
                trend_result=trend_result,
            )

            gamma_context = gamma_engine.compute(chain, spot, t_years=t_years, risk_free_rate=risk_free_rate,
                                                   expected_daily_move=next_vol["daily_vol"] * spot)
            skew_info = compute_skew(chain)
            realized_vol_snapshot = realized_vol_tracker.snapshot()
            print(f"  [gamma] regime={gamma_context['dealer_regime']} "
                  f"flip~{gamma_context['gamma_flip_point']} "
                  f"call_wall~{gamma_context['call_wall']} put_wall~{gamma_context['put_wall']}")
            print(f"  [gamma+] local_gex={gamma_context['local_gex']:,.0f} "
                  f"slope={gamma_context['gex_slope_near_spot']} max_pain={gamma_context['max_pain']}")
            print(f"  [vol] skew={skew_info.get('skew')} realized_5/15/30min="
                  f"{realized_vol_snapshot.get('realized_vol_5min')}/"
                  f"{realized_vol_snapshot.get('realized_vol_15min')}/"
                  f"{realized_vol_snapshot.get('realized_vol_30min')} vs atm_iv={atm_iv:.2%}")
            print(f"  [futures] oi={futures_info.get('futures_oi')} volume={futures_info.get('futures_volume')}")

            signal = signal_engine.generate(regime, vrp, next_vol, spot, gamma_context=gamma_context)

            # Log every metric this cycle produced — building the real
            # historical record needed to eventually validate GEX/regime
            # against what NIFTY actually did afterward. See
            # utils/daily_logger.py's module docstring for why this
            # exists and what it does NOT claim.
            common_metrics = {
                "option_expiry": option_expiry,
                "spot": spot,
                "forward": forward,
                "futures_oi": futures_info.get("futures_oi"),
                "futures_volume": futures_info.get("futures_volume"),
                "garch_forecast_vol": next_vol["annualized_vol"],
                "garch_model_used": next_vol["model_used"],
                "atm_iv": atm_iv,
                "vrp_ratio": vrp["vrp_ratio"],
                "iv_skew": skew_info.get("skew"),
                "call_25d_iv": skew_info.get("call_25d_iv"),
                "put_25d_iv": skew_info.get("put_25d_iv"),
                "realized_vol_5min": realized_vol_snapshot.get("realized_vol_5min"),
                "realized_vol_15min": realized_vol_snapshot.get("realized_vol_15min"),
                "realized_vol_30min": realized_vol_snapshot.get("realized_vol_30min"),
                "total_gex": gamma_context["total_gex"],
                "local_gex": gamma_context["local_gex"],
                "gex_slope_near_spot": gamma_context["gex_slope_near_spot"],
                "max_pain": gamma_context["max_pain"],
                "dealer_regime": gamma_context["dealer_regime"],
                "gamma_flip_point": gamma_context["gamma_flip_point"],
                "distance_to_flip": gamma_context["distance_to_flip"],
                "distance_to_flip_in_expected_moves": gamma_context["distance_to_flip_in_expected_moves"],
                "call_wall": gamma_context["call_wall"],
                "distance_to_call_wall": gamma_context["distance_to_call_wall"],
                "put_wall": gamma_context["put_wall"],
                "distance_to_put_wall": gamma_context["distance_to_put_wall"],
                "trend_direction": trend_result.get("direction"),
                "trend_tau": trend_result.get("tau"),
                "trend_p_value": trend_result.get("p_value"),
                "vol_regime": regime["volatility"].get("regime"),
                "regime_label": regime["label"],
                "signal_strategy": signal.strategy,
                "signal_direction": signal.direction,
                "iv_outliers_excluded": len(iv_outliers),
            }
            log_snapshot(common_metrics)

            log_strike_breakdown(gamma_context["row_level_detail"])

            # --- richer snapshot for the dashboard (adds structured lists
            # the flat CSV log doesn't carry) ---
            top_oi_calls = (chain[chain["option_type"] == "CE"]
                             .nlargest(4, "oi")[["strike", "oi", "ltp"]]
                             .to_dict("records"))
            top_oi_puts = (chain[chain["option_type"] == "PE"]
                            .nlargest(4, "oi")[["strike", "oi", "ltp"]]
                            .to_dict("records"))
            gamma_by_strike = [{"strike": float(k), "gex": float(v)}
                                for k, v in gamma_context["by_strike_gex"].items()]

            dashboard_server.update_state({
                **common_metrics,
                "regime_note": gamma_context["regime_note"],
                "top_oi_calls": top_oi_calls,
                "top_oi_puts": top_oi_puts,
                "gamma_by_strike": gamma_by_strike,
                "signal": {
                    "strategy": signal.strategy,
                    "direction": signal.direction,
                    "rationale": signal.rationale,
                    "stop_loss_price_move": signal.stop_loss_price_move,
                    "target_price_move": signal.target_price_move,
                },
            })

            print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] spot={spot} "
                  f"regime={regime['label']} VRP={vrp['vrp_ratio']:+.1%}")
            print(f"  -> {signal.strategy} ({signal.direction})")
            print(f"     {signal.rationale}")
            if signal.gamma_note:
                print(f"     [gamma note] {signal.gamma_note}")
            if signal.stop_loss_price_move:
                print(f"     stop~{signal.stop_loss_price_move:.1f}pts  "
                      f"target~{signal.target_price_move:.1f}pts  "
                      f"(vol-scaled, not fixed)")
            if signal.target_delta:
                strike_row = signal_engine.pick_strike_by_delta(
                    chain, signal.target_delta,
                    option_type="call" if signal.direction == "bullish" else "put")
                if strike_row is not None:
                    print(f"     suggested strike: {strike_row['strike']} "
                          f"(delta={strike_row['delta']:.2f}, "
                          f"theta/day={strike_row['theta_per_day']:.2f}, "
                          f"vega/1%={strike_row['vega_per_1pct']:.2f})")

        except Exception as e:
            print(f"[error] {e}")

        time.sleep(config.SIGNAL_POLL_SECONDS)


def main():
    dashboard_server.start()  # opens the dashboard in your browser immediately;
                               # it'll show "starting..." until the first cycle lands

    feed = KotakDataFeed()
    feed.login()

    dashboard_server.set_status_message("fitting volatility model on historical data…")
    vol_engine, forecast, daily_closes = session_start_fit(feed)
    dashboard_server.set_status_message("reading trend…")
    trend_result = session_trend(feed, daily_closes)
    forecast_history = load_forecast_history()
    dashboard_server.set_status_message("reading lot size and expiry…")
    contract_multiplier = get_contract_multiplier(feed)
    option_expiry = resolve_option_expiry(feed)

    live_loop(feed, vol_engine, trend_result, forecast_history, contract_multiplier, option_expiry)


if __name__ == "__main__":
    main()
