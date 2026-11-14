"""
config.py
----------
Everything in this file is an OPERATIONAL choice (which instrument,
which timeframe, how often to refit) — not a statistical parameter.
Statistical thresholds (regime cutoffs, model selection, trend
significance) are computed from data inside core/ and utils/, never
set here. This separation is deliberate: it's how the engine stays
"no artificial value feeding" — if you catch yourself wanting to add
a hard-coded number here that isn't an operational choice, it belongs
in a test or a percentile calculation instead.
"""

# --- instrument ---------------------------------------------------------
UNDERLYING_SYMBOL = "NIFTY"          # or a stock symbol / BANKNIFTY
EXCHANGE_SEGMENT = "nse_fo"          # per Kotak Neo SDK's segment codes

# --- market inputs and execution conventions --------------------------------
# Market input: update from a current short-tenor INR risk-free proxy
# (prefer a maturity-matched T-bill/OIS input); this is not a fitted parameter.
RISK_FREE_RATE_ANNUALIZED = 0.065

# Operational starting conventions, not empirically optimized thresholds.
DIRECTIONAL_TARGET_DELTA = 0.40
STOP_LOSS_VOL_MULTIPLE = 1.0
TARGET_VOL_MULTIPLE = 2.0
OPTION_EXPIRY_OVERRIDE = None        # leave None for auto-selection (see
                                      # EXPIRY_SELECTION below). Set an exact
                                      # scrip-master expiry string here ONLY
                                      # if you want to manually pin a specific
                                      # contract instead of the auto-picked
                                      # current/next one.
EXPIRY_SELECTION = "current"         # "current" -> nearest expiry that hasn't
                                      # passed its market close yet. "next" ->
                                      # the one after that. Auto-resolved fresh
                                      # every session_start from the live scrip
                                      # master — never a hand-typed date that
                                      # goes stale after rollover.
EXPIRY_DATE_FORMAT = None            # Kotak Expiries API returns ISO dates.
                                      # None lets date parsing handle ISO or a
                                      # manually supplied override format.

# --- timeframes -----------------------------------------------------------
# Short-term execution timeframe (signal polling / entry timing)
LOWER_TF_INTERVAL = "5minute"

# --- historical data source ------------------------------------------------
# Kotak's current SDK exposes historical OHLC candles, but this
# orchestrator does not yet route that endpoint into the fit loader.
# Until that integration is completed and tested, use one of these
# explicit history sources:
HISTORICAL_DATA_SOURCE = "csv"       # "csv" (your own data), "jugaad" (pulls
                                      # daily NSE data live, no file needed),
                                      # or "yfinance" (Yahoo Finance)

# --- "csv" source settings (used when HISTORICAL_DATA_SOURCE == "csv") -----
DAILY_HISTORY_CSV_PATH = "Raw data/nifty_ohlc.csv"  # your confirmed real file, 1988 rows,
                                                      # Aug 2018 - Sep 2026, verified working
DAILY_HISTORY_CSV_DATE_COLUMN = "Date"              # match your file's actual header
DAILY_HISTORY_CSV_CLOSE_COLUMN = "Close"            # match your file's actual header

# --- "jugaad" source settings (used when HISTORICAL_DATA_SOURCE == "jugaad") ---
JUGAAD_INDEX_SYMBOL = "NIFTY 50"     # exact NSE index name jugaad-data expects
                                      # ('NIFTY 50', 'NIFTY BANK', etc.)
JUGAAD_SYNC_TO_CSV = True             # persist each day's jugaad pull into
                                      # DAILY_HISTORY_CSV_PATH, so your local
                                      # archive keeps growing automatically —
                                      # set False to use jugaad live without
                                      # ever writing to the CSV file

# --- "yfinance" source settings (used when HISTORICAL_DATA_SOURCE == "yfinance") ---
# '^NSEI' for NIFTY 50, '^NSEBANK' for BANK NIFTY.
HISTORICAL_DATA_TICKER = "^NSEI"

DAILY_HISTORY_LOOKBACK_DAYS = 400

# --- trend timeframe --------------------------------------------------------
# "daily" (default): the trend test runs on the same daily closes used
# for the volatility model, over the last HIGHER_TF_LOOKBACK_SESSIONS
# sessions — no separate historical intraday data needed. Real intraday
# granularity still comes from Kotak LIVE during the session; this only
# affects the HISTORICAL trend read. Reliable free intraday historical
# NIFTY data doesn't really exist publicly (see historical_source.py's
# docstring) — "daily" avoids depending on that entirely.
# "intraday": set this ONLY if you have your own genuine intraday
# historical CSV — then also fill in the INTRADAY_HISTORY_CSV_* settings
# below (or use "yfinance", which does offer limited-lookback intraday).
TREND_TIMEFRAME_SOURCE = "daily"
HIGHER_TF_LOOKBACK_SESSIONS = 20     # how many sessions of history feed the trend test

# --- "intraday" trend source settings (only used if TREND_TIMEFRAME_SOURCE == "intraday") ---
INTRADAY_HISTORY_CSV_PATH = "nifty_5min.csv"
INTRADAY_HISTORY_CSV_DATE_COLUMN = "Date"
INTRADAY_HISTORY_CSV_CLOSE_COLUMN = "Close"
DERIVE_DAILY_FROM_INTRADAY = False   # True = also derive the daily-closes need
                                      # from this same intraday file, if you
                                      # don't have a separate daily source
HIGHER_TF_INTERVAL = "15m"           # yfinance interval string, only used if
                                      # HISTORICAL_DATA_SOURCE == "yfinance" AND
                                      # TREND_TIMEFRAME_SOURCE == "intraday"

# --- option chain scope -----------------------------------------------------
# Limits which strikes get fetched each cycle, to a band around the
# current spot — far OTM strikes (thousands of points away) are
# essentially dead (near-zero OI, no real trading) and just waste
# quotes() calls/time fetching them. This is a real speedup (fewer,
# more relevant API calls), unlike shrinking batch_size further, which
# only trades speed for safety margin without cutting actual waste.
OPTION_CHAIN_STRIKE_RANGE_PCT = 0.15  # ±15% of spot — comfortably covers
                                        # every strike a 0.40-delta directional
                                        # pick or a reasonable premium-selling
                                        # strike would ever need

# --- refit / polling cadence ----------------------------------------------
REFIT_VOL_MODEL_ONCE_PER_SESSION = True   # refit at session start, not per-tick
SIGNAL_POLL_SECONDS = 90                  # how often the live loop checks for a new signal

# --- contract spec ---------------------------------------------------------
# Lot size is read from Kotak Option Chain common_data.mktLot at session
# start instead of being hard-coded here. A stale lot size corrupts GEX.
# Fallback stays disabled unless the user explicitly verifies a current
# contract specification.
CONTRACT_MULTIPLIER_FALLBACK = None  # Fail closed if live lot-size lookup fails.
                                       # Never reuse a stale NSE lot size in GEX math.

# --- persistence ------------------------------------------------------------
VOL_FORECAST_HISTORY_PATH = "vol_forecast_history.csv"   # regime terciles are
                                                            # computed from this
