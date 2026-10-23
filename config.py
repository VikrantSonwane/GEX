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
EXPIRY_DATE_FORMAT = None            # strptime format matching the scrip
                                      # master's expiry strings (e.g. "%d-%b-%Y") —
                                      # read this off inspect_scrip_master.py's
                                      # output and set it explicitly; None falls
                                      # back to format-guessing, a bridge, not
                                      # something to rely on going live.

# --- timeframes -----------------------------------------------------------
# Short-term execution timeframe (signal polling / entry timing)
LOWER_TF_INTERVAL = "5minute"

# --- historical data source ------------------------------------------------
# Kotak has no historical-candle endpoint (confirmed live, see
# data/historical_source.py's module docstring). Three ways to supply
# daily history instead:
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
# Lot size is now read live from the scrip master at session start
# (data/kotak_client.py's get_lot_size()) instead of being hard-coded
# here — NSE revises it periodically and a stale config constant would
# silently corrupt every GEX number. This fallback is used ONLY if that
# live lookup fails for some reason (e.g. scrip master column name
# mismatch) — treat a run that falls back to this as a signal to fix
# the lookup, not as normal operation.
CONTRACT_MULTIPLIER_FALLBACK = 65   # NSE circular SEBI/HO/MRD-PoD2/CIR/P/2024/00181,
                                     # effective 30-Dec-2025 EOD — VERIFY this is still
                                     # current before trusting it; check the live lookup instead.

# --- persistence ------------------------------------------------------------
VOL_FORECAST_HISTORY_PATH = "vol_forecast_history.csv"   # regime terciles are
                                                            # computed from this
