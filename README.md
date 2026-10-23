# Intraday Options Signal Engine (GJR-GARCH / EGARCH)

Live-data-feed, manual-execution decision support for intraday NSE
options trading. **This engine never places an order** — it prints a
structured signal (strategy, direction, suggested strike by delta,
vol-scaled stop/target) for you to act on manually via Kotak Neo.

## Design principle: no artificial value feeding

Every threshold in this engine is either:
- derived from the instrument's own trailing distribution (percentiles/terciles),
- the output of a statistical significance test (Mann-Kendall p-value), or
- selected by an information criterion (BIC, choosing between GJR-GARCH and EGARCH).

`config.py` holds only genuinely operational choices (symbol, timeframe,
refit cadence) — never a hidden trading threshold. The two exceptions,
called out explicitly in code comments where they appear, are:
- the Student-t innovation distribution (standard practice for fat-tailed
  equity returns, not a tuned parameter),
- the 0.40 target delta for directional strike selection and the 1x/2x
  stop:target volatility multiple (documented trading conventions, kept
  as clearly-labeled config rather than presented as statistically derived).

## How the pieces fit together

```
daily history ──> VolatilityForecastEngine ──> next-session vol forecast
                   (fits GJR-GARCH + EGARCH,        │
                    keeps the one with lower BIC)    │
                                                      ▼
higher-TF closes ──> TrendEngine (Mann-Kendall) ──> RegimeClassifier ──> OptionsSignalEngine
                                                      ▲                        │
forecast history (CSV) ───────────────────────────────                        ▼
                                                                strategy + strike + vol-scaled stop/target
live option chain (IV) ──> VRP (implied vs forecast) ─────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
```

Kotak Neo credentials (never hard-code these — the client reads them
from the environment):

```bash
export KOTAK_CONSUMER_KEY="..."      # Neo app/web -> More -> Trade API -> Generate application
export KOTAK_MOBILE_NUMBER="..."
export KOTAK_MPIN="..."
export KOTAK_TOTP="..."              # current TOTP code, or wire in pyotp against your registered secret
```

TOTP must be registered once on the Kotak Neo site before this works
(More -> Trade API -> TOTP Registration).

**Easiest way to set these on Windows/PyCharm**: copy `.env.example` to
`.env` in this folder and fill in your real values — `main.py` loads it
automatically via `python-dotenv`. Never commit the real `.env` file
(`.gitignore` already excludes it). Alternatively, set them in
PyCharm's Run/Debug Configuration under "Environment variables", or in
PowerShell for the current session only:
```powershell
$env:KOTAK_CONSUMER_KEY="..."
$env:KOTAK_MOBILE_NUMBER="..."
$env:KOTAK_MPIN="..."
$env:KOTAK_TOTP="..."
```

## Gamma exposure (GEX): what it adds and its honest limits

`core/gamma_exposure.py` estimates **dealer gamma exposure** from the
option chain's REAL open interest (pulled from Kotak Neo's `quotes()`
endpoint — see below), using the standard convention (also used by
SpotGamma/SqueezeMetrics-style dashboards) that dealers are net long
the calls and net short the puts customers trade with them. From that:

- **Positive GEX** → dealers hedge *against* moves → volatility gets
  dampened, price tends to pin near high-OI strikes.
- **Negative GEX** → dealers hedge *with* moves → volatility gets
  amplified, trends can accelerate sharply.
- The **gamma flip point** (where cumulative GEX crosses zero) acts as
  a pivot — calmer above it, more volatile below it, under this
  convention.

This is wired into `signal_engine.py`: when the trend test finds no
significant edge but conditions otherwise look like a premium-selling
setup, a negative-gamma reading **overrides** that to a long-gamma
structure instead — that combination (rangebound-looking price action,
dealers positioned to amplify a break) is exactly where naked premium
selling gets hurt worst. Everywhere else it attaches an informational
note (flip distance, call/put walls) without silently changing the
trend/VRP-driven call.

**Honest limitation**: this is inferred from OI, not observed dealer
books. It's a widely used and reasonable proxy for liquid index chains,
but treat it as a probabilistic tilt on top of the trend/vol signal,
not a certainty — especially early in a session before OI has settled,
or around expiry when OI positioning shifts fast.

## Historical data: Kotak has none — bring your own CSV, or use Yahoo Finance

**Important finding, confirmed on a live authenticated session, not
assumed**: Kotak Neo's Trade API (`kotakneoapi==3.0.1`) has **no
historical-candle endpoint of any kind** — daily or intraday. Running
`dir()` on a real, logged-in `NeoAPI` client shows its actual methods:
`cancel_order`, `create_order_feed`, `create_websocket`, `holdings`,
`limits`, `logout`, `margin_required`, `modify_order`, `order_history`,
`order_report`, `place_order`, `positions`, `quotes`, `scrip_master`,
`search_scrip`, `subscribe`, `totp_login`, `totp_validate`,
`trade_report`, `whatsmyip` — nothing resembling OHLC history. Kotak's
own GitHub issue tracker shows other developers hitting the same wall.

`data/historical_source.py` supports three sources, chosen via
`config.HISTORICAL_DATA_SOURCE`:

**`"csv"` (your own data)** — point `config.DAILY_HISTORY_CSV_PATH` at
your file, set `..._DATE_COLUMN` / `..._CLOSE_COLUMN` to match your
actual header names (defaults assume `Date`/`Close`, the common
NSE-bhavcopy convention).

**`"jugaad"` (recommended if you don't have your own file)** — pulls
daily closes directly from NSE's own website via the `jugaad-data`
package, no file needed. Arguably better than Yahoo Finance for this:
it's official NSE data, actively maintained, and needs no ticker-symbol
translation (`'NIFTY 50'`, not `'^NSEI'`). **Daily only** — see the
note below on why intraday isn't offered here.

**`"yfinance"`** — Yahoo Finance, useful specifically if you want
intraday *historical* bars (Yahoo offers these within a limited
lookback window; jugaad-data/NSE don't publish intraday history at
all through any official channel).

**Important honesty note on intraday historical data**: I looked, and
there is no genuinely reliable, free, actively-maintained source of
historical 5-minute NIFTY data — NSE doesn't publish it publicly, and
the community-scraped bulk datasets you'll find on GitHub are
unofficial one-time snapshots of uncertain provenance and no ongoing
maintenance. Rather than build on that, `config.TREND_TIMEFRAME_SOURCE`
defaults to `"daily"`: the trend test (`core/trend_engine.py`) reuses
the SAME daily closes fetched for the volatility model, over the last
`HIGHER_TF_LOOKBACK_SESSIONS` sessions — no separate intraday history
is needed at all. Real intraday granularity still comes from Kotak
**live**, during the actual trading session (option chain, quotes) —
it's only the *historical* trend read that uses daily data instead.
Set `TREND_TIMEFRAME_SOURCE = "intraday"` only if you have your own
genuine intraday historical CSV, or accept Yahoo's limited-lookback
intraday via `"yfinance"`.

Everything that genuinely must be broker-specific and real-time — live
quotes, option chain OI/LTP, futures LTP, lot size, expiries — still
comes from Kotak via `kotak_client.py`, completely unchanged regardless
of which historical source you pick. `kotak_client.py`'s old
`get_historical_candles()` method is kept only as a loud, explanatory
`NotImplementedError` in case anything still tries to call it.

## Expiry: auto-resolved, not hand-typed

Earlier versions of this project required manually typing a fixed
expiry string into `config.py` — which goes stale the moment that
contract expires, and doesn't distinguish "this week's" vs "next
week's" expiry. `main.py`'s `resolve_option_expiry()` fixes this:

- Pulls every live expiry from the scrip master via
  `feed.get_available_expiries()`.
- Drops any that have already passed **market close** (15:30 IST by
  default) — so on expiry day itself, before close, that day's own
  contract still correctly counts as "current"; only after close does
  it roll to the next one.
- Sorts chronologically (not alphabetically — `"04-Oct-2026"` sorting
  before `"11-Sep-2026"` as a plain string would silently pick the
  wrong contract) and picks index 0 for `config.EXPIRY_SELECTION =
  "current"` or index 1 for `"next"`.
- Re-checked at the top of every `live_loop()` iteration, so a process
  left running across an expiry rollover rolls itself forward instead
  of silently querying an expired contract forever.
- `config.OPTION_EXPIRY_OVERRIDE` lets you manually pin an exact
  contract instead, if you want that.

`test_pipeline.py`'s section 7 tests this logic in isolation (a stub
feed returning expiries in a deliberately shuffled order, to prove the
chronological sort — not the earlier alphabetical bug it initially
caught — is what's actually running).

## Where the option chain data actually comes from

Kotak Neo's Trade API does **not** expose a single "option chain"
endpoint (their own FAQ page confirms this isn't available as of
writing). `data/kotak_client.py`'s `get_option_chain()` builds the chain
from what the API actually offers:

1. `scrip_master(exchange_segment)` — the full instrument list (cached
   per session, it's a large file that doesn't need re-fetching every poll).
2. Filtered locally for this symbol/expiry's CE and PE rows -> every
   real strike and its instrument token.
3. `quotes(instrument_tokens=[...], quote_type="all")`, batched — this
   is where the **real open interest and real LTP** come from.

Kotak's quote response does **not** include implied volatility or
greeks (true of most Indian broker APIs, not a Kotak-specific gap).

## Pricing anchor: real futures forward, not spot + an assumed dividend yield

`core/options_math.py` now uses **Black-76** (forward-based option
pricing) instead of spot-based Black-Scholes. The difference matters:
spot-based pricing requires you to *assume* a dividend yield/cost-of-
carry for the index, which is exactly the kind of "artificial value"
this project is built to avoid. Black-76 sidesteps that entirely by
pricing off the **futures LTP** — a real, live, exchange-traded number
that the market has already built cost-of-carry into.

`data/kotak_client.py`'s `get_futures_ltp()` pulls the nearest-expiry
future (>= the option's expiry) via the same `scrip_master` + `quotes()`
pattern as the option chain. `main.py`'s `live_loop()` uses that as the
forward anchor for `implied_volatility()`, then derives delta, gamma,
theta, and vega from the recovered IV. Strikes where no IV solves
(stale/illiquid LTP) are dropped rather than filled with a guess —
`test_pipeline.py` verifies the solver itself by generating a synthetic
price from a *known* IV and confirming it recovers that exact IV before
any of this touches live data.

**One flagged approximation**: NSE doesn't list a future for every
weekly options expiry — only monthly futures exist. For a weekly
option, `get_futures_ltp()` falls back to the current-month future as
the closest available forward proxy, which can carry a small basis
mismatch versus the true implied forward for that specific week. A more
precise version would interpolate between the current- and next-month
futures by time-to-expiry; not implemented yet, called out explicitly
in the method's docstring rather than silently assumed away.

The only remaining assumed input is a short-term discount rate `r`
(`main.py`'s `risk_free_rate`), used purely to discount settlement value
back to today — a much smaller, lower-controversy lever than the
dividend-yield assumption it replaces. Keep it close to the current
T-bill/repo rate and update periodically.

## Before going live

1. **Run `python test_pipeline.py` first, always.** It exercises the entire
   statistical chain on synthetic data with no API calls, so you can
   confirm the math behaves before touching real credentials.
2. **Run `python inspect_scrip_master.py` once, after logging in.** It
   prints every real expiry string for your `UNDERLYING_SYMBOL` in
   Kotak's exact format — set `config.EXPIRY_DATE_FORMAT` to the
   matching strptime pattern (e.g. `"%d-%b-%Y"`); expiry SELECTION
   itself (current vs next) is now automatic, see below. It also prints
   a sample option and futures row so you can confirm the column names
   `kotak_client.py` assumes (`symbol`, `strike_price`, `option_type`,
   `instrument_type`, `lot_size`, `instrument_token`) actually match
   your SDK version.
3. **Lot size is read live, not hard-coded.** `main.py` calls
   `feed.get_lot_size()` at session start, which reads it straight from
   the scrip master — this self-updates when NSE revises it, instead of
   depending on a config constant that goes stale. `config.CONTRACT_MULTIPLIER_FALLBACK`
   only kicks in if that live lookup fails, and the code prints a loud
   warning when it does — treat that warning as "go fix the lookup,"
   not as normal operation.
4. **Time-to-expiry is computed for real** via `utils/dates.py`'s
   `time_to_expiry_years()`, counting to the actual expiry date's
   market close (15:30 IST by default) rather than a placeholder
   constant. Pass `config.EXPIRY_DATE_FORMAT` explicitly once you know
   it from step 2 — the function's format-guessing fallback exists only
   to get you unblocked before that, not to rely on long-term.
5. **Verify column names against your SDK version.** Kotak has shipped
   several SDK generations (`neo_api_client` legacy, `kotak-neo-api-v2`,
   current `kotakneoapi`). `data/kotak_client.py`'s `scrip_master()` and
   `quotes()` field names are written against the current SDK's
   documented shape, but check a real response against your installed
   version (step 2 does this for you) before trusting the merge in
   `get_option_chain()`.
6. **Give the vol model enough daily history.** `VolatilityForecastEngine.fit()`
   refuses to fit below 250 daily observations on purpose — an
   under-fed GARCH model produces unstable parameters, which is exactly
   the "artificial value" this project is built to avoid.

## Files

| File | Responsibility |
|---|---|
| `utils/stats_utils.py` | Realized vol, Mann-Kendall trend test, distribution-derived regime cutoffs |
| `core/volatility_engine.py` | Fits GJR-GARCH & EGARCH, picks by BIC, forecasts vol, computes VRP |
| `core/trend_engine.py` | Higher-timeframe trend significance test |
| `core/regime_classifier.py` | Combines trend + vol distribution into one regime label |
| `core/signal_engine.py` | Regime (+ optional gamma context) -> strategy/direction/strike-delta/vol-scaled stop-target |
| `core/options_math.py` | Black-76 (forward-based) price/delta/gamma/theta/vega + IV solver (single source of truth) |
| `core/gamma_exposure.py` | Dealer gamma exposure (GEX) proxy from real OI: dealer regime, gamma flip point, call/put walls, net delta exposure |
| `data/historical_source.py` | Daily closes from your own CSV, jugaad-data (official NSE, recommended), or Yahoo Finance — Kotak has no historical-candle endpoint at all (confirmed live). Trend test defaults to reusing daily closes rather than needing separate intraday history. |
| `data/kotak_client.py` | Kotak Neo data-only wrapper: live quote, futures LTP (forward anchor), lot size, expiries, and option chain built from real scrip_master + quotes() (OI/LTP real; IV/greeks computed) |
| `utils/dates.py` | Real time-to-expiry calculation (to market close, not midnight) — replaces the old placeholder |
| `inspect_scrip_master.py` | One-off script: prints real expiry strings/formats and sample rows so you can fill in `config.py` correctly |
| `main.py` | Orchestrates one session: fit -> trend -> live polling loop (prints signals) |
| `test_pipeline.py` | End-to-end synthetic-data test, no API required |
