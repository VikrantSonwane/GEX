"""
Historical close/OHLC loaders.

CSV is the default source to preserve user-controlled data provenance.
Yahoo Finance and jugaad-data are optional alternatives loaded lazily.
No network request is made at import time.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_series(series: pd.Series, name: str = "close") -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    out = out[np.isfinite(out.to_numpy()) & (out > 0)]
    if out.empty:
        raise ValueError(f"No valid positive observations found for {name}.")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.name = name
    return out


def load_closes_from_csv(path: str, date_column: str = "Date",
                         close_column: str = "Close",
                         lookback_days: int | None = None) -> pd.Series:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Historical CSV not found: {source.resolve()}")
    frame = pd.read_csv(source)
    missing = {date_column, close_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing column(s): {sorted(missing)}")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[close_column] = pd.to_numeric(frame[close_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, close_column]).sort_values(date_column)
    frame = frame.drop_duplicates(subset=[date_column], keep="last")
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive or None.")
        cutoff = frame[date_column].max() - pd.Timedelta(days=int(lookback_days))
        frame = frame[frame[date_column] >= cutoff]
    series = pd.Series(frame[close_column].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(frame[date_column]), name="close")
    return _clean_series(series)


def resample_intraday_to_daily_closes(intraday_closes: pd.Series) -> pd.Series:
    if not isinstance(intraday_closes.index, pd.DatetimeIndex):
        raise TypeError("Intraday close series must use a DatetimeIndex.")
    clean = _clean_series(intraday_closes)
    daily = clean.groupby(clean.index.normalize()).last()
    daily.index = pd.DatetimeIndex(daily.index)
    daily.name = "close"
    return daily


def get_daily_closes(ticker: str, lookback_days: int) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to use HISTORICAL_DATA_SOURCE='yfinance'.") from exc
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(lookback_days))
    frame = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         interval="1d", auto_adjust=True, progress=False)
    if frame.empty or "Close" not in frame:
        raise RuntimeError(f"Yahoo Finance returned no daily close data for {ticker}.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return _clean_series(close, name="close")


def get_intraday_closes(ticker: str, interval: str = "15m",
                        lookback_days: int = 20) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to use Yahoo Finance intraday history.") from exc
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(lookback_days))
    frame = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         interval=interval, auto_adjust=True, progress=False)
    if frame.empty or "Close" not in frame:
        raise RuntimeError(f"Yahoo Finance returned no {interval} data for {ticker}.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return _clean_series(close, name="close")


def get_daily_ohlc_jugaad(symbol: str, lookback_days: int) -> pd.DataFrame:
    try:
        from jugaad_data.nse import index_df, stock_df
    except ImportError as exc:
        raise RuntimeError("Install jugaad-data to use HISTORICAL_DATA_SOURCE='jugaad'.") from exc
    end = date.today()
    start = end - timedelta(days=int(lookback_days))
    if symbol.upper() in {"NIFTY 50", "NIFTY BANK", "NIFTY NEXT 50"}:
        raw = index_df(symbol, start, end)
        date_col = "HistoricalDate"
        close_col = "CLOSE"
        open_col, high_col, low_col = "OPEN", "HIGH", "LOW"
    else:
        raw = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
        date_col = "DATE"
        close_col = "CLOSE"
        open_col, high_col, low_col = "OPEN", "HIGH", "LOW"
    if raw.empty:
        raise RuntimeError(f"jugaad-data returned no OHLC observations for {symbol}.")
    frame = raw.rename(columns={
        date_col: "Date", open_col: "Open", high_col: "High",
        low_col: "Low", close_col: "Close",
    })
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    return frame[["Date", "Open", "High", "Low", "Close"]].reset_index(drop=True)


def get_daily_closes_jugaad(symbol: str, lookback_days: int) -> pd.Series:
    frame = get_daily_ohlc_jugaad(symbol, lookback_days)
    return pd.Series(frame["Close"].to_numpy(), index=pd.DatetimeIndex(frame["Date"]), name="close")


def sync_ohlc_to_csv(ohlc: pd.DataFrame, path: str,
                     date_column: str = "Date") -> int:
    required = {"Date", "Close"}
    missing = required.difference(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {sorted(missing)}")
    source = ohlc.copy().rename(columns={"Date": date_column})
    source[date_column] = pd.to_datetime(source[date_column], errors="coerce")
    source["Close"] = pd.to_numeric(source["Close"], errors="coerce")
    source = source.dropna(subset=[date_column, "Close"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_csv(target)
        if date_column not in existing.columns:
            raise ValueError(f"Existing CSV does not contain date column {date_column!r}.")
        existing[date_column] = pd.to_datetime(existing[date_column], errors="coerce")
        if "Close" not in existing.columns and "Price" in existing.columns:
            source = source.rename(columns={"Close": "Price"})
            close_col = "Price"
        else:
            close_col = "Close"
        new_dates = set(source[date_column].dropna()) - set(existing[date_column].dropna())
        columns = list(dict.fromkeys([*existing.columns, *source.columns]))
        merged = pd.concat([existing.reindex(columns=columns),
                            source.reindex(columns=columns)], ignore_index=True)
        merged = merged.dropna(subset=[date_column]).sort_values(date_column)
        merged = merged.drop_duplicates(subset=[date_column], keep="last")
        merged.to_csv(target, index=False)
        return len(new_dates)
    source.to_csv(target, index=False)
    return int(source[date_column].nunique())
