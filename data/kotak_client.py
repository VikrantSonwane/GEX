"""
Read-only Kotak Neo market-data adapter used by main.py.

This adapter deliberately exposes no order-placement methods. It uses
Kotak Neo's supported market-data methods (search_scrip, quotes,
expiries, option_chain) and normalizes broker responses into stable
pandas structures consumed by the quantitative core.

Environment variables:
  KOTAK_CONSUMER_KEY, KOTAK_MOBILE_NUMBER, KOTAK_UCC, KOTAK_TOTP, KOTAK_MPIN
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


class KotakDataFeed:
    def __init__(self, client: Any | None = None, option_chain_count: int = 100):
        if option_chain_count < 10 or option_chain_count % 10:
            raise ValueError("option_chain_count must be a positive multiple of 10.")
        self.client = client
        self.option_chain_count = int(option_chain_count)
        self._spot_instruments: dict[tuple[str, str], dict] = {}

    def login(self) -> None:
        """Initialize SDK and perform TOTP + MPIN authentication."""
        consumer_key = os.getenv("KOTAK_CONSUMER_KEY")
        mobile = os.getenv("KOTAK_MOBILE_NUMBER")
        ucc = os.getenv("KOTAK_UCC")
        totp = os.getenv("KOTAK_TOTP")
        mpin = os.getenv("KOTAK_MPIN")
        missing = [name for name, value in (
            ("KOTAK_CONSUMER_KEY", consumer_key), ("KOTAK_MOBILE_NUMBER", mobile),
            ("KOTAK_UCC", ucc), ("KOTAK_TOTP", totp), ("KOTAK_MPIN", mpin)
        ) if not value]
        if missing:
            raise RuntimeError(f"Missing required Kotak environment variable(s): {', '.join(missing)}")
        if self.client is None:
            try:
                from neo_api_client import NeoAPI
            except ImportError as exc:
                raise RuntimeError(
                    "Kotak Neo SDK is not installed. Install requirements.txt in this environment."
                ) from exc
            self.client = NeoAPI(consumer_key=consumer_key, environment=os.getenv("KOTAK_ENVIRONMENT", "prod"))
        login_response = self.client.totp_login(
            mobile_number=mobile, ucc=ucc, totp=totp
        )
        self._raise_for_api_error(login_response, "TOTP login")
        validate_response = self.client.totp_validate(mpin=mpin)
        self._raise_for_api_error(validate_response, "TOTP validation")

    @staticmethod
    def _raise_for_api_error(response: Any, operation: str) -> None:
        if isinstance(response, dict):
            errors = response.get("error") or response.get("errors")
            status = str(response.get("status", "")).lower()
            if errors or status in {"error", "failed", "failure"}:
                raise RuntimeError(f"Kotak {operation} failed: {errors or response}")

    def _client(self):
        if self.client is None:
            raise RuntimeError("Kotak client is not initialized. Call login() first.")
        return self.client

    @staticmethod
    def _data(response: Any) -> dict:
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected Kotak response type: {type(response).__name__}")
        data = response.get("data", response)
        if not isinstance(data, dict):
            raise RuntimeError("Kotak response does not contain a data object.")
        return data

    @staticmethod
    def _iso_date(value: str | date | datetime) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.date().isoformat()

    @staticmethod
    def _to_float(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Kotak response field {field!r} is not numeric: {value!r}") from exc
        if not np.isfinite(result):
            raise RuntimeError(f"Kotak response field {field!r} is not finite.")
        return result

    def get_available_expiries(self, symbol: str, exchange_segment: str) -> list[str]:
        response = self._client().expiries(exchange=exchange_segment, underlying=symbol)
        if not isinstance(response, dict):
            raise RuntimeError("Unexpected Kotak expiries response.")
        expiries = response.get("expiries")
        if expiries is None:
            expiries = self._data(response).get("expiries")
        if not isinstance(expiries, list):
            raise RuntimeError(f"Kotak expiries response missing list: {response}")
        parsed = sorted({self._iso_date(item) for item in expiries})
        if not parsed:
            raise RuntimeError(f"No expiries returned for {symbol} on {exchange_segment}.")
        return parsed

    def _fetch_chain_payload(self, symbol: str, expiry: str | None,
                             exchange_segment: str, instrument_type: str,
                             count: int | None = None) -> dict:
        response = self._client().option_chain(
            exchange=exchange_segment,
            underlying=symbol,
            expiry=self._iso_date(expiry) if expiry else None,
            instrument_type=instrument_type,
            count=count or self.option_chain_count,
        )
        return self._data(response)

    def get_lot_size(self, symbol: str, exchange_segment: str) -> int:
        expiry = self.get_available_expiries(symbol, exchange_segment)[0]
        data = self._fetch_chain_payload(
            symbol, expiry, exchange_segment, "option", count=10
        )
        common = data.get("common_data", {})
        lot = common.get("mktLot")
        value = int(float(lot))
        if value <= 0:
            raise RuntimeError(f"Invalid live lot size from Kotak option chain: {lot!r}")
        return value

    def get_live_quote(self, symbol: str, exchange_segment: str) -> dict:
        """Fetch underlying cash/index LTP using its NSE cash scrip token."""
        search_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}.get(
            symbol.upper(), symbol
        )
        cache_key = ("nse_cm", search_symbol.upper())
        instrument = self._spot_instruments.get(cache_key)
        if instrument is None:
            matches = self._client().search_scrip(
                exchange_segment="nse_cm",
                symbol=search_symbol,
                expiry="",
                option_type="",
                strike_price="",
            )
            if not isinstance(matches, list) or not matches:
                raise RuntimeError(
                    f"No NSE cash/index scrip found for {search_symbol!r}. "
                    "Verify symbol mapping and Kotak scrip-master availability."
                )
            exact = [
                row for row in matches
                if str(row.get("pSymbolName", "")).strip().upper() == search_symbol.upper()
            ]
            instrument = (exact or matches)[0]
            token = instrument.get("pSymbol")
            if token is None:
                raise RuntimeError(f"Matched Kotak scrip has no pSymbol token: {instrument}")
            self._spot_instruments[cache_key] = instrument
        token = str(instrument["pSymbol"])
        response = self._client().quotes(
            instrument_tokens=[{"instrument_token": token, "exchange_segment": "nse_cm"}],
            quote_type="all",
        )
        if isinstance(response, dict):
            rows = response.get("data", response.get("quotes", []))
        else:
            rows = response
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"No live quote returned for {search_symbol}: {response}")
        quote = rows[0]
        ltp = self._to_float(quote.get("ltp"), "ltp")
        if ltp <= 0:
            raise RuntimeError(f"Invalid underlying LTP for {search_symbol}: {ltp}")
        return {"ltp": ltp, "volume": quote.get("last_volume"),
                "instrument_token": token, "exchange_segment": "nse_cm"}

    def get_option_chain(self, symbol: str, expiry: str, spot: float,
                         strike_range_pct: float = 0.15) -> pd.DataFrame:
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be finite and strictly positive.")
        if not np.isfinite(strike_range_pct) or strike_range_pct <= 0:
            raise ValueError("strike_range_pct must be finite and strictly positive.")
        data = self._fetch_chain_payload(
            symbol, expiry, "nse_fo", "option", count=self.option_chain_count
        )
        rows = []
        for option_type, side in (("CE", "call"), ("PE", "put")):
            for item in data.get(side, []):
                instrument = item.get("instrument") or {}
                quote = item.get("quote") or {}
                oi_info = item.get("openInterest") or {}
                neo_symbol = str(instrument.get("neoSymbol", ""))
                token = neo_symbol.split("|")[-1] if "|" in neo_symbol else None
                try:
                    strike = self._to_float(instrument.get("strikePrice"), "strikePrice")
                    ltp = self._to_float(quote.get("ltp"), "ltp")
                    oi = self._to_float(oi_info.get("current", 0), "openInterest.current")
                except RuntimeError:
                    continue
                rows.append({
                    "strike": strike, "option_type": option_type,
                    "oi": oi, "ltp": ltp,
                    "volume": pd.to_numeric(quote.get("volume"), errors="coerce"),
                    "symbol": instrument.get("symbol"),
                    "instrument_token": token,
                    "exchange_segment": "nse_fo",
                    "previous_oi": pd.to_numeric(oi_info.get("previous"), errors="coerce"),
                    "oi_change": pd.to_numeric(oi_info.get("change"), errors="coerce"),
                })
        columns = ["strike", "option_type", "oi", "ltp", "volume", "symbol",
                   "instrument_token", "exchange_segment", "previous_oi", "oi_change"]
        chain = pd.DataFrame(rows, columns=columns)
        if chain.empty:
            return chain
        lower, upper = spot * (1 - strike_range_pct), spot * (1 + strike_range_pct)
        chain = chain[chain["strike"].between(lower, upper)].copy()
        return chain.sort_values(["strike", "option_type"]).reset_index(drop=True)

    def get_futures_ltp(self, symbol: str, option_expiry: str) -> dict:
        """Use the exact expiry future where listed; otherwise nearest later future."""
        option_date = pd.Timestamp(self._iso_date(option_expiry))
        data = self._fetch_chain_payload(
            symbol, None, "nse_fo", "fut", count=self.option_chain_count
        )
        futures = data.get("fut", [])
        candidates = []
        for item in futures:
            inst = item.get("inst") or {}
            quote = item.get("quote") or {}
            expiry_value = inst.get("expiryDt")
            if not expiry_value:
                continue
            expiry_date = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True)
            if pd.isna(expiry_date):
                continue
            ltp = pd.to_numeric(quote.get("ltp"), errors="coerce")
            if pd.isna(ltp) or float(ltp) <= 0:
                continue
            candidates.append((expiry_date.normalize(), float(ltp), item))
        eligible = [x for x in candidates if x[0] >= option_date]
        if not eligible:
            raise RuntimeError(
                f"No futures contract on or after option expiry {option_date.date()} "
                f"returned by Kotak for {symbol}."
            )
        chosen = min(eligible, key=lambda x: x[0])
        expiry_date, ltp, item = chosen
        oi_data = item.get("oi") or {}
        quote = item.get("quote") or {}
        return {
            "futures_ltp": ltp,
            "futures_expiry": expiry_date.date().isoformat(),
            "is_exact_expiry_match": bool(expiry_date == option_date),
            "futures_oi": pd.to_numeric(oi_data.get("cur"), errors="coerce"),
            "futures_volume": pd.to_numeric(quote.get("vol"), errors="coerce"),
        }
