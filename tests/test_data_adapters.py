import pandas as pd
from data.kotak_client import KotakDataFeed
from data.historical_source import (
    load_closes_from_csv, resample_intraday_to_daily_closes, sync_ohlc_to_csv,
)


class FakeKotakClient:
    def expiries(self, exchange, underlying):
        return {"exchange": exchange, "underlying": underlying,
                "expiries": ["2026-11-27", "2026-11-20"]}

    def option_chain(self, exchange, underlying, expiry=None,
                     instrument_type=None, count=None):
        if instrument_type == "fut":
            return {"data": {"fut": [
                {"inst": {"expiryDt": "27-NOV-2026"},
                 "quote": {"ltp": "22100.5", "vol": "10000"},
                 "oi": {"cur": "5000"}}
            ]}}
        return {"data": {
            "common_data": {"mktLot": "65"},
            "call": [
                {"instrument": {"neoSymbol": "nse_fo|101", "symbol": "NIFTYCE",
                                "optionType": "CE", "strikePrice": "100"},
                 "quote": {"ltp": "5.5", "volume": 100},
                 "openInterest": {"current": 1000, "previous": 900, "change": 100}},
                {"instrument": {"neoSymbol": "nse_fo|102", "symbol": "NIFTYCE2",
                                "optionType": "CE", "strikePrice": "130"},
                 "quote": {"ltp": "1.0", "volume": 10},
                 "openInterest": {"current": 100, "previous": 100, "change": 0}},
            ],
            "put": [
                {"instrument": {"neoSymbol": "nse_fo|201", "symbol": "NIFTYPE",
                                "optionType": "PE", "strikePrice": "100"},
                 "quote": {"ltp": "4.5", "volume": 90},
                 "openInterest": {"current": 1200, "previous": 1100, "change": 100}},
            ],
        }}

    def search_scrip(self, **kwargs):
        return [{"pSymbol": "999", "pSymbolName": "NIFTY 50"}]

    def quotes(self, **kwargs):
        return [{"ltp": "22000.5", "last_volume": "200000"}]


def test_kotak_expiry_lot_and_chain_normalization():
    feed = KotakDataFeed(client=FakeKotakClient())
    assert feed.get_available_expiries("NIFTY", "nse_fo") == ["2026-11-20", "2026-11-27"]
    assert feed.get_lot_size("NIFTY", "nse_fo") == 65
    chain = feed.get_option_chain("NIFTY", "2026-11-20", spot=100, strike_range_pct=0.15)
    assert set(chain["option_type"]) == {"CE", "PE"}
    assert set(chain["strike"]) == {100.0}
    assert set(chain["instrument_token"]) == {"101", "201"}


def test_kotak_underlying_quote_and_futures_proxy():
    feed = KotakDataFeed(client=FakeKotakClient())
    quote = feed.get_live_quote("NIFTY", "nse_fo")
    assert quote["ltp"] == 22000.5
    future = feed.get_futures_ltp("NIFTY", "2026-11-20")
    assert future["futures_ltp"] == 22100.5
    assert future["is_exact_expiry_match"] is False
    assert future["futures_expiry"] == "2026-11-27"


def test_load_closes_from_csv_validates_and_sorts(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-01", "2026-01-03"],
        "Close": [102, 100, 103],
    }).to_csv(path, index=False)
    prices = load_closes_from_csv(str(path))
    assert prices.index.is_monotonic_increasing
    assert prices.iloc[0] == 100
    assert len(prices) == 3


def test_resample_intraday_closes_to_sessions():
    index = pd.to_datetime([
        "2026-01-01 09:15", "2026-01-01 09:20",
        "2026-01-02 09:15", "2026-01-02 09:20",
    ])
    series = pd.Series([100, 101, 102, 103], index=index)
    daily = resample_intraday_to_daily_closes(series)
    assert list(daily.values) == [101, 103]
    assert len(daily) == 2


def test_sync_ohlc_upserts_by_date_and_preserves_existing_columns(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"Date": ["2026-01-01"], "Close": [100], "Source": ["old"]}).to_csv(path, index=False)
    incoming = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Open": [99, 100], "High": [102, 104], "Low": [98, 99], "Close": [101, 103],
    })
    added = sync_ohlc_to_csv(incoming, str(path))
    result = pd.read_csv(path)
    assert added == 1
    assert len(result) == 2
    assert result.loc[result["Date"] == "2026-01-01", "Close"].iloc[0] == 101
