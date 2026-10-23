"""
inspect_scrip_master.py
--------------------------
Run this ONCE, manually, before setting config.OPTION_EXPIRY. It logs
in, pulls the real scrip master, and prints:
  - every available expiry string for your UNDERLYING_SYMBOL, exactly
    as Kotak's scrip master formats them (this is the format
    config.OPTION_EXPIRY must match, character-for-character)
  - one full sample row so you can also confirm the OTHER column names
    this project assumes ('symbol', 'strike_price', 'option_type',
    'instrument_type', 'lot_size', 'instrument_token') actually exist
    under those names in your SDK version.

    python3 inspect_scrip_master.py
"""

import config
from data.kotak_client import KotakDataFeed

feed = KotakDataFeed()
feed.login()

raw_master = feed.client.scrip_master(exchange_segment=config.EXCHANGE_SEGMENT)
import pandas as pd
master = pd.DataFrame(raw_master)

print("=== Columns available in your scrip master ===")
print(list(master.columns))

symbol_rows = master[master["symbol"] == config.UNDERLYING_SYMBOL]
if symbol_rows.empty:
    print(f"\nNo rows found for symbol == '{config.UNDERLYING_SYMBOL}'. "
          f"Print master['symbol'].unique() yourself to find the exact "
          f"string Kotak uses for this underlying (it may differ from "
          f"the plain 'NIFTY' assumed in config.py).")
else:
    print(f"\n=== Available expiries for {config.UNDERLYING_SYMBOL} "
          f"(exact format to use in config.OPTION_EXPIRY) ===")
    print(sorted(symbol_rows["expiry"].unique().tolist(),
                 key=lambda s: pd.to_datetime(s, dayfirst=True)))

    print(f"\n=== One sample option row (verify column names/values) ===")
    option_row = symbol_rows[symbol_rows["instrument_type"].isin(["OPTIDX", "OPTSTK"])].iloc[0]
    print(option_row.to_dict())

    print(f"\n=== One sample futures row ===")
    fut_rows = symbol_rows[symbol_rows["instrument_type"].isin(["FUTIDX", "FUTSTK"])]
    if not fut_rows.empty:
        print(fut_rows.iloc[0].to_dict())
    else:
        print("No futures rows found — check 'instrument_type' values with "
              "symbol_rows['instrument_type'].unique()")
