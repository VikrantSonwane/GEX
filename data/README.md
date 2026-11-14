# Data adapters

This directory contains Python source adapters and is intentionally tracked
by Git. Local datasets, credentials and generated outputs belong outside
version control.

- kotak_client.py wraps Kotak Neo market-data methods only; it does not place orders.
- historical_source.py loads CSV, Yahoo Finance or jugaad-data daily/intraday history.
- The .gitignore intentionally ignores local datasets, not Python source.

For broker access, create a .env from .env.example and provide the consumer
key, registered mobile number, UCC, current TOTP and MPIN. Never commit .env.

The broker adapter uses the current kotakneoapi package interface:
expiries, option_chain, search_scrip, quotes, totp_login and totp_validate.
No credentials or network calls are required by the unit tests; those use a
mocked API client. Verify live response compatibility with a read-only market
session before relying on it for trading decisions.
