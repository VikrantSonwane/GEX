"""
Higher-timeframe trend context using an autocorrelation-adjusted
Mann-Kendall test on log-price levels.
"""
from __future__ import annotations

import pandas as pd
from utils.stats_utils import trend_test


class TrendEngine:
    def __init__(self, alpha: float = 0.05, method: str = "hamed_rao"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        if method not in {"hamed_rao", "original"}:
            raise ValueError("method must be 'hamed_rao' or 'original'.")
        self.alpha = float(alpha)
        self.method = method

    def analyze(self, higher_tf_closes: pd.Series) -> dict:
        if higher_tf_closes is None:
            raise ValueError("higher_tf_closes cannot be None.")
        if len(higher_tf_closes) < 8:
            return {"direction": "insufficient_data", "p_value": None,
                    "tau": None, "significant": False,
                    "observations": int(len(higher_tf_closes)),
                    "method": self.method, "input": "log_prices"}
        return trend_test(higher_tf_closes, alpha=self.alpha, method=self.method)
