"""
Higher-timeframe trend context based on return increments.

The statistical test is run on log returns rather than price levels.
The Hamed-Rao modified Mann-Kendall variance correction is used by
default to reduce false significance from serial autocorrelation.
"""
from __future__ import annotations

import pandas as pd
from utils.stats_utils import trend_test


class TrendEngine:
    def __init__(self, alpha: float = 0.05, method: str = "hamed_rao"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        self.alpha = float(alpha)
        self.method = method

    def analyze(self, higher_tf_closes: pd.Series) -> dict:
        if higher_tf_closes is None:
            raise ValueError("higher_tf_closes cannot be None.")
        if len(higher_tf_closes) < 9:
            return {"direction": "insufficient_data", "p_value": None,
                    "tau": None, "significant": False, "observations": max(0, len(higher_tf_closes)-1),
                    "method": self.method, "input": "log_returns"}
        return trend_test(higher_tf_closes, alpha=self.alpha, method=self.method)
