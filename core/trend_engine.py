"""
trend_engine.py
-----------------
Higher-timeframe context ("check the long-term trend first" from your
brief). Uses the Mann-Kendall test rather than moving averages, because
MA crossovers are themselves an arbitrary parameter choice (why 20/50?
why not 21/48?) and lag price heavily. Mann-Kendall asks a cleaner
question directly: "is there a statistically significant monotonic
drift in this window, and how strong is it?"
"""

from __future__ import annotations
import pandas as pd
from utils.stats_utils import trend_test


class TrendEngine:

    def __init__(self, alpha: float = 0.05):
        """
        alpha: significance level for the trend test. 0.05 is the
        conventional statistical default (95% confidence) — not a
        trading-specific tuning knob.
        """
        self.alpha = alpha

    def analyze(self, higher_tf_closes: pd.Series) -> dict:
        """
        higher_tf_closes: close prices on the "long-term" timeframe you
        want context from (e.g. 15-min or hourly bars spanning several
        sessions, or daily closes over the last N weeks).

        Returns direction, significance, and strength (tau). Downstream
        code should treat 'no_trend' or non-significant results as
        genuinely "no directional edge" rather than picking a side.
        """
        if len(higher_tf_closes) < 8:
            return {"direction": "insufficient_data", "p_value": None,
                     "tau": None, "significant": False}

        result = trend_test(higher_tf_closes, alpha=self.alpha)

        if not result["significant"]:
            result["direction"] = "no_trend"

        return result
