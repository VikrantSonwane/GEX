"""
regime_classifier.py
----------------------
Combines the volatility engine's output with the trend engine's output
into one regime label. This is where "long-term trend + short-term
timeframe" from your brief actually connects: the trend comes from the
higher timeframe, the vol regime comes from the (E)GARCH forecast, and
both feed the SignalEngine together.

No fixed cutoffs here either — the vol regime boundary is the trailing
distribution's own terciles (see stats_utils.distributional_regime).
"""

from __future__ import annotations
import pandas as pd
from utils.stats_utils import distributional_regime


class RegimeClassifier:

    def classify(self, current_vol_forecast: float,
                 vol_forecast_history: pd.Series,
                 trend_result: dict) -> dict:
        """
        current_vol_forecast: today's annualized GARCH/EGARCH/GJR forecast.
        vol_forecast_history: a trailing series of PAST forecasts (build
            this by running the volatility engine's forecast once per
            session and appending) — used only to derive this instrument's
            own low/normal/high vol terciles.
        trend_result: output of TrendEngine.analyze().
        """
        vol_regime_info = distributional_regime(current_vol_forecast, vol_forecast_history)
        vol_regime = vol_regime_info["regime"]
        trend_direction = trend_result.get("direction", "no_trend")

        label = f"{trend_direction}_{vol_regime}vol"

        return {
            "label": label,
            "trend": trend_result,
            "volatility": vol_regime_info,
        }
