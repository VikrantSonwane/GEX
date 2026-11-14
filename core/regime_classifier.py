"""
Regime classification against strictly prior forecast history.
Thresholds remain distribution-derived; no return fitting is performed here.
"""
from __future__ import annotations

import pandas as pd
from utils.stats_utils import distributional_regime


class RegimeClassifier:
    def classify(self, current_vol_forecast: float, vol_forecast_history: pd.Series,
                 trend_result: dict) -> dict:
        if trend_result is None:
            raise ValueError("trend_result is required.")
        info = distributional_regime(current_vol_forecast, vol_forecast_history)
        direction = trend_result.get("direction", "no_trend")
        return {
            "label": f"{direction}_{info['regime']}vol",
            "trend": trend_result,
            "volatility": info,
        }
