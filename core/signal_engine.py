"""
Long-options-only signal engine.

Directional entries require statistically significant trend context.
Volatility risk premium is an entry-quality diagnostic, not permission
to sell options. The engine never recommends short premium or an
unrequested multi-leg credit structure.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from core.options_math import black_scholes_delta


@dataclass
class OptionSignal:
    strategy: str
    direction: str
    rationale: str
    target_delta: float | None
    stop_loss_vol_multiple: float
    target_vol_multiple: float
    stop_loss_price_move: float | None = None
    target_price_move: float | None = None
    gamma_note: str | None = None


class OptionsSignalEngine:
    def __init__(self, directional_target_delta: float = 0.40,
                 stop_loss_vol_multiple: float = 1.0,
                 target_vol_multiple: float = 2.0):
        if not 0 < directional_target_delta < 1:
            raise ValueError("directional_target_delta must be between 0 and 1.")
        if stop_loss_vol_multiple <= 0 or target_vol_multiple <= 0:
            raise ValueError("Stop and target volatility multiples must be positive.")
        self.directional_target_delta = float(directional_target_delta)
        self.stop_loss_vol_multiple = float(stop_loss_vol_multiple)
        self.target_vol_multiple = float(target_vol_multiple)

    def generate(self, regime: dict, vrp: dict, next_session_vol: dict,
                 spot: float, gamma_context: dict | None = None) -> OptionSignal:
        trend = regime.get("trend", {})
        vol = regime.get("volatility", {})
        trend_dir = trend.get("direction", "no_trend")
        significant = bool(trend.get("significant", False))
        vol_regime = vol.get("regime", "insufficient_history")
        vrp_ratio = float(vrp.get("vrp_ratio", float("nan")))
        daily_vol = float(next_session_vol.get("daily_vol", float("nan")))
        stop_mult, target_mult = self.stop_loss_vol_multiple, self.target_vol_multiple
        gamma_note = self._build_gamma_note(gamma_context, spot, daily_vol)

        def no_signal(reason: str) -> OptionSignal:
            return OptionSignal("NO_SIGNAL", "none", reason, None, stop_mult,
                                target_mult, gamma_note=gamma_note)

        if not (spot > 0 and 0 < daily_vol < 1 and vrp_ratio == vrp_ratio):
            return no_signal("Invalid spot, volatility forecast, or VRP input; refusing to emit a trade.")
        if vol_regime == "insufficient_history":
            return no_signal("Insufficient strictly-prior volatility forecast history for regime classification.")
        if not significant or trend_dir not in {"up", "down"}:
            return no_signal("No statistically significant directional trend. Long option entry is not qualified.")

        direction = "bullish" if trend_dir == "up" else "bearish"
        option_type = "call" if direction == "bullish" else "put"
        stop_move = spot * daily_vol * stop_mult
        target_move = spot * daily_vol * target_mult

        # A positive VRP means implied vol exceeds the model forecast.
        # It flags expensive convexity; it does not justify selling.
        rationale = (
            f"Significant {trend_dir} drift in return series "
            f"(Kendall tau={trend.get('tau', float('nan')):.3f}, "
            f"p={trend.get('p_value', float('nan')):.4g}). "
            f"VRP={vrp_ratio:+.1%} is retained as a premium-cost diagnostic; "
            "entry still requires live spread/liquidity and execution checks."
        )
        if gamma_context:
            rationale += (
                " GEX is an OI-based positioning scenario, not observed dealer inventory; "
                "it is contextual and does not increase position size."
            )
        return OptionSignal(
            strategy=f"LONG_{option_type.upper()}",
            direction=direction,
            rationale=rationale,
            target_delta=self.directional_target_delta,
            stop_loss_vol_multiple=stop_mult,
            target_vol_multiple=target_mult,
            stop_loss_price_move=float(stop_move),
            target_price_move=float(target_move),
            gamma_note=gamma_note,
        )

    @staticmethod
    def _build_gamma_note(gamma_context: dict | None, spot: float,
                          daily_vol: float) -> str | None:
        if not gamma_context:
            return None
        parts = ["GEX is scenario-based: OI does not identify dealer inventory."]
        flip = gamma_context.get("gamma_flip_point")
        if flip is not None:
            units = (spot - float(flip)) / (spot * daily_vol) if daily_vol > 0 else None
            parts.append(f"Modelled gamma flip ~{float(flip):.1f}; spot relation: {gamma_context.get('spot_vs_flip')}.")
            if units is not None:
                parts.append(f"Distance to flip: {units:+.2f} forecast daily-move units.")
        for key, label in (("call_wall", "call-side gamma concentration"),
                           ("put_wall", "put-side gamma concentration")):
            if gamma_context.get(key) is not None:
                parts.append(f"{label} near {float(gamma_context[key]):.1f}.")
        return " ".join(parts)

    @staticmethod
    def pick_strike_by_delta(chain_df, target_delta: float, option_type: str):
        required = {"strike", "option_type", "delta"}
        missing = required.difference(chain_df.columns)
        if missing:
            raise ValueError(f"Option chain missing columns: {sorted(missing)}")
        subset = chain_df[
            chain_df["option_type"].astype(str).str.lower().str.startswith(option_type[0].lower())
        ].copy()
        if subset.empty:
            return None
        subset = subset.dropna(subset=["strike", "delta"])
        subset = subset[np.isfinite(subset["delta"]) & np.isfinite(subset["strike"])]
        if subset.empty:
            return None
        idx = (subset["delta"].abs() - abs(target_delta)).abs().idxmin()
        return subset.loc[idx]
