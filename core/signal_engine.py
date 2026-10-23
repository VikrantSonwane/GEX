"""
signal_engine.py
------------------
Turns (trend regime, volatility regime, VRP) into an options signal —
strategy type, direction, which strike to look at (by DELTA, not a
fixed points-away-from-spot offset), and a stop/target expressed as a
multiple of the model's own volatility forecast (not a fixed point
value). This is the "manual execution" layer: it never places an
order, it prints a structured recommendation for you to act on.
"""

from __future__ import annotations
from dataclasses import dataclass
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

    def __init__(self, directional_target_delta: float = 0.40):
        """
        directional_target_delta: the delta magnitude used to pick a
        strike for a DIRECTIONAL buy (calls/puts). 0.40 is the
        conventional "slightly OTM, still meaningfully responsive to
        the underlying" convention widely used for short-dated
        directional option buying — documented as a starting convention,
        not backtested into this engine. Treat it as configurable, not
        as a statistically-derived output like everything else here.
        """
        self.directional_target_delta = directional_target_delta

    def generate(self, regime: dict, vrp: dict, next_session_vol: dict,
                 spot: float, gamma_context: dict | None = None) -> OptionSignal:
        """
        gamma_context: optional output of GammaExposureEngine.compute().
        When supplied, it can (a) override the no-trend/high-vol premium-
        selling branch when dealers are net short gamma — that's exactly
        the regime where naked premium selling is most dangerous, so the
        override matters more than a footnote — and (b) attach an
        informational note everywhere else (gamma flip proximity, wall
        levels) without silently changing the trend/VRP-driven strategy
        choice.
        """
        trend_dir = regime["trend"]["direction"]
        trend_significant = regime["trend"].get("significant", False)
        vol_regime = regime["volatility"]["regime"]
        vrp_ratio = vrp["vrp_ratio"]

        daily_vol = next_session_vol["daily_vol"]
        stop_mult, target_mult = 1.0, 2.0   # risk:reward convention, not a statistical output
        stop_price_move = spot * daily_vol * stop_mult
        target_price_move = spot * daily_vol * target_mult

        gamma_note = self._build_gamma_note(gamma_context, spot, daily_vol)

        # --- decision logic -------------------------------------------------
        if vol_regime == "insufficient_history":
            return OptionSignal(
                strategy="NO_SIGNAL",
                direction="none",
                rationale="Not enough trailing volatility-forecast history yet "
                           "to classify a regime (need >=20 sessions of stored "
                           "forecasts). Building history — no artificial signal "
                           "will be generated until this exists.",
                target_delta=None,
                stop_loss_vol_multiple=stop_mult,
                target_vol_multiple=target_mult,
            )

        if not trend_significant:
            # No statistically valid directional edge from the higher timeframe.
            if vol_regime == "high" and vrp_ratio > 0:
                if gamma_context and gamma_context["dealer_regime"] == "negative_gamma":
                    # This is precisely the regime where naked premium
                    # selling is most dangerous: rangebound-looking price
                    # action but dealers are hedging in a way that
                    # amplifies any move that does start. Override the
                    # premium-selling call to a long-gamma structure.
                    return OptionSignal(
                        strategy="LONG_STRADDLE_OR_STRANGLE (long gamma protection)",
                        direction="neutral",
                        rationale=(f"No significant trend (p={regime['trend']['p_value']:.3f}) "
                                   f"and rich premium (VRP={vrp_ratio:+.1%}) would normally favor "
                                   f"selling, BUT dealers are net short gamma "
                                   f"(GEX={gamma_context['total_gex']:,.0f}) — hedging flow "
                                   f"amplifies moves in this regime, so a rangebound read is "
                                   f"less trustworthy than usual. Overridden to a long-gamma "
                                   f"structure instead of naked premium selling."),
                        target_delta=None,
                        stop_loss_vol_multiple=stop_mult,
                        target_vol_multiple=target_mult,
                        stop_loss_price_move=stop_price_move,
                        target_price_move=target_price_move,
                        gamma_note=gamma_note,
                    )
                return OptionSignal(
                    strategy="SHORT_STRADDLE_OR_STRANGLE (premium selling)",
                    direction="neutral",
                    rationale=(f"No significant trend (p={regime['trend']['p_value']:.3f}) "
                               f"and volatility regime is HIGH with options priced above "
                               f"the model's forecast (VRP={vrp_ratio:+.1%}). Range-bound, "
                               f"rich premium: classic premium-selling setup."),
                    target_delta=None,
                    stop_loss_vol_multiple=stop_mult,
                    target_vol_multiple=target_mult,
                    stop_loss_price_move=stop_price_move,
                    target_price_move=target_price_move,
                    gamma_note=gamma_note,
                )
            return OptionSignal(
                strategy="NO_SIGNAL",
                direction="none",
                rationale=(f"No significant trend (p={regime['trend']['p_value']:.3f}) and "
                           f"conditions don't meet the premium-selling criteria either "
                           f"(vol_regime={vol_regime}, VRP={vrp_ratio:+.1%}). Standing aside "
                           f"is the statistically honest call here."),
                target_delta=None,
                stop_loss_vol_multiple=stop_mult,
                target_vol_multiple=target_mult,
                gamma_note=gamma_note,
            )

        # Significant trend exists.
        direction = "bullish" if trend_dir == "up" else "bearish"
        option_type = "call" if direction == "bullish" else "put"

        if vrp_ratio > 0.15:
            # Directional bias exists but options are meaningfully rich —
            # a debit spread caps the vol-selling cost you'd otherwise absorb.
            strategy = f"{option_type.upper()}_DEBIT_SPREAD"
            rationale = (f"Significant {trend_dir}trend (tau={regime['trend']['tau']:.2f}, "
                         f"p={regime['trend']['p_value']:.3f}) but options are rich vs the "
                         f"model forecast (VRP={vrp_ratio:+.1%}). A debit spread expresses "
                         f"the direction while limiting exposure to the rich premium.")
        else:
            strategy = f"LONG_{option_type.upper()}"
            rationale = (f"Significant {trend_dir}trend (tau={regime['trend']['tau']:.2f}, "
                         f"p={regime['trend']['p_value']:.3f}) and options priced at/below "
                         f"the model's forecast (VRP={vrp_ratio:+.1%}) — no rich-premium "
                         f"penalty for buying the directional exposure outright.")

        if gamma_context and gamma_context["dealer_regime"] == "negative_gamma":
            rationale += (f" Dealers are also net short gamma (GEX={gamma_context['total_gex']:,.0f}), "
                          f"which tends to reinforce continuation once a trend is already underway "
                          f"— directional conviction here is a bit higher than the trend test alone implies.")

        return OptionSignal(
            strategy=strategy,
            direction=direction,
            rationale=rationale,
            target_delta=self.directional_target_delta,
            stop_loss_vol_multiple=stop_mult,
            target_vol_multiple=target_mult,
            stop_loss_price_move=stop_price_move,
            target_price_move=target_price_move,
            gamma_note=gamma_note,
        )

    @staticmethod
    def _build_gamma_note(gamma_context: dict | None, spot: float, daily_vol: float) -> str | None:
        """
        Builds an informational note about gamma flip proximity and wall
        levels. Distance to the flip point is expressed in units of the
        model's own forecasted daily vol (a z-score-like measure) rather
        than an arbitrary points/percentage cutoff — consistent with the
        rest of the engine never hard-coding a threshold.
        """
        if not gamma_context or gamma_context.get("gamma_flip_point") is None:
            return None

        flip = gamma_context["gamma_flip_point"]
        distance_in_vol_units = (spot - flip) / (spot * daily_vol) if daily_vol > 0 else None

        parts = [f"Gamma flip point ~{flip:.1f} ({gamma_context['spot_vs_flip']})."]
        if distance_in_vol_units is not None and abs(distance_in_vol_units) < 1.0:
            parts.append("Spot is within one forecasted-day's expected move of the flip point — "
                         "the dampened/amplified regime could switch within a session.")
        if gamma_context.get("call_wall") is not None:
            parts.append(f"Call wall (largest positive call gamma*OI) ~{gamma_context['call_wall']:.0f} — "
                         f"a plausible upside magnet/resistance.")
        if gamma_context.get("put_wall") is not None:
            parts.append(f"Put wall ~{gamma_context['put_wall']:.0f} — a plausible downside magnet/support.")
        return " ".join(parts)

    @staticmethod
    def pick_strike_by_delta(chain_df, target_delta: float, option_type: str):
        """
        chain_df: pandas DataFrame from KotakDataFeed.get_option_chain(),
        expected columns include ['strike', 'option_type', 'delta']
        (delta pulled live if the chain provides it; else pre-populate
        this column using black_scholes_delta() before calling this).

        Selects the strike whose |delta| is closest to target_delta —
        this is how the strike is chosen, never a fixed "spot + 100 pts"
        offset, since a fixed point offset means something completely
        different in high-vol vs low-vol regimes.
        """
        subset = chain_df[chain_df["option_type"].str.lower().str.startswith(option_type[0].lower())]
        if subset.empty:
            return None
        idx = (subset["delta"].abs() - target_delta).abs().idxmin()
        return subset.loc[idx]
