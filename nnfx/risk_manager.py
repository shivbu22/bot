"""Risk management and dynamic position sizing engine adhering to NNFX rules."""

from dataclasses import dataclass
from typing import Optional
from nnfx.config import RiskConfig


@dataclass(frozen=True)
class PositionSpecs:
    """Detailed specifications for an executed two-tranche NNFX position."""
    direction: int                # +1 for Long, -1 for Short
    entry_price: float
    atr: float
    initial_sl: float
    tp1_price: float
    breakeven_price: float
    risk_amount_dollars: float
    total_units: float
    tranche_1_units: float        # Scale 1 (TP1) - 50%
    tranche_2_units: float        # Scale 2 (Runner) - 50%
    sl_distance: float
    tp1_distance: float


class RiskManager:
    """
    Computes ATR-based dynamic position sizing, initial stop loss,
    take-profit 1 level, and two-tranche allocations according to NNFX rules:
    - Total trade risk is strictly capped at risk_pct of current equity (1-2%).
    - Stop Loss placed at exactly 1.5 * ATR(14).
    - Take Profit 1 placed at exactly 1.0 * ATR(14).
    - Tranche 1: 50% of units.
    - Tranche 2: 50% of units.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def calculate_position(
        self,
        current_equity: float,
        entry_price: float,
        atr: float,
        direction: int,
    ) -> PositionSpecs:
        """
        Calculates position size and price targets.

        Parameters
        ----------
        current_equity : float
            Current account equity in quote currency.
        entry_price : float
            Anticipated or actual execution price.
        atr : float
            Current Average True Range value (period 14).
        direction : int
            +1 for Long, -1 for Short.

        Returns
        -------
        PositionSpecs
            Complete mathematical specification of the trade.
        """
        if current_equity <= 0:
            raise ValueError("Account equity must be strictly positive.")
        if entry_price <= 0:
            raise ValueError("Entry price must be strictly positive.")
        if atr <= 0:
            raise ValueError("ATR must be strictly positive.")
        if direction not in (1, -1):
            raise ValueError("Direction must be either +1 (Long) or -1 (Short).")

        # 1. Total dollar risk on account equity
        risk_dollars = current_equity * self.config.risk_pct

        # 2. Distances based on ATR
        sl_distance = self.config.sl_atr_multiplier * atr
        tp1_distance = self.config.tp1_atr_multiplier * atr

        # 3. Stop loss and TP1 price levels
        if direction == 1:
            initial_sl = entry_price - sl_distance
            tp1_price = entry_price + tp1_distance
        else:
            initial_sl = entry_price + sl_distance
            tp1_price = entry_price - tp1_distance

        breakeven_price = entry_price

        # 4. Total units calculation to guarantee exact risk percentage
        # Risk = Units * sl_distance => Units = Risk / sl_distance
        raw_units = risk_dollars / sl_distance

        # Ensure minimum position size threshold
        total_units = max(raw_units, self.config.min_position_size)

        # 5. Partition into 50% Tranche 1 and 50% Tranche 2
        tranche_1_units = total_units * self.config.tranche_1_pct
        tranche_2_units = total_units * self.config.tranche_2_pct

        return PositionSpecs(
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            initial_sl=initial_sl,
            tp1_price=tp1_price,
            breakeven_price=breakeven_price,
            risk_amount_dollars=risk_dollars,
            total_units=total_units,
            tranche_1_units=tranche_1_units,
            tranche_2_units=tranche_2_units,
            sl_distance=sl_distance,
            tp1_distance=tp1_distance,
        )

    def calculate_trailing_stop(
        self,
        direction: int,
        current_sl: float,
        highest_high_since_tp1: float,
        lowest_low_since_tp1: float,
        current_atr: float,
    ) -> float:
        """
        Calculates ATR trailing stop for the runner tranche.
        Stop trails runner_trailing_atr_mult * ATR behind highest high (for Long) or lowest low (for Short).
        Never loosens the stop (monotonic ratchet).
        """
        trail_mult = getattr(self.config, "runner_trailing_atr_mult", self.config.sl_atr_multiplier)
        trail_dist = trail_mult * current_atr
        if direction == 1:
            proposed_sl = highest_high_since_tp1 - trail_dist
            return max(current_sl, proposed_sl)
        else:
            proposed_sl = lowest_low_since_tp1 + trail_dist
            return min(current_sl, proposed_sl)

    def calculate_ratchet_stop(
        self,
        direction: int,
        entry_price: float,
        current_sl: float,
        highest_high_since_tp1: float,
        lowest_low_since_tp1: float,
        current_atr: float,
    ) -> float:
        """
        Calculates dynamic multi-tier ratchet stop for the runner tranche.
        Guarantees that as profit expands, higher profit levels are permanently locked in.
        Tiers default: ((2.0, 0.5), (4.0, 2.0), (8.0, 5.0))
        - At +2.0 ATR profit: locks in +0.5 ATR guaranteed profit.
        - At +4.0 ATR profit: locks in +2.0 ATR guaranteed profit.
        - At +8.0 ATR profit: locks in +5.0 ATR guaranteed profit.
        Monotonic ratchet: Stop only advances, never loosens.
        """
        if current_atr <= 0:
            return current_sl

        proposed_sl = current_sl

        if direction == 1:
            profit_atr = (highest_high_since_tp1 - entry_price) / current_atr
            for threshold, lock_atr in sorted(self.config.ratchet_tiers, key=lambda x: x[0]):
                if profit_atr >= threshold:
                    tier_sl = entry_price + (lock_atr * current_atr)
                    proposed_sl = max(proposed_sl, tier_sl)

            trail_mult = getattr(self.config, "runner_trailing_atr_mult", 2.0)
            trail_sl = highest_high_since_tp1 - (trail_mult * current_atr)
            proposed_sl = max(proposed_sl, trail_sl)
            return max(current_sl, proposed_sl)
        else:
            profit_atr = (entry_price - lowest_low_since_tp1) / current_atr
            for threshold, lock_atr in sorted(self.config.ratchet_tiers, key=lambda x: x[0]):
                if profit_atr >= threshold:
                    tier_sl = entry_price - (lock_atr * current_atr)
                    proposed_sl = min(proposed_sl, tier_sl)

            trail_mult = getattr(self.config, "runner_trailing_atr_mult", 2.0)
            trail_sl = lowest_low_since_tp1 + (trail_mult * current_atr)
            proposed_sl = min(proposed_sl, trail_sl)
            return min(current_sl, proposed_sl)
