"""NNFX Algorithm evaluating multi-layer trade rules, filters, and exit conditions."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd
from nnfx.config import IndicatorConfig, RiskConfig


class SignalType(Enum):
    NONE = 0
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class SignalDecision:
    """Detailed record of an evaluation at candle close t."""
    bar_index: int
    timestamp: Any
    signal: SignalType
    is_valid: bool
    direction: int                  # +1 Long, -1 Short, 0 None
    rejection_reason: Optional[str] = None
    baseline_val: float = 0.0
    baseline_dist_atr: float = 0.0
    ssl_direction: int = 0
    volume_active: bool = False
    exit_triggered: bool = False


class NNFXAlgorithm:
    """
    Evaluates multi-layer signals based on the No Nonsense Forex framework:
    1. Baseline Rule: Price must be on the correct side (Close > Baseline for Long, < Baseline for Short).
    2. Baseline Extension / Distance Filter: Distance from Baseline must NOT exceed 1.5 * ATR.
    3. Confirmation Indicator (C1): Confirms initial trend direction flip (or baseline bridge cross).
    4. Volume / Volatility Filter (WAE): Confirms volume is above explosion line and dead zone.
    5. Trend Strength Filter (ADX): Rejects entries in low-momentum consolidation.
    6. Exit Indicator: Signals early exit for the runner tranche.
    """

    def __init__(
        self,
        config: Optional[IndicatorConfig] = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.config = config or IndicatorConfig()
        self.risk_config = risk_config or RiskConfig()

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        current_idx: int,
        active_position_dir: int = 0,
        bars_since_last_loss: int = 999999,
    ) -> SignalDecision:
        """
        Evaluates candle at current_idx.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame with calculated indicators.
        current_idx : int
            Current bar index in df.
        active_position_dir : int
            Direction of currently open position (+1 Long, -1 Short, 0 None).

        Returns
        -------
        SignalDecision
            Structured decision including signal, validity, rejection reason, and exit trigger.
        """
        if current_idx < 1:
            return SignalDecision(
                bar_index=current_idx,
                timestamp=df.index[current_idx],
                signal=SignalType.NONE,
                is_valid=False,
                direction=0,
                rejection_reason="Insufficient history",
            )

        row = df.iloc[current_idx]
        prev_row = df.iloc[current_idx - 1]
        ts = df.index[current_idx]

        close = row["close"]
        baseline = row["baseline"]
        atr = row["atr"]
        baseline_dir = int(row["baseline_dir"])
        baseline_dist_atr = float(row["baseline_atr_dist"])
        ssl_dir = int(row["ssl_dir"])
        ssl_flip = int(row["ssl_flip"])
        wae_bull_vol = bool(row["wae_bull_vol"])
        wae_bear_vol = bool(row["wae_bear_vol"])

        # Check for exit signals if a position is currently open
        exit_triggered = False
        if active_position_dir == 1 and bool(row["exit_long_signal"]):
            exit_triggered = True
        elif active_position_dir == -1 and bool(row["exit_short_signal"]):
            exit_triggered = True

        # If a position is already active in single-pair mode, no new entry signal
        if active_position_dir != 0:
            return SignalDecision(
                bar_index=current_idx,
                timestamp=ts,
                signal=SignalType.NONE,
                is_valid=False,
                direction=0,
                rejection_reason="Position already open",
                baseline_val=baseline,
                baseline_dist_atr=baseline_dist_atr,
                ssl_direction=ssl_dir,
                volume_active=wae_bull_vol if active_position_dir == 1 else wae_bear_vol,
                exit_triggered=exit_triggered,
            )

        # Baseline cross detection (Bridge trade setup)
        prev_close = prev_row["close"]
        prev_baseline = prev_row["baseline"]
        baseline_cross_bull = (prev_close <= prev_baseline) and (close > baseline)
        baseline_cross_bear = (prev_close >= prev_baseline) and (close < baseline)

        c2_dir = int(row["c2_dir"]) if "c2_dir" in row else ssl_dir
        candle_valid = bool(row["candle_valid"]) if "candle_valid" in row else True
        candle_atr_ratio = float(row["candle_atr_ratio"]) if "candle_atr_ratio" in row else 1.0
        baseline_slope = float(row["baseline_slope"]) if "baseline_slope" in row else 0.0

        # Evaluate potential LONG setup (C1 Flip or Bridge Trade)
        is_bull_c1_flip = (ssl_flip == 1)
        is_bull_bridge = (baseline_cross_bull and ssl_dir == 1 and not is_bull_c1_flip)
        long_setup = is_bull_c1_flip or is_bull_bridge

        # Evaluate potential SHORT setup (C1 Flip or Bridge Trade)
        is_bear_c1_flip = (ssl_flip == -1)
        is_bear_bridge = (baseline_cross_bear and ssl_dir == -1 and not is_bear_c1_flip)
        short_setup = is_bear_c1_flip or is_bear_bridge

        if long_setup:
            # 0. Post-Loss Whipsaw Cooldown Window
            if bars_since_last_loss < self.risk_config.loss_cooldown_bars:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason=f"Post-loss whipsaw cooldown active ({bars_since_last_loss} < {self.risk_config.loss_cooldown_bars} bars)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bull_vol,
                    exit_triggered=exit_triggered,
                )

            # 0.5 Bridge Trade Precision Rules
            if is_bull_bridge:
                if not self.config.allow_bridge_trades:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.BUY,
                        is_valid=False,
                        direction=1,
                        rejection_reason="Bridge trade rejected (Strict C1 entry mode enabled)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bull_vol,
                        exit_triggered=exit_triggered,
                    )
                bars_since_flip = 9999
                ssl_flips = df["ssl_flip"].values
                for back_idx in range(current_idx - 1, max(-1, current_idx - 50), -1):
                    val = int(ssl_flips[back_idx])
                    if val == 1:
                        bars_since_flip = current_idx - back_idx
                        break
                    if val == -1:
                        break
                if bars_since_flip > self.config.max_bridge_bars:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.BUY,
                        is_valid=False,
                        direction=1,
                        rejection_reason=f"Bridge trade expired ({bars_since_flip} bars > {self.config.max_bridge_bars} max bars since C1 flip)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bull_vol,
                        exit_triggered=exit_triggered,
                    )
                if candle_atr_ratio < self.config.min_bridge_candle_atr:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.BUY,
                        is_valid=False,
                        direction=1,
                        rejection_reason=f"Bridge candle impulse too weak ({candle_atr_ratio:.2f} ATR < {self.config.min_bridge_candle_atr} ATR)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bull_vol,
                        exit_triggered=exit_triggered,
                    )

            # 1. Baseline Side Check
            if baseline_dir != 1:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason="Price is below baseline (Long rule violation)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bull_vol,
                    exit_triggered=exit_triggered,
                )

            # 2. Baseline Distance Rule
            if baseline_dist_atr > self.config.baseline_max_atr_dist:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason=f"Price extended too far from baseline ({baseline_dist_atr:.2f} ATR > {self.config.baseline_max_atr_dist} ATR)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bull_vol,
                    exit_triggered=exit_triggered,
                )

            # 3. Baseline Slope Rule (No counter-trend slope)
            if self.config.baseline_slope_filter and baseline_slope < -1e-6:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason=f"Baseline is sloping downward ({baseline_slope:.5f}) - counter-trend entry rejected",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bull_vol,
                    exit_triggered=exit_triggered,
                )

            # 4. Giant Climax Candle Filter (NNFX Climax Rule)
            if not candle_valid:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason=f"Giant exhaustion candle ({candle_atr_ratio:.2f} ATR > {self.config.max_candle_atr_ratio} ATR limit)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bull_vol,
                    exit_triggered=exit_triggered,
                )

            # 5. C2 Secondary Confirmation Filter (Vortex Agreement & Spread)
            if self.config.c2_enabled:
                if c2_dir != 1:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.BUY,
                        is_valid=False,
                        direction=1,
                        rejection_reason=f"C2 confirmation indicator is not bullish (C2 dir: {c2_dir})",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bull_vol,
                        exit_triggered=exit_triggered,
                    )
                if self.config.min_c2_spread > 0:
                    vx_plus = float(row["vortex_plus"]) if "vortex_plus" in row else 1.0
                    vx_minus = float(row["vortex_minus"]) if "vortex_minus" in row else 0.0
                    vx_spread = abs(vx_plus - vx_minus)
                    if vx_spread < self.config.min_c2_spread:
                        return SignalDecision(
                            bar_index=current_idx,
                            timestamp=ts,
                            signal=SignalType.BUY,
                            is_valid=False,
                            direction=1,
                            rejection_reason=f"C2 Vortex spread too weak ({vx_spread:.3f} < {self.config.min_c2_spread:.3f})",
                            baseline_val=baseline,
                            baseline_dist_atr=baseline_dist_atr,
                            ssl_direction=ssl_dir,
                            volume_active=wae_bull_vol,
                            exit_triggered=exit_triggered,
                        )

            # 6. Trend Strength Filter (Wilder's ADX)
            if self.config.adx_enabled:
                adx_val = float(row["adx"]) if "adx" in row else 25.0
                if adx_val < self.config.min_adx:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.BUY,
                        is_valid=False,
                        direction=1,
                        rejection_reason=f"ADX trend strength too low ({adx_val:.1f} < {self.config.min_adx:.1f})",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bull_vol,
                        exit_triggered=exit_triggered,
                    )

            # 7. Volume / Volatility Filter
            if not wae_bull_vol:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.BUY,
                    is_valid=False,
                    direction=1,
                    rejection_reason="Volume filter invalid (WAE below threshold/dead zone)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=False,
                    exit_triggered=exit_triggered,
                )

            # All filters passed!
            return SignalDecision(
                bar_index=current_idx,
                timestamp=ts,
                signal=SignalType.BUY,
                is_valid=True,
                direction=1,
                rejection_reason=None,
                baseline_val=baseline,
                baseline_dist_atr=baseline_dist_atr,
                ssl_direction=ssl_dir,
                volume_active=True,
                exit_triggered=exit_triggered,
            )

        elif short_setup:
            # 0. Post-Loss Whipsaw Cooldown Window
            if bars_since_last_loss < self.risk_config.loss_cooldown_bars:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason=f"Post-loss whipsaw cooldown active ({bars_since_last_loss} < {self.risk_config.loss_cooldown_bars} bars)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bear_vol,
                    exit_triggered=exit_triggered,
                )

            # 0.5 Bridge Trade Precision Rules
            if is_bear_bridge:
                if not self.config.allow_bridge_trades:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.SELL,
                        is_valid=False,
                        direction=-1,
                        rejection_reason="Bridge trade rejected (Strict C1 entry mode enabled)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bear_vol,
                        exit_triggered=exit_triggered,
                    )
                bars_since_flip = 9999
                ssl_flips = df["ssl_flip"].values
                for back_idx in range(current_idx - 1, max(-1, current_idx - 50), -1):
                    val = int(ssl_flips[back_idx])
                    if val == -1:
                        bars_since_flip = current_idx - back_idx
                        break
                    if val == 1:
                        break
                if bars_since_flip > self.config.max_bridge_bars:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.SELL,
                        is_valid=False,
                        direction=-1,
                        rejection_reason=f"Bridge trade expired ({bars_since_flip} bars > {self.config.max_bridge_bars} max bars since C1 flip)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bear_vol,
                        exit_triggered=exit_triggered,
                    )
                if candle_atr_ratio < self.config.min_bridge_candle_atr:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.SELL,
                        is_valid=False,
                        direction=-1,
                        rejection_reason=f"Bridge candle impulse too weak ({candle_atr_ratio:.2f} ATR < {self.config.min_bridge_candle_atr} ATR)",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bear_vol,
                        exit_triggered=exit_triggered,
                    )

            # 1. Baseline Side Check
            if baseline_dir != -1:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason="Price is above baseline (Short rule violation)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bear_vol,
                    exit_triggered=exit_triggered,
                )

            # 2. Baseline Distance Rule
            if baseline_dist_atr > self.config.baseline_max_atr_dist:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason=f"Price extended too far from baseline ({baseline_dist_atr:.2f} ATR > {self.config.baseline_max_atr_dist} ATR)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bear_vol,
                    exit_triggered=exit_triggered,
                )

            # 3. Baseline Slope Rule (No counter-trend slope)
            if self.config.baseline_slope_filter and baseline_slope > 1e-6:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason=f"Baseline is sloping upward ({baseline_slope:.5f}) - counter-trend entry rejected",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bear_vol,
                    exit_triggered=exit_triggered,
                )

            # 4. Giant Climax Candle Filter (NNFX Climax Rule)
            if not candle_valid:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason=f"Giant exhaustion candle ({candle_atr_ratio:.2f} ATR > {self.config.max_candle_atr_ratio} ATR limit)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=wae_bear_vol,
                    exit_triggered=exit_triggered,
                )

            # 5. C2 Secondary Confirmation Filter (Vortex Agreement & Spread)
            if self.config.c2_enabled:
                if c2_dir != -1:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.SELL,
                        is_valid=False,
                        direction=-1,
                        rejection_reason=f"C2 confirmation indicator is not bearish (C2 dir: {c2_dir})",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bear_vol,
                        exit_triggered=exit_triggered,
                    )
                if self.config.min_c2_spread > 0:
                    vx_plus = float(row["vortex_plus"]) if "vortex_plus" in row else 0.0
                    vx_minus = float(row["vortex_minus"]) if "vortex_minus" in row else 1.0
                    vx_spread = abs(vx_plus - vx_minus)
                    if vx_spread < self.config.min_c2_spread:
                        return SignalDecision(
                            bar_index=current_idx,
                            timestamp=ts,
                            signal=SignalType.SELL,
                            is_valid=False,
                            direction=-1,
                            rejection_reason=f"C2 Vortex spread too weak ({vx_spread:.3f} < {self.config.min_c2_spread:.3f})",
                            baseline_val=baseline,
                            baseline_dist_atr=baseline_dist_atr,
                            ssl_direction=ssl_dir,
                            volume_active=wae_bear_vol,
                            exit_triggered=exit_triggered,
                        )

            # 6. Trend Strength Filter (Wilder's ADX)
            if self.config.adx_enabled:
                adx_val = float(row["adx"]) if "adx" in row else 25.0
                if adx_val < self.config.min_adx:
                    return SignalDecision(
                        bar_index=current_idx,
                        timestamp=ts,
                        signal=SignalType.SELL,
                        is_valid=False,
                        direction=-1,
                        rejection_reason=f"ADX trend strength too low ({adx_val:.1f} < {self.config.min_adx:.1f})",
                        baseline_val=baseline,
                        baseline_dist_atr=baseline_dist_atr,
                        ssl_direction=ssl_dir,
                        volume_active=wae_bear_vol,
                        exit_triggered=exit_triggered,
                    )

            # 7. Volume / Volatility Filter
            if not wae_bear_vol:
                return SignalDecision(
                    bar_index=current_idx,
                    timestamp=ts,
                    signal=SignalType.SELL,
                    is_valid=False,
                    direction=-1,
                    rejection_reason="Volume filter invalid (WAE below threshold/dead zone)",
                    baseline_val=baseline,
                    baseline_dist_atr=baseline_dist_atr,
                    ssl_direction=ssl_dir,
                    volume_active=False,
                    exit_triggered=exit_triggered,
                )

            # All filters passed!
            return SignalDecision(
                bar_index=current_idx,
                timestamp=ts,
                signal=SignalType.SELL,
                is_valid=True,
                direction=-1,
                rejection_reason=None,
                baseline_val=baseline,
                baseline_dist_atr=baseline_dist_atr,
                ssl_direction=ssl_dir,
                volume_active=True,
                exit_triggered=exit_triggered,
            )

        # No setup triggered on this bar
        return SignalDecision(
            bar_index=current_idx,
            timestamp=ts,
            signal=SignalType.NONE,
            is_valid=False,
            direction=0,
            rejection_reason=None,
            baseline_val=baseline,
            baseline_dist_atr=baseline_dist_atr,
            ssl_direction=ssl_dir,
            volume_active=False,
            exit_triggered=exit_triggered,
        )
