"""Portfolio tracking, order execution simulation, and trade journaling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from nnfx.config import ExecutionConfig, RiskConfig
from nnfx.risk_manager import PositionSpecs, RiskManager


class ExitReason(Enum):
    INITIAL_SL = "INITIAL_SL"
    TP1_HIT = "TP1_HIT"
    BREAKEVEN_SL = "BREAKEVEN_SL"
    TRAILING_SL = "TRAILING_SL"
    EXIT_INDICATOR = "EXIT_INDICATOR"
    END_OF_DATA = "END_OF_DATA"


@dataclass
class TrancheRecord:
    """Audit record for a single closed tranche."""
    trade_id: int
    tranche_name: str               # "TP1_Tranche" or "Runner_Tranche"
    direction: int                  # +1 Long, -1 Short
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    units: float
    pnl_dollars: float
    pnl_pips: float
    exit_reason: ExitReason
    holding_bars: int


@dataclass
class CompletedTrade:
    """Audit record for a completed multi-tranche trade lifecycle."""
    trade_id: int
    direction: int
    entry_time: Any
    entry_price: float
    exit_time: Any
    initial_sl: float
    tp1_price: float
    total_units: float
    tp1_hit: bool
    total_pnl_dollars: float
    total_pnl_pips: float
    return_pct: float
    tranches: List[TrancheRecord] = field(default_factory=list)


class ActiveTrade:
    """Tracks state and intrabar progression of an active two-tranche NNFX position."""

    def __init__(
        self,
        trade_id: int,
        specs: PositionSpecs,
        entry_time: Any,
        entry_bar_idx: int,
    ):
        self.trade_id = trade_id
        self.specs = specs
        self.direction = specs.direction
        self.entry_time = entry_time
        self.entry_bar_idx = entry_bar_idx
        self.entry_price = specs.entry_price
        self.current_sl = specs.initial_sl
        self.tp1_price = specs.tp1_price

        # Tranche state
        self.t1_units = specs.tranche_1_units
        self.t2_units = specs.tranche_2_units
        self.t1_open = True
        self.t2_open = True
        self.tp1_hit = False

        # Runner tracking
        self.highest_since_entry = specs.entry_price
        self.lowest_since_entry = specs.entry_price
        self.highest_since_tp1 = specs.entry_price
        self.lowest_since_tp1 = specs.entry_price

        self.closed_tranches: List[TrancheRecord] = []

    def update_extremes(self, high: float, low: float):
        """Updates high/low extremes for trailing logic."""
        self.highest_since_entry = max(self.highest_since_entry, high)
        self.lowest_since_entry = min(self.lowest_since_entry, low)
        if self.tp1_hit:
            self.highest_since_tp1 = max(self.highest_since_tp1, high)
            self.lowest_since_tp1 = min(self.lowest_since_tp1, low)

    @property
    def is_fully_closed(self) -> bool:
        return (not self.t1_open) and (not self.t2_open)


class PortfolioTracker:
    """
    Manages portfolio cash, open positions, order execution with spread & slippage,
    two-tranche lifecycle updates, and complete trade journaling.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_config: Optional[RiskConfig] = None,
        exec_config: Optional[ExecutionConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.realized_pnl = 0.0
        self.risk_config = risk_config or RiskConfig()
        self.exec_config = exec_config or ExecutionConfig()
        self.risk_manager = RiskManager(self.risk_config)

        self.active_trade: Optional[ActiveTrade] = None
        self.trade_counter = 0
        self.completed_trades: List[CompletedTrade] = []
        self.all_tranche_records: List[TrancheRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.last_sl_bar_idx: Optional[int] = None

    def get_bars_since_last_sl(self, current_bar_idx: int) -> int:
        """Returns number of bars elapsed since the most recent initial stop loss hit."""
        if self.last_sl_bar_idx is None:
            return 999_999
        return max(0, current_bar_idx - self.last_sl_bar_idx)

    @property
    def current_equity(self) -> float:
        """Returns cash plus unrealized PnL of active tranches."""
        return self.cash

    def open_trade(
        self,
        direction: int,
        market_open_price: float,
        current_atr: float,
        timestamp: Any,
        bar_idx: int,
    ) -> ActiveTrade:
        """
        Executes a new two-tranche trade at the market open with spread and slippage.
        """
        if self.active_trade is not None:
            raise RuntimeError("Cannot open trade while another position is active in single-asset mode.")

        # Simulate spread and slippage
        cost_adjustment = (self.exec_config.spread_pips / 2.0) + self.exec_config.slippage_pips
        if direction == 1:
            exec_price = market_open_price + cost_adjustment
        else:
            exec_price = market_open_price - cost_adjustment

        # Calculate position sizing and targets
        specs = self.risk_manager.calculate_position(
            current_equity=self.current_equity,
            entry_price=exec_price,
            atr=current_atr,
            direction=direction,
        )

        self.trade_counter += 1
        self.active_trade = ActiveTrade(
            trade_id=self.trade_counter,
            specs=specs,
            entry_time=timestamp,
            entry_bar_idx=bar_idx,
        )
        return self.active_trade

    def process_bar(
        self,
        bar_idx: int,
        timestamp: Any,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        current_atr: float,
        exit_indicator_signal: bool,
    ):
        """
        Evaluates active trade against the current bar's intrabar price action.
        Follows NNFX rules:
        - Check initial SL.
        - Check TP1. If hit, close Tranche 1 and move Runner SL to Breakeven.
        - Check Runner SL (Breakeven or Trailed).
        - Check Exit Indicator signal for Runner.
        """
        if self.active_trade is None:
            # Record equity snapshot
            self.equity_curve.append({
                "timestamp": timestamp,
                "bar_idx": bar_idx,
                "cash": self.cash,
                "equity": self.cash,
                "drawdown_pct": 0.0,
            })
            return

        trade = self.active_trade
        trade.update_extremes(high_price, low_price)
        holding_bars = bar_idx - trade.entry_bar_idx

        # Evaluate Long Position
        if trade.direction == 1:
            self._process_long_bar(
                trade=trade,
                bar_idx=bar_idx,
                timestamp=timestamp,
                open_p=open_price,
                high_p=high_price,
                low_p=low_price,
                close_p=close_price,
                current_atr=current_atr,
                exit_signal=exit_indicator_signal,
                holding_bars=holding_bars,
            )
        # Evaluate Short Position
        else:
            self._process_short_bar(
                trade=trade,
                bar_idx=bar_idx,
                timestamp=timestamp,
                open_p=open_price,
                high_p=high_price,
                low_p=low_price,
                close_p=close_price,
                current_atr=current_atr,
                exit_signal=exit_indicator_signal,
                holding_bars=holding_bars,
            )

        # If trade is fully closed, archive it
        if trade.is_fully_closed:
            self._finalize_trade(trade, timestamp)
            self.active_trade = None

        # Record equity snapshot
        unrealized = self._calculate_unrealized_pnl(close_price)
        current_eq = self.cash + unrealized
        self.equity_curve.append({
            "timestamp": timestamp,
            "bar_idx": bar_idx,
            "cash": self.cash,
            "equity": current_eq,
        })

    def _process_long_bar(
        self,
        trade: ActiveTrade,
        bar_idx: int,
        timestamp: Any,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        current_atr: float,
        exit_signal: bool,
        holding_bars: int,
    ):
        """Intrabar evaluation for Long positions."""
        # 1. Check if Stop Loss hit before TP1
        if not trade.tp1_hit:
            # Check conservative conflict: Did bar hit SL first?
            hit_sl = low_p <= trade.current_sl
            hit_tp1 = high_p >= trade.tp1_price

            if hit_sl and hit_tp1:
                # Ambiguous intrabar order: Conservative assumption: Check open distance
                if abs(open_p - trade.current_sl) <= abs(open_p - trade.tp1_price):
                    self._close_tranche(trade, "TP1_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                    self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                    return
                else:
                    # TP1 hit first
                    self._handle_long_tp1_hit(trade, timestamp, holding_bars, current_atr)
                    # Check if Runner hits stop later in the same bar
                    if low_p <= trade.current_sl:
                        reason = ExitReason.BREAKEVEN_SL if abs(trade.current_sl - trade.entry_price) < 1e-6 else ExitReason.TRAILING_SL
                        self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, reason, holding_bars)
                    return

            elif hit_sl:
                # Stop loss hit on both tranches
                self._close_tranche(trade, "TP1_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                return

            elif hit_tp1:
                self._handle_long_tp1_hit(trade, timestamp, holding_bars, current_atr)

        # 2. If TP1 was already hit (or hit this bar and runner still open), manage Runner
        if trade.tp1_hit and trade.t2_open:
            # Check if Runner SL was hit
            if low_p <= trade.current_sl:
                reason = ExitReason.BREAKEVEN_SL if abs(trade.current_sl - trade.entry_price) < 1e-6 else ExitReason.TRAILING_SL
                self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, reason, holding_bars)
                return

            # Check Exit Indicator
            if exit_signal:
                # Exit runner at market close price
                self._close_tranche(trade, "Runner_Tranche", close_p, timestamp, ExitReason.EXIT_INDICATOR, holding_bars)
                return

            # Update stop for runner (Dynamic Ratchet or Standard Trailing)
            if getattr(self.risk_config, "profit_ratchet_enabled", False):
                new_sl = self.risk_manager.calculate_ratchet_stop(
                    direction=1,
                    entry_price=trade.entry_price,
                    current_sl=trade.current_sl,
                    highest_high_since_tp1=trade.highest_since_tp1,
                    lowest_low_since_tp1=trade.lowest_since_tp1,
                    current_atr=current_atr,
                )
            else:
                new_sl = self.risk_manager.calculate_trailing_stop(
                    direction=1,
                    current_sl=trade.current_sl,
                    highest_high_since_tp1=trade.highest_since_tp1,
                    lowest_low_since_tp1=trade.lowest_since_tp1,
                    current_atr=current_atr,
                )
            trade.current_sl = new_sl

    def _process_short_bar(
        self,
        trade: ActiveTrade,
        bar_idx: int,
        timestamp: Any,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        current_atr: float,
        exit_signal: bool,
        holding_bars: int,
    ):
        """Intrabar evaluation for Short positions."""
        # 1. Check if Stop Loss hit before TP1
        if not trade.tp1_hit:
            hit_sl = high_p >= trade.current_sl
            hit_tp1 = low_p <= trade.tp1_price

            if hit_sl and hit_tp1:
                # Ambiguous: check open distance
                if abs(open_p - trade.current_sl) <= abs(open_p - trade.tp1_price):
                    self._close_tranche(trade, "TP1_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                    self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                    return
                else:
                    self._handle_short_tp1_hit(trade, timestamp, holding_bars, current_atr)
                    if high_p >= trade.current_sl:
                        reason = ExitReason.BREAKEVEN_SL if abs(trade.current_sl - trade.entry_price) < 1e-6 else ExitReason.TRAILING_SL
                        self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, reason, holding_bars)
                    return

            elif hit_sl:
                self._close_tranche(trade, "TP1_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, ExitReason.INITIAL_SL, holding_bars)
                return

            elif hit_tp1:
                self._handle_short_tp1_hit(trade, timestamp, holding_bars, current_atr)

        # 2. If TP1 hit, manage Runner
        if trade.tp1_hit and trade.t2_open:
            if high_p >= trade.current_sl:
                reason = ExitReason.BREAKEVEN_SL if abs(trade.current_sl - trade.entry_price) < 1e-6 else ExitReason.TRAILING_SL
                self._close_tranche(trade, "Runner_Tranche", trade.current_sl, timestamp, reason, holding_bars)
                return

            if exit_signal:
                self._close_tranche(trade, "Runner_Tranche", close_p, timestamp, ExitReason.EXIT_INDICATOR, holding_bars)
                return

            # Update stop for runner (Dynamic Ratchet or Standard Trailing)
            if getattr(self.risk_config, "profit_ratchet_enabled", False):
                new_sl = self.risk_manager.calculate_ratchet_stop(
                    direction=-1,
                    entry_price=trade.entry_price,
                    current_sl=trade.current_sl,
                    highest_high_since_tp1=trade.highest_since_tp1,
                    lowest_low_since_tp1=trade.lowest_since_tp1,
                    current_atr=current_atr,
                )
            else:
                new_sl = self.risk_manager.calculate_trailing_stop(
                    direction=-1,
                    current_sl=trade.current_sl,
                    highest_high_since_tp1=trade.highest_since_tp1,
                    lowest_low_since_tp1=trade.lowest_since_tp1,
                    current_atr=current_atr,
                )
            trade.current_sl = new_sl

    def _handle_long_tp1_hit(self, trade: ActiveTrade, timestamp: Any, holding_bars: int, current_atr: float = 0.0):
        """Executes TP1 fill and moves Runner SL to Breakeven (or buffered stop) for Long trade."""
        self._close_tranche(trade, "TP1_Tranche", trade.tp1_price, timestamp, ExitReason.TP1_HIT, holding_bars)
        trade.tp1_hit = True
        if self.risk_config.move_to_be_on_tp1:
            if getattr(self.risk_config, "profit_ratchet_enabled", False):
                buffer = getattr(self.risk_config, "runner_buffer_atr", 0.5) * current_atr
                trade.current_sl = max(trade.current_sl, trade.entry_price - buffer)
            else:
                trade.current_sl = trade.entry_price  # Breakeven stop

    def _handle_short_tp1_hit(self, trade: ActiveTrade, timestamp: Any, holding_bars: int, current_atr: float = 0.0):
        """Executes TP1 fill and moves Runner SL to Breakeven (or buffered stop) for Short trade."""
        self._close_tranche(trade, "TP1_Tranche", trade.tp1_price, timestamp, ExitReason.TP1_HIT, holding_bars)
        trade.tp1_hit = True
        if self.risk_config.move_to_be_on_tp1:
            if getattr(self.risk_config, "profit_ratchet_enabled", False):
                buffer = getattr(self.risk_config, "runner_buffer_atr", 0.5) * current_atr
                trade.current_sl = min(trade.current_sl, trade.entry_price + buffer)
            else:
                trade.current_sl = trade.entry_price  # Breakeven stop

    def _close_tranche(
        self,
        trade: ActiveTrade,
        tranche_name: str,
        exit_price: float,
        timestamp: Any,
        reason: ExitReason,
        holding_bars: int,
    ):
        """Calculates PnL for a tranche and credit cash balance."""
        if tranche_name == "TP1_Tranche":
            units = trade.t1_units
            trade.t1_open = False
        else:
            units = trade.t2_units
            trade.t2_open = False

        if trade.direction == 1:
            pnl_dollars = (exit_price - trade.entry_price) * units
            pnl_pips = (exit_price - trade.entry_price) * 10_000
        else:
            pnl_dollars = (trade.entry_price - exit_price) * units
            pnl_pips = (trade.entry_price - exit_price) * 10_000

        self.cash += pnl_dollars
        self.realized_pnl += pnl_dollars

        record = TrancheRecord(
            trade_id=trade.trade_id,
            tranche_name=tranche_name,
            direction=trade.direction,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            exit_time=timestamp,
            exit_price=exit_price,
            units=units,
            pnl_dollars=pnl_dollars,
            pnl_pips=pnl_pips,
            exit_reason=reason,
            holding_bars=holding_bars,
        )
        trade.closed_tranches.append(record)
        self.all_tranche_records.append(record)

        if reason == ExitReason.INITIAL_SL:
            self.last_sl_bar_idx = trade.entry_bar_idx + holding_bars

    def _finalize_trade(self, trade: ActiveTrade, exit_time: Any):
        """Aggregates both tranches into a CompletedTrade record."""
        total_pnl = sum(t.pnl_dollars for t in trade.closed_tranches)
        total_pips = sum(t.pnl_pips for t in trade.closed_tranches) / 2.0
        return_pct = (total_pnl / self.initial_capital) * 100.0

        completed = CompletedTrade(
            trade_id=trade.trade_id,
            direction=trade.direction,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            exit_time=exit_time,
            initial_sl=trade.specs.initial_sl,
            tp1_price=trade.specs.tp1_price,
            total_units=trade.specs.total_units,
            tp1_hit=trade.tp1_hit,
            total_pnl_dollars=total_pnl,
            total_pnl_pips=total_pips,
            return_pct=return_pct,
            tranches=list(trade.closed_tranches),
        )
        self.completed_trades.append(completed)

    def force_close_at_end(self, final_close: float, timestamp: Any, bar_idx: int):
        """Closes any remaining active positions at the final bar close."""
        if self.active_trade is not None:
            trade = self.active_trade
            holding_bars = bar_idx - trade.entry_bar_idx
            if trade.t1_open:
                self._close_tranche(trade, "TP1_Tranche", final_close, timestamp, ExitReason.END_OF_DATA, holding_bars)
            if trade.t2_open:
                self._close_tranche(trade, "Runner_Tranche", final_close, timestamp, ExitReason.END_OF_DATA, holding_bars)
            self._finalize_trade(trade, timestamp)
            self.active_trade = None

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculates mark-to-market unrealized PnL of open tranches."""
        if self.active_trade is None:
            return 0.0
        trade = self.active_trade
        unrealized = 0.0
        if trade.t1_open:
            if trade.direction == 1:
                unrealized += (current_price - trade.entry_price) * trade.t1_units
            else:
                unrealized += (trade.entry_price - current_price) * trade.t1_units
        if trade.t2_open:
            if trade.direction == 1:
                unrealized += (current_price - trade.entry_price) * trade.t2_units
            else:
                unrealized += (trade.entry_price - current_price) * trade.t2_units
        return unrealized
