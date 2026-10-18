"""Sequential, lookahead-free backtester orchestrating the NNFX system."""

from typing import Optional, Dict, Any
import pandas as pd
from nnfx.config import BacktestConfig
from nnfx.indicator_engine import IndicatorEngine
from nnfx.algorithm import NNFXAlgorithm, SignalType
from nnfx.execution import PortfolioTracker
from nnfx.metrics import MetricsCalculator, PerformanceMetrics


class Backtester:
    """
    Drives the candle-by-candle simulation loop for NNFX trading systems.
    Guarantees zero lookahead bias by:
    1. Evaluating indicator signals strictly at bar close t.
    2. Filling entry orders at the open of bar t+1.
    3. Simulating intrabar price dynamics (SL, TP1, Trailing SL) on bar t+1.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.indicator_engine = IndicatorEngine(self.config.indicators)
        self.algorithm = NNFXAlgorithm(self.config.indicators, self.config.risk)
        self.portfolio = PortfolioTracker(
            initial_capital=self.config.initial_capital,
            risk_config=self.config.risk,
            exec_config=self.config.execution,
        )

    def run(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        enriched_df: Optional[pd.DataFrame] = None,
    ) -> PerformanceMetrics:
        """
        Executes backtest over the provided OHLCV dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Validated OHLCV DataFrame with DatetimeIndex.
        verbose : bool
            If True, prints execution milestones and trade events.
        enriched_df : pd.DataFrame, optional
            Precomputed enriched DataFrame with indicators (speeds up grid search).

        Returns
        -------
        PerformanceMetrics
            Complete statistical performance assessment.
        """
        # 1. Compute all indicators in vectorized pipeline (if not precomputed)
        if enriched_df is None:
            enriched_df = self.indicator_engine.compute_all(df)

        # 2. Warmup window (skip initial bars where rolling indicators are NaN)
        warmup_bars = max(self.config.indicators.baseline_period, 50)
        n_bars = len(enriched_df)

        if n_bars <= warmup_bars + 5:
            raise ValueError(
                f"Insufficient historical bars ({n_bars}). Needs at least {warmup_bars + 10} bars for indicator initialization."
            )

        if verbose:
            print(f"[*] Starting NNFX Backtest | {n_bars} bars | Capital: ${self.config.initial_capital:,.2f}")
            print(f"[*] Warmup window: {warmup_bars} bars | Simulating bars {warmup_bars} to {n_bars - 1}...")

        # 3. Bar-by-bar sequential simulation loop
        for t in range(warmup_bars, n_bars - 1):
            curr_ts = enriched_df.index[t]
            active_dir = self.portfolio.active_trade.direction if self.portfolio.active_trade else 0
            bars_since_loss = self.portfolio.get_bars_since_last_sl(t)

            # Step A: Evaluate rules on close of bar t
            decision = self.algorithm.evaluate_bar(
                df=enriched_df,
                current_idx=t,
                active_position_dir=active_dir,
                bars_since_last_loss=bars_since_loss,
            )

            # Step B: Advance to bar t+1
            next_row = enriched_df.iloc[t + 1]
            next_ts = enriched_df.index[t + 1]
            next_open = float(next_row["open"])
            next_high = float(next_row["high"])
            next_low = float(next_row["low"])
            next_close = float(next_row["close"])
            next_atr = float(next_row["atr"])

            # Step C: If entry signal fired at bar t close, execute fill at bar t+1 open
            if decision.is_valid and self.portfolio.active_trade is None:
                entry_trade = self.portfolio.open_trade(
                    direction=decision.direction,
                    market_open_price=next_open,
                    current_atr=float(enriched_df.iloc[t]["atr"]),
                    timestamp=next_ts,
                    bar_idx=t + 1,
                )
                if verbose:
                    side = "BUY" if decision.direction == 1 else "SELL"
                    print(
                        f"  [ENTRY] #{entry_trade.trade_id} {side} @ {entry_trade.entry_price:.5f} on {next_ts} "
                        f"| SL: {entry_trade.specs.initial_sl:.5f} | TP1: {entry_trade.specs.tp1_price:.5f} "
                        f"| Units: {entry_trade.specs.total_units:.2f}"
                    )

            # Step D: Process intrabar mechanics for bar t+1
            exit_flag = decision.exit_triggered
            self.portfolio.process_bar(
                bar_idx=t + 1,
                timestamp=next_ts,
                open_price=next_open,
                high_price=next_high,
                low_price=next_low,
                close_price=next_close,
                current_atr=next_atr,
                exit_indicator_signal=exit_flag,
            )

        # 4. Close any open position at end of simulation
        final_idx = n_bars - 1
        final_close = float(enriched_df.iloc[final_idx]["close"])
        final_ts = enriched_df.index[final_idx]
        self.portfolio.force_close_at_end(final_close=final_close, timestamp=final_ts, bar_idx=final_idx)

        # 5. Compute performance metrics
        metrics = MetricsCalculator.calculate(
            initial_capital=self.config.initial_capital,
            completed_trades=self.portfolio.completed_trades,
            equity_curve=self.portfolio.equity_curve,
            annualization_factor=self.config.annualization_factor,
        )

        return metrics
