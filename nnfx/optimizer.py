"""Strategy Optimizer performing grid search and out-of-sample walk-forward validation."""

from dataclasses import dataclass, replace
from typing import List, Dict, Any, Tuple
import itertools
import pandas as pd
from tabulate import tabulate

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig
from nnfx.backtester import Backtester
from nnfx.metrics import PerformanceMetrics


@dataclass
class OptimizationResult:
    """Individual parameter evaluation result."""
    params: Dict[str, Any]
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics
    score: float


class StrategyOptimizer:
    """
    Grid search and train/test out-of-sample validation engine for NNFX.
    Evaluates candidate parameter configurations to maximize precision,
    win rate, and risk-adjusted return without overfitting.
    """

    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config

    def optimize(
        self,
        df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        train_ratio: float = 0.70,
        top_n: int = 5,
        verbose: bool = True,
    ) -> List[OptimizationResult]:
        """
        Runs grid search across parameter combinations with train/test split.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        param_grid : Dict[str, List[Any]]
            Mapping of parameter names to lists of candidate values.
        train_ratio : float
            Fraction of historical data used for in-sample training (default: 0.70).
        top_n : int
            Number of top results to return.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        List[OptimizationResult]
            Top N parameter sets sorted by performance score.
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        min_warmup = max(self.base_config.indicators.baseline_period + 20, 220)
        if len(train_df) < min_warmup or len(test_df) < min_warmup:
            raise ValueError(
                f"Insufficient data for train/test split ({len(train_df)} train, {len(test_df)} test). "
                f"Need at least {min_warmup} bars for both train and test partitions to satisfy baseline warmup."
            )

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if verbose:
            print(f"[*] Starting Strategy Optimization: {len(combinations)} parameter sets", flush=True)
            print(f"[*] In-Sample: {len(train_df):,} bars | Out-of-Sample: {len(test_df):,} bars", flush=True)

        results: List[OptimizationResult] = []

        for idx, p in enumerate(combinations):
            # Create configured backtest instance
            cfg = self._build_config_from_params(p)

            # 1. Run In-Sample
            bt_train = Backtester(config=cfg)
            train_metrics = bt_train.run(train_df, verbose=False)

            # 2. Run Out-of-Sample
            bt_test = Backtester(config=cfg)
            test_metrics = bt_test.run(test_df, verbose=False)

            # Scoring: Combines Win Rate, Profit Factor, and Sharpe Ratio
            # Score heavily penalizes zero trades or negative returns
            if train_metrics.total_trades >= 3 and train_metrics.profit_factor > 1.0:
                score = (
                    (train_metrics.win_rate_pct * 0.4) +
                    (min(train_metrics.profit_factor, 5.0) * 15.0) +
                    (max(train_metrics.sharpe_ratio, 0.0) * 10.0)
                )
            else:
                score = -100.0

            results.append(
                OptimizationResult(
                    params=p,
                    in_sample_metrics=train_metrics,
                    out_of_sample_metrics=test_metrics,
                    score=score,
                )
            )

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)

        if verbose:
            print(f"[+] Optimization finished. Analyzed {len(results)} configurations.")
            self._print_top_results(results[:top_n])

        return results[:top_n]

    def _build_config_from_params(self, params: Dict[str, Any]) -> BacktestConfig:
        """Constructs a BacktestConfig replacing specified parameters."""
        risk_kwargs = {}
        ind_kwargs = {}

        for k, v in params.items():
            if hasattr(self.base_config.risk, k):
                risk_kwargs[k] = v
            elif hasattr(self.base_config.indicators, k):
                ind_kwargs[k] = v

        new_risk = replace(self.base_config.risk, **risk_kwargs) if risk_kwargs else self.base_config.risk
        new_ind = replace(self.base_config.indicators, **ind_kwargs) if ind_kwargs else self.base_config.indicators

        return replace(self.base_config, risk=new_risk, indicators=new_ind)

    @staticmethod
    def _print_top_results(top_results: List[OptimizationResult]):
        """Prints a formatted ASCII summary of top optimized parameter sets."""
        rows = []
        for rank, res in enumerate(top_results, 1):
            p_str = ", ".join(f"{k}={v}" for k, v in res.params.items())
            rows.append([
                f"#{rank}",
                f"{res.in_sample_metrics.win_rate_pct:.1f}%",
                f"{res.in_sample_metrics.profit_factor:.2f}",
                f"{res.out_of_sample_metrics.win_rate_pct:.1f}%",
                f"{res.out_of_sample_metrics.profit_factor:.2f}",
                f"{res.out_of_sample_metrics.max_drawdown_pct:.1f}%",
                p_str,
            ])

        headers = ["Rank", "Train Win%", "Train PF", "Test Win%", "Test PF", "Test MaxDD%", "Parameters"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
