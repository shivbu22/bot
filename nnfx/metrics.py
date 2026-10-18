"""Institutional quantitative performance analytics and metrics computation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate
from nnfx.execution import CompletedTrade, TrancheRecord


@dataclass
class PerformanceMetrics:
    """Comprehensive statistical metrics of a completed backtest."""
    initial_capital: float
    final_equity: float
    total_net_profit: float
    total_return_pct: float
    cagr_pct: float
    exposure_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    tp1_hit_rate_pct: float
    runner_win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    avg_trade_dollars: float
    avg_win_dollars: float
    avg_loss_dollars: float
    win_loss_ratio: float
    expectancy_dollars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_bars: float


class MetricsCalculator:
    """Calculates full institutional risk/return metrics from trades and equity curves."""

    @classmethod
    def calculate(
        cls,
        initial_capital: float,
        completed_trades: List[CompletedTrade],
        equity_curve: List[Dict[str, Any]],
        annualization_factor: int = 252,
    ) -> PerformanceMetrics:
        """
        Computes all quantitative performance metrics.
        """
        eq_df = pd.DataFrame(equity_curve)
        if eq_df.empty:
            final_equity = initial_capital
        else:
            final_equity = float(eq_df["equity"].iloc[-1])

        total_net_profit = final_equity - initial_capital
        total_return_pct = (total_net_profit / initial_capital) * 100.0

        # Drawdown calculation
        if not eq_df.empty:
            eq_series = eq_df["equity"]
            running_peak = eq_series.cummax()
            dd_dollars = running_peak - eq_series
            dd_pct = (dd_dollars / running_peak) * 100.0
            max_dd_dollars = float(dd_dollars.max())
            max_dd_pct = float(dd_pct.max())
        else:
            max_dd_dollars = 0.0
            max_dd_pct = 0.0

        # Trade stats
        total_trades = len(completed_trades)
        pnls = [t.total_pnl_dollars for t in completed_trades]
        wins = [p for p in pnls if p > 1e-4]
        losses = [p for p in pnls if p < -1e-4]
        breakevens = [p for p in pnls if abs(p) <= 1e-4]

        winning_trades = len(wins)
        losing_trades = len(losses)
        breakeven_trades = len(breakevens)

        win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        # TP1 hit rate
        tp1_hits = sum(1 for t in completed_trades if t.tp1_hit)
        tp1_hit_rate_pct = (tp1_hits / total_trades * 100.0) if total_trades > 0 else 0.0

        # Runner profitability
        runner_wins = 0
        total_runners = 0
        holding_bars_list = []
        for t in completed_trades:
            for tr in t.tranches:
                holding_bars_list.append(tr.holding_bars)
                if tr.tranche_name == "Runner_Tranche":
                    total_runners += 1
                    if tr.pnl_dollars > 1e-4:
                        runner_wins += 1

        runner_win_rate_pct = (runner_wins / total_runners * 100.0) if total_runners > 0 else 0.0
        avg_holding_bars = float(np.mean(holding_bars_list)) if holding_bars_list else 0.0

        # Profit Factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        avg_trade_dollars = float(np.mean(pnls)) if pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_loss_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 0 else 0.0

        # Mathematical Expectancy
        if total_trades > 0:
            win_prob = winning_trades / total_trades
            loss_prob = losing_trades / total_trades
            expectancy = (win_prob * avg_win) - (loss_prob * abs(avg_loss))
        else:
            expectancy = 0.0

        # Consecutive streaks
        max_consec_wins = 0
        max_consec_losses = 0
        curr_wins = 0
        curr_losses = 0
        for p in pnls:
            if p > 1e-4:
                curr_wins += 1
                curr_losses = 0
                max_consec_wins = max(max_consec_wins, curr_wins)
            elif p < -1e-4:
                curr_losses += 1
                curr_wins = 0
                max_consec_losses = max(max_consec_losses, curr_losses)
            else:
                curr_wins = 0
                curr_losses = 0

        # CAGR & Risk-adjusted returns from equity curve
        if not eq_df.empty and len(eq_df) > 1:
            total_bars = len(eq_df)
            years = total_bars / annualization_factor
            if years > 0 and final_equity > 0:
                cagr_pct = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
            else:
                cagr_pct = 0.0

            # Daily / bar returns
            bar_returns = eq_df["equity"].pct_change().dropna()
            mean_ret = bar_returns.mean()
            std_ret = bar_returns.std()

            if std_ret > 0:
                sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(annualization_factor))
            else:
                sharpe_ratio = 0.0

            # Sortino
            downside_returns = bar_returns[bar_returns < 0]
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = float((mean_ret / downside_std) * np.sqrt(annualization_factor))
            else:
                sortino_ratio = 0.0

            # Calmar
            if max_dd_pct > 0:
                calmar_ratio = cagr_pct / max_dd_pct
            else:
                calmar_ratio = 0.0

            # Exposure (% time in trade)
            bars_in_trade = sum(1 for t in completed_trades for tr in t.tranches if tr.tranche_name == "TP1_Tranche")
            exposure_pct = min(100.0, (bars_in_trade * avg_holding_bars / total_bars) * 100.0) if total_bars > 0 else 0.0
        else:
            cagr_pct = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
            exposure_pct = 0.0

        return PerformanceMetrics(
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_net_profit=total_net_profit,
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            exposure_pct=exposure_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate_pct=win_rate_pct,
            tp1_hit_rate_pct=tp1_hit_rate_pct,
            runner_win_rate_pct=runner_win_rate_pct,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown_dollars=max_dd_dollars,
            max_drawdown_pct=max_dd_pct,
            avg_trade_dollars=avg_trade_dollars,
            avg_win_dollars=avg_win,
            avg_loss_dollars=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expectancy_dollars=expectancy,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_holding_bars=avg_holding_bars,
        )

    @staticmethod
    def format_summary_table(metrics: PerformanceMetrics) -> str:
        """Returns an ASCII table of strategy performance metrics."""
        table_data = [
            ["Initial Capital", f"${metrics.initial_capital:,.2f}"],
            ["Final Account Equity", f"${metrics.final_equity:,.2f}"],
            ["Net Profit ($)", f"${metrics.total_net_profit:+,.2f}"],
            ["Total Return (%)", f"{metrics.total_return_pct:+.2f}%"],
            ["Annualized Return (CAGR)", f"{metrics.cagr_pct:+.2f}%"],
            ["Max Drawdown ($)", f"${metrics.max_drawdown_dollars:,.2f}"],
            ["Max Drawdown (%)", f"{metrics.max_drawdown_pct:.2f}%"],
            ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
            ["Sortino Ratio", f"{metrics.sortino_ratio:.2f}"],
            ["Calmar Ratio", f"{metrics.calmar_ratio:.2f}"],
            ["Profit Factor", f"{metrics.profit_factor:.2f}"],
            ["Total Completed Trades", f"{metrics.total_trades}"],
            ["Winning Trades / Losing / BE", f"{metrics.winning_trades} / {metrics.losing_trades} / {metrics.breakeven_trades}"],
            ["Overall Win Rate", f"{metrics.win_rate_pct:.1f}%"],
            ["TP1 Hit Rate (Scale 1)", f"{metrics.tp1_hit_rate_pct:.1f}%"],
            ["Runner Win Rate (Scale 2)", f"{metrics.runner_win_rate_pct:.1f}%"],
            ["Average Trade PnL", f"${metrics.avg_trade_dollars:+,.2f}"],
            ["Average Win", f"${metrics.avg_win_dollars:+,.2f}"],
            ["Average Loss", f"${metrics.avg_loss_dollars:+,.2f}"],
            ["Win / Loss Ratio", f"{metrics.win_loss_ratio:.2f}"],
            ["Trade Expectancy", f"${metrics.expectancy_dollars:+,.2f}"],
            ["Max Consecutive Wins / Losses", f"{metrics.max_consecutive_wins} / {metrics.max_consecutive_losses}"],
            ["Average Holding Duration", f"{metrics.avg_holding_bars:.1f} bars"],
        ]
        return tabulate(table_data, headers=["NNFX Metric", "Value"], tablefmt="grid")

    @staticmethod
    def format_trades_table(completed_trades: List[CompletedTrade], max_rows: int = 25) -> str:
        """Returns a formatted table of individual closed trades."""
        if not completed_trades:
            return "No trades were completed during this period."

        rows = []
        for t in completed_trades[-max_rows:]:
            dir_str = "LONG" if t.direction == 1 else "SHORT"
            tp1_status = "YES" if t.tp1_hit else "NO"
            exit_reasons = ", ".join(tr.exit_reason.value for tr in t.tranches)
            rows.append([
                t.trade_id,
                dir_str,
                str(t.entry_time)[:16],
                f"{t.entry_price:.5f}",
                str(t.exit_time)[:16],
                f"{t.total_units:.2f}",
                tp1_status,
                f"${t.total_pnl_dollars:+,.2f}",
                f"{t.return_pct:+.2f}%",
                exit_reasons,
            ])

        headers = ["ID", "Side", "Entry Time", "Entry Px", "Exit Time", "Units", "TP1 Hit", "PnL ($)", "Return %", "Tranche Exits"]
        table = tabulate(rows, headers=headers, tablefmt="simple")
        if len(completed_trades) > max_rows:
            table += f"\n... (Showing last {max_rows} of {len(completed_trades)} completed trades)"
        return table
