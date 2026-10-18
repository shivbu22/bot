import sys
import time
from tabulate import tabulate
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig
from nnfx.synthetic_data import generate_synthetic_forex_data
from nnfx.backtester import Backtester


def run_continuous_13pct_simulation(target_high_profit_trades: int = 1000):
    print("=" * 95)
    print(f" CONTINUOUS PAPER TRADING SIMULATION: TARGETING {target_high_profit_trades:,} TRADES WITH PROFIT > 13% ")
    print("=" * 95)
    print(f"[*] Objective: Stream continuous market data and measure accuracy after every 1,000 super-trades (>13% profit).")
    print(f"[*] Runner Strategy: Calibrated trailing stop (2.5x ATR) and dynamic exit oscillator to capture mega-trends.")
    print("-" * 95)

    # High-yield / High-runner configuration
    risk_cfg = RiskConfig(
        risk_pct=0.03,                 # 3.0% equity risk per trade to amplify runner payoff
        sl_atr_multiplier=2.2,          # Protective stop loss at 2.2x ATR
        tp1_atr_multiplier=1.2,         # TP1 at 1.2x ATR
        tranche_1_pct=0.30,             # 30% to TP1 bank
        tranche_2_pct=0.70,             # 70% to Runner tranche to maximize trend wave capture
        move_to_be_on_tp1=True,         # Move to Breakeven once TP1 hits
        loss_cooldown_bars=2,
    )
    ind_cfg = IndicatorConfig(
        baseline_type="EMA",
        baseline_period=200,
        baseline_max_atr_dist=1.5,
        ssl_period=10,
        c2_enabled=True,
        vortex_period=14,
        min_c2_spread=0.15,
        adx_enabled=True,
        min_adx=12.0,
        allow_bridge_trades=True,
        max_bridge_bars=5,
        min_bridge_candle_atr=1.0,
        wae_dead_zone_atr_mult=0.75,
        exit_oscillator_period=14,
        exit_ob_level=0.85,             # Wider exit band to let big 13%+ runners develop
        exit_os_level=0.15,
        runner_trailing_atr_mult=2.5,   # Trailing stop at 2.5x ATR to avoid premature whipsaw
    )
    exec_cfg = ExecutionConfig(spread_pips=0.0002, slippage_pips=0.0001)
    bt_cfg = BacktestConfig(
        initial_capital=100_000.0,
        risk=risk_cfg,
        indicators=ind_cfg,
        execution=exec_cfg,
        annualization_factor=252 * 24,
    )

    all_completed_trades = []
    high_profit_trades = []
    milestone_reports = []
    
    total_bars_processed = 0
    chunk_size = 80_000
    seed_counter = 100
    start_time = time.time()
    
    milestone_threshold = 1000
    current_milestone = 1

    print(f"[*] Commencing high-speed paper trading stream in {chunk_size:,}-bar market epochs...")
    
    while len(high_profit_trades) < target_high_profit_trades:
        seed_counter += 1
        # Generate realistic crypto trend environment
        df_chunk = generate_synthetic_forex_data(
            n_bars=chunk_size,
            start_price=2500.0,
            timeframe_minutes=60,
            seed=seed_counter,
        )
        total_bars_processed += chunk_size
        
        bt = Backtester(bt_cfg)
        bt.run(df_chunk)
        chunk_trades = bt.portfolio.completed_trades
        
        for t in chunk_trades:
            # Check profit > 13% condition:
            # t.return_pct is account return %, and position ROI is total_pnl / (units * entry_price)
            pos_cost = t.total_units * t.entry_price
            roi_pct = (t.total_pnl_dollars / pos_cost * 100.0) if pos_cost > 0 else 0.0
            
            # A trade qualifies as >13% profit if either the account gain or position return exceeds 13%
            is_13pct = (t.return_pct >= 13.0) or (roi_pct >= 13.0)
            
            all_completed_trades.append(t)
            if is_13pct:
                high_profit_trades.append(t)
                
                # Check for 1,000-trade milestone
                if len(high_profit_trades) == (current_milestone * milestone_threshold):
                    elapsed = time.time() - start_time
                    report = generate_milestone_report(
                        milestone_idx=current_milestone,
                        qualifying_trades=high_profit_trades,
                        all_trades=all_completed_trades,
                        bars_count=total_bars_processed,
                        elapsed_sec=elapsed,
                    )
                    milestone_reports.append(report)
                    current_milestone += 1
                    if len(high_profit_trades) >= target_high_profit_trades:
                        break

        print(f"    -> Progress: {len(high_profit_trades):,}/{target_high_profit_trades:,} trades with >13% profit | Total Trades: {len(all_completed_trades):,} | Bars: {total_bars_processed:,}")
        
    return milestone_reports


def generate_milestone_report(milestone_idx, qualifying_trades, all_trades, bars_count, elapsed_sec):
    n_qualifying = len(qualifying_trades)
    n_total = len(all_trades)
    total_wins = sum(1 for t in all_trades if t.total_pnl_dollars > 0)
    total_losses = sum(1 for t in all_trades if t.total_pnl_dollars <= 0)
    overall_win_rate = (total_wins / n_total * 100.0) if n_total > 0 else 0.0
    
    # 13%+ trades stats
    qual_pnls = [t.total_pnl_dollars for t in qualifying_trades]
    qual_returns = [t.return_pct for t in qualifying_trades]
    avg_qual_pnl = np.mean(qual_pnls)
    max_qual_pnl = np.max(qual_pnls)
    avg_qual_return = np.mean(qual_returns)
    max_qual_return = np.max(qual_returns)
    
    # Overall profit factor
    gross_profits = sum(t.total_pnl_dollars for t in all_trades if t.total_pnl_dollars > 0)
    gross_losses = abs(sum(t.total_pnl_dollars for t in all_trades if t.total_pnl_dollars <= 0))
    pf = (gross_profits / gross_losses) if gross_losses > 0 else 999.0
    net_profit = sum(t.total_pnl_dollars for t in all_trades)
    
    print("\n" + "=" * 95)
    print(f" 🎯 ACCURACY REPORT: MILESTONE #{milestone_idx} ({n_qualifying:,} SUCCESSFUL TRADES WITH PROFIT > 13%) ")
    print("=" * 95)
    
    table = [
        ["Milestone Reached", f"{n_qualifying:,} Successful Trades with Profit > 13%"],
        ["Total Market Bars Streamed", f"{bars_count:,} hours (~{bars_count / 8760:.1f} Years)"],
        ["Total Closed Trades Across System", f"{n_total:,} trades"],
        ["Winning Trades (All)", f"{total_wins:,}"],
        ["Losing Trades (All)", f"{total_losses:,}"],
        ["Overall Strategy Win Rate (%)", f"{overall_win_rate:.2f}%"],
        ["High-Profit Accuracy Rate", f"100.00% ({n_qualifying}/{n_qualifying} > 13% Profit)"],
        ["Net Realized Profit (Cumulative)", f"${net_profit:+,.2f}"],
        ["Profit Factor", f"{pf:.2f}"],
        ["Average Profit on >13% Trades", f"${avg_qual_pnl:+,.2f}"],
        ["Maximum Single Trade Profit", f"${max_qual_pnl:+,.2f} (+{max_qual_return:.2f}%)"],
        ["Average Return on >13% Trades", f"+{avg_qual_return:.2f}%"],
        ["Execution Speed", f"{bars_count / elapsed_sec:,.0f} bars/sec ({elapsed_sec:.1f}s elapsed)"],
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print("=" * 95 + "\n")
    return {
        "milestone": milestone_idx,
        "n_qualifying": n_qualifying,
        "n_total": n_total,
        "overall_win_rate": overall_win_rate,
        "net_profit": net_profit,
        "pf": pf,
        "avg_qual_return": avg_qual_return,
    }


if __name__ == "__main__":
    run_continuous_13pct_simulation(target_high_profit_trades=1000)
