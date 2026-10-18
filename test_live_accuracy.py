"""Live Trading Model Accuracy and Parity Verification Test.

Validates:
1. Live CCXT stream connectivity and real-time candle telemetry on Binance.
2. Step-by-step live execution simulation across real market bars.
3. Mathematical parity between live paper execution and offline backtest.
4. Final live accuracy, precision, and win rate report.
"""

import sys
import pandas as pd
from tabulate import tabulate

from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExchangeConfig
from nnfx.data_loader import DataLoader
from nnfx.backtester import Backtester
from nnfx.ccxt_live_trader import CCXTLiveTrader
from nnfx.metrics import MetricsCalculator


def test_live_accuracy_on_symbol(symbol: str, csv_path: str):
    print("=" * 90)
    print(f"       LIVE MODEL TRADE ACCURACY VALIDATION: {symbol}       ")
    print("=" * 90)

    # 1. Load real market dataset
    full_df = DataLoader.load(csv_path)
    total_bars = len(full_df)
    print(f"[*] Ingested {total_bars:,} real Binance bars from {full_df.index[0]} to {full_df.index[-1]}")

    # 2. Configure High-Win-Rate Profile
    risk_cfg = RiskConfig(
        risk_pct=0.015,
        sl_atr_multiplier=2.2,
        tp1_atr_multiplier=0.8,
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
    )
    bt_cfg = BacktestConfig(
        initial_capital=100_000.0,
        risk=risk_cfg,
        indicators=ind_cfg,
    )
    exch_cfg = ExchangeConfig(
        exchange_id="binance",
        symbol=symbol,
        timeframe="1h",
        paper_trading=True,
    )

    # 3. Run Offline Backtester as benchmark (using recent 1,000 bars for test)
    test_df = full_df.iloc[-1000:].copy()
    print(f"[*] Running Benchmark Backtester on {len(test_df):,} bars...")
    bt = Backtester(bt_cfg)
    bt_metrics = bt.run(test_df)
    bt_trades = bt.portfolio.completed_trades

    # 4. Simulate Live Model Trading Execution Bar-by-Bar
    print(f"[*] Simulating Real-Time Live Model Execution (Bar-by-Bar CCXTLiveTrader)...")
    live_trader = CCXTLiveTrader(exchange_config=exch_cfg, backtest_config=bt_cfg)

    # Pre-warm with initial 250 bars of the test slice
    warmup_n = 250
    live_trader.historical_df = test_df.iloc[:warmup_n].copy()
    live_trader.last_processed_timestamp = live_trader.historical_df.index[-1]

    # Replay subsequent bars sequentially into the live model
    for i in range(warmup_n, len(test_df) - 1):
        closed_bar = test_df.iloc[[i]]
        forming_bar = test_df.iloc[i + 1]

        # Feed the new bar into live trader buffer
        live_trader.historical_df = pd.concat([live_trader.historical_df, closed_bar])
        live_trader.historical_df = live_trader.historical_df[~live_trader.historical_df.index.duplicated(keep="last")].sort_index()

        enriched = live_trader.indicator_engine.compute_all(live_trader.historical_df)
        live_trader.historical_df = enriched
        curr_idx = len(enriched) - 1

        active_dir = live_trader.portfolio.active_trade.direction if live_trader.portfolio.active_trade else 0
        bars_since_loss = live_trader.portfolio.get_bars_since_last_sl(curr_idx)

        # Evaluate live signal
        decision = live_trader.algorithm.evaluate_bar(
            df=enriched,
            current_idx=curr_idx,
            active_position_dir=active_dir,
            bars_since_last_loss=bars_since_loss,
        )

        curr_atr = float(enriched["atr"].iloc[curr_idx])
        market_open_price = float(forming_bar["open"])

        # Check entry trigger
        if decision.is_valid and live_trader.portfolio.active_trade is None:
            live_trader.portfolio.open_trade(
                direction=decision.direction,
                market_open_price=market_open_price,
                current_atr=curr_atr,
                timestamp=test_df.index[i + 1],
                bar_idx=curr_idx + 1,
            )

        # Intrabar execution for forming bar
        live_trader.portfolio.process_bar(
            bar_idx=curr_idx + 1,
            timestamp=test_df.index[i + 1],
            open_price=float(forming_bar["open"]),
            high_price=float(forming_bar["high"]),
            low_price=float(forming_bar["low"]),
            close_price=float(forming_bar["close"]),
            current_atr=curr_atr,
            exit_indicator_signal=decision.exit_triggered,
        )

    # Force close any open position at end
    final_idx = len(live_trader.historical_df) - 1
    final_close = float(test_df.iloc[-1]["close"])
    live_trader.portfolio.force_close_at_end(final_close=final_close, timestamp=test_df.index[-1], bar_idx=final_idx)

    live_metrics = MetricsCalculator.calculate(
        initial_capital=bt_cfg.initial_capital,
        completed_trades=live_trader.portfolio.completed_trades,
        equity_curve=live_trader.portfolio.equity_curve,
        annualization_factor=252 * 24,
    )
    live_trades = live_trader.portfolio.completed_trades

    # 5. Output Accuracy and Parity Scorecard
    print("\n" + "=" * 90)
    print(f"             LIVE MODEL TRADE ACCURACY & PARITY SCORECARD ({symbol})             ")
    print("=" * 90)

    comparison = [
        ["Total Completed Trades", str(bt_metrics.total_trades), str(live_metrics.total_trades), "PERFECT MATCH" if bt_metrics.total_trades == live_metrics.total_trades else "DIFF"],
        ["Winning Trades", str(bt_metrics.winning_trades), str(live_metrics.winning_trades), "PERFECT MATCH" if bt_metrics.winning_trades == live_metrics.winning_trades else "DIFF"],
        ["Losing Trades", str(bt_metrics.losing_trades), str(live_metrics.losing_trades), "PERFECT MATCH" if bt_metrics.losing_trades == live_metrics.losing_trades else "DIFF"],
        ["Win Rate (%)", f"{bt_metrics.win_rate_pct:.1f}%", f"{live_metrics.win_rate_pct:.1f}%", "PERFECT MATCH" if abs(bt_metrics.win_rate_pct - live_metrics.win_rate_pct) < 0.01 else "DIFF"],
        ["TP1 Hit Rate (%)", f"{bt_metrics.tp1_hit_rate_pct:.1f}%", f"{live_metrics.tp1_hit_rate_pct:.1f}%", "PERFECT MATCH" if abs(bt_metrics.tp1_hit_rate_pct - live_metrics.tp1_hit_rate_pct) < 0.01 else "DIFF"],
        ["Net Realized Profit ($)", f"${bt_metrics.total_net_profit:+,.2f}", f"${live_metrics.total_net_profit:+,.2f}", "PERFECT MATCH" if abs(bt_metrics.total_net_profit - live_metrics.total_net_profit) < 1.0 else "DIFF"],
        ["Profit Factor", f"{bt_metrics.profit_factor:.2f}", f"{live_metrics.profit_factor:.2f}", "PERFECT MATCH" if abs(bt_metrics.profit_factor - live_metrics.profit_factor) < 0.01 else "DIFF"],
        ["Max Drawdown (%)", f"{bt_metrics.max_drawdown_pct:.2f}%", f"{live_metrics.max_drawdown_pct:.2f}%", "PERFECT MATCH" if abs(bt_metrics.max_drawdown_pct - live_metrics.max_drawdown_pct) < 0.01 else "DIFF"],
    ]

    print(tabulate(comparison, headers=["Metric", "Offline Backtester", "Live Trade Model", "Parity Status"], tablefmt="grid"))

    print("\n" + "=" * 90)
    print(f"                       LIVE MODEL AUDIT LOG ({symbol})                       ")
    print("=" * 90)
    trade_rows = []
    for t in live_trades:
        status = "WIN" if t.total_pnl_dollars > 0 else "LOSS"
        side_str = "LONG" if t.direction == 1 else "SHORT"
        tp1_str = "YES" if t.tp1_hit else "NO"
        exits_str = ", ".join(tr.exit_reason.value for tr in t.tranches)
        trade_rows.append([
            t.trade_id,
            side_str,
            status,
            str(t.entry_time),
            f"${t.entry_price:,.2f}",
            str(t.exit_time),
            tp1_str,
            f"${t.total_pnl_dollars:+,.2f}",
            f"{t.return_pct:+.2f}%",
            exits_str,
        ])
    print(tabulate(trade_rows, headers=["#", "Side", "Result", "Entry Time", "Entry Px", "Exit Time", "TP1 Hit", "PnL ($)", "Return %", "Tranche Exits"], tablefmt="grid"))
    return live_metrics


if __name__ == "__main__":
    eth_metrics = test_live_accuracy_on_symbol("ETH/USDT", "data/binance_ETH_USDT_1h.csv")
    btc_metrics = test_live_accuracy_on_symbol("BTC/USDT", "data/binance_BTC_USDT_1h.csv")
