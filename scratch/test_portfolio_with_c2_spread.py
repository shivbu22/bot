import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="BTC/USDT", timeframe="1h"))
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

for min_sp, min_imp in [(0.15, 1.0), (0.18, 1.0), (0.18, 1.1)]:
    print(f"\n==================== SPREAD >= {min_sp}, IMPULSE >= {min_imp} ====================")
    tot_trades = 0
    tot_wins = 0
    tot_losses = 0
    tot_pnl = 0.0

    for sym in symbols:
        df = feed.fetch_historical_ohlcv(symbol=sym, timeframe="1h", limit=1000, use_cache=True)
        cfg = BacktestConfig(
            risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
            indicators=IndicatorConfig(
                adx_enabled=True,
                min_adx=12.0,
                allow_bridge_trades=True,
                max_bridge_bars=5,
                min_bridge_candle_atr=min_imp
            )
        )
        b = Backtester(cfg)
        orig_eval = b.algorithm.evaluate_bar
        def custom_eval(df, current_idx, active_position_dir=0, bars_since_last_loss=9999, ms=min_sp):
            res = orig_eval(df, current_idx, active_position_dir, bars_since_last_loss)
            if res.is_valid and active_position_dir == 0:
                row = df.iloc[current_idx]
                spread = abs(float(row["vortex_plus"]) - float(row["vortex_minus"]))
                if spread < ms:
                    from nnfx.algorithm import SignalDecision, SignalType
                    return SignalDecision(
                        bar_index=current_idx, timestamp=df.index[current_idx], signal=SignalType.NONE,
                        is_valid=False, direction=0, rejection_reason=f"C2 Vortex spread too weak ({spread:.3f} < {ms})",
                        baseline_val=row["baseline"], baseline_dist_atr=0.0, ssl_direction=row["ssl_dir"],
                        volume_active=False, exit_triggered=False
                    )
            return res
        b.algorithm.evaluate_bar = custom_eval
        m = b.run(df)
        tot_trades += m.total_trades
        tot_wins += m.winning_trades
        tot_losses += m.losing_trades
        tot_pnl += m.total_net_profit
        print(f"{sym:<10}: Trades={m.total_trades:<2} Wins={m.winning_trades:<2} Losses={m.losing_trades:<2} WinRate={m.win_rate_pct:>5.1f}% PnL=${m.total_net_profit:+,.2f} MaxDD={m.max_drawdown_pct:.2f}% PF={m.profit_factor:.2f}")

    wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0
    print("-" * 75)
    print(f"PORTFOLIO: Trades={tot_trades} | Wins={tot_wins} | Losses={tot_losses} | WinRate={wr:.1f}% | Net PnL=${tot_pnl:+,.2f}")
