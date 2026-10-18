import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="BTC/USDT", timeframe="1h"))

symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
datasets = {sym: feed.fetch_historical_ohlcv(symbol=sym, timeframe="1h", limit=2000, use_cache=True) for sym in symbols}

for slope_filter in [False, True]:
    print(f"\n==================== BASELINE SLOPE FILTER = {slope_filter} (Strict C1) ====================")
    tot_trades = 0
    tot_wins = 0
    tot_losses = 0
    tot_pnl = 0.0
    
    for sym in symbols:
        df = datasets[sym]
        cfg = BacktestConfig(
            risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
            indicators=IndicatorConfig(
                adx_enabled=True,
                min_adx=12.0,
                allow_bridge_trades=False,
                baseline_slope_filter=slope_filter
            )
        )
        b = Backtester(cfg)
        m = b.run(df)
        tot_trades += m.total_trades
        tot_wins += m.winning_trades
        tot_losses += m.losing_trades
        tot_pnl += m.total_net_profit
        print(f"{sym:<10}: Trades={m.total_trades:<2} Wins={m.winning_trades:<2} Losses={m.losing_trades:<2} WinRate={m.win_rate_pct:>5.1f}% PnL=${m.total_net_profit:+,.2f} MaxDD={m.max_drawdown_pct:.2f}% PF={m.profit_factor:.2f}")

    wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0
    print("-" * 75)
    print(f"PORTFOLIO TOTAL: Trades={tot_trades} | Wins={tot_wins} | Losses={tot_losses} | WinRate={wr:.1f}% | Net PnL=${tot_pnl:+,.2f}")
