import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_full = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=2000, use_cache=True)

# Test with baseline slope filter
for slope_filter in [False, True]:
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
    m = b.run(df_full)
    print(f"ETH Strict C1 (Baseline Slope Filter = {slope_filter}): Trades={m.total_trades} | Wins={m.winning_trades} | Losses={m.losing_trades} | Win Rate={m.win_rate_pct:.1f}% | Net PnL=${m.total_net_profit:+,.2f} | PF={m.profit_factor:.2f}")
