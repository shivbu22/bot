import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_1000 = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)

for slope_filter in [False, True]:
    cfg = BacktestConfig(
        risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
        indicators=IndicatorConfig(
            adx_enabled=True,
            min_adx=12.0,
            allow_bridge_trades=True,
            max_bridge_bars=5,
            min_bridge_candle_atr=0.8,
            baseline_slope_filter=slope_filter
        )
    )
    b = Backtester(cfg)
    m = b.run(df_1000)
    print(f"ETH 1,000 bars (Baseline Slope Filter = {slope_filter}): Trades={m.total_trades} | Wins={m.winning_trades} | Losses={m.losing_trades} | WinRate={m.win_rate_pct:.1f}% | Net PnL=${m.total_net_profit:+,.2f} | PF={m.profit_factor:.2f}")
