import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)

b = Backtester(BacktestConfig(
    risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
    indicators=IndicatorConfig(adx_enabled=True, min_adx=12.0)
))

ind_df = b.indicator_engine.compute_all(df)
m = b.run(df)

for t in b.portfolio.completed_trades:
    if t.total_pnl_dollars <= 0:
        e_idx = df.index.get_loc(t.entry_time)
        sig_idx = e_idx - 1
        sig = ind_df.iloc[sig_idx]
        print(f"\nLoss Trade #{t.trade_id} ({'LONG' if t.direction == 1 else 'SHORT'}):")
        print(f"  Entry at {t.entry_time}, Exit at {t.exit_time}")
        print(f"  Close={sig['close']}, Baseline={sig['baseline']:.2f}")
        print(f"  Baseline Slope={sig['baseline_slope']:.6f}")
        print(f"  Baseline ATR Dist={sig['baseline_atr_dist']:.2f}")
        print(f"  Candle ATR Ratio={sig['candle_atr_ratio']:.2f}")
        print(f"  SSL Flip={sig['ssl_flip']}, SSL Dir={sig['ssl_dir']}")
        print(f"  ADX={sig['adx']:.2f}")
        print(f"  WAE Explosion={sig['wae_explosion']:.4f}, WAE Up={sig['wae_up']:.4f}, WAE Down={sig['wae_down']:.4f}, DeadZone={sig['wae_dead_zone']:.4f}")
        print(f"  DeMarker Oscillator={sig['exit_oscillator']:.4f}")
