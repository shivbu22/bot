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
    e_idx = df.index.get_loc(t.entry_time)
    sig_idx = e_idx - 1
    sig = ind_df.iloc[sig_idx]
    res = "WIN" if t.total_pnl_dollars > 0 else "LOSS"
    
    # Close relative to high-low range
    rng = sig["high"] - sig["low"]
    pos = (sig["close"] - sig["low"]) / (rng + 1e-10)
    print(f"Trade #{t.trade_id} [{res}] ({'LONG' if t.direction == 1 else 'SHORT'}): Close Position={pos:.2f} | Candle Range/ATR={sig['candle_atr_ratio']:.2f} | Vortex Spread={abs(sig['vortex_plus'] - sig['vortex_minus']):.3f}")
