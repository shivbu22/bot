import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_full = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=2000, use_cache=True)

cfg = BacktestConfig(
    risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
    indicators=IndicatorConfig(adx_enabled=True, min_adx=12.0, allow_bridge_trades=False)
)
b = Backtester(cfg)
ind_df = b.indicator_engine.compute_all(df_full)
m = b.run(df_full)

print(f"ETH 2000 bars Strict C1: Trades={m.total_trades}, Wins={m.winning_trades}, Losses={m.losing_trades}, WR={m.win_rate_pct:.1f}%")

for t in b.portfolio.completed_trades:
    res = "WIN" if t.total_pnl_dollars > 0 else "LOSS"
    e_idx = df_full.index.get_loc(t.entry_time)
    sig_idx = e_idx - 1
    sig = ind_df.iloc[sig_idx]
    side = "LONG" if t.direction == 1 else "SHORT"
    print(f"\nTrade #{t.trade_id} [{res}] {side} @ {t.entry_price:.2f} ({t.entry_time}) -> Exit @ {t.exit_time} | PnL=${t.total_pnl_dollars:+,.2f} | TP1 Hit={t.tp1_hit}")
    print(f"  Signal Bar at {df_full.index[sig_idx]}: Close={sig['close']}, Baseline={sig['baseline']:.2f}, Slope={sig['baseline_slope']:.6f}, Dist={sig['baseline_atr_dist']:.2f}, ADX={sig['adx']:.1f}, Candle/ATR={sig['candle_atr_ratio']:.2f}")
    if res == "LOSS":
        x_idx = df_full.index.get_loc(t.exit_time)
        print(f"  Bars in trade:")
        print(df_full.iloc[e_idx:x_idx+1][["open", "high", "low", "close"]])
