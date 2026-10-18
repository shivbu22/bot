import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)

# Current settings from main.py
cfg = BacktestConfig(
    risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
    indicators=IndicatorConfig(
        adx_enabled=True,
        min_adx=12.0,
        allow_bridge_trades=True,
        max_bridge_bars=5,
        min_bridge_candle_atr=0.8
    )
)

b = Backtester(cfg)
ind_df = b.indicator_engine.compute_all(df)
m = b.run(df)

print(f"ETH/USDT (1,000 bars) current results:")
print(f"Trades: {m.total_trades} | Wins: {m.winning_trades} | Losses: {m.losing_trades} | Win Rate: {m.win_rate_pct:.1f}% | PnL: ${m.total_net_profit:+,.2f}")

print("\n--- DETAILED TRADE BREAKDOWN ---")
for t in b.portfolio.completed_trades:
    e_idx = df.index.get_loc(t.entry_time)
    sig_idx = e_idx - 1
    sig = ind_df.iloc[sig_idx]
    res = "WIN" if t.total_pnl_dollars > 0 else "LOSS"
    side = "LONG" if t.direction == 1 else "SHORT"
    dist = abs(sig['close'] - sig['baseline']) / sig['atr']
    print(f"Trade #{t.trade_id} [{res}] {side} @ {t.entry_price:.2f} ({t.entry_time}) -> Exit @ {t.exit_time} | PnL=${t.total_pnl_dollars:+,.2f} | TP1 Hit={t.tp1_hit}")
    print(f"    Signal at {df.index[sig_idx]}: SSL_Flip={sig['ssl_flip']}, SSL_Dir={sig['ssl_dir']}, C2={sig['c2_dir']}, ADX={sig['adx']:.1f}, WAE_Bull={sig['wae_bull_vol']}, WAE_Bear={sig['wae_bear_vol']}, Candle/ATR={sig['candle_atr_ratio']:.2f}, Dist/ATR={dist:.2f}")
    if res == "LOSS":
        x_idx = df.index.get_loc(t.exit_time)
        trade_bars = df.iloc[e_idx:x_idx+1]
        print(f"    Loss details: initial SL={t.initial_sl:.2f}, TP1={t.tp1_price:.2f}")
