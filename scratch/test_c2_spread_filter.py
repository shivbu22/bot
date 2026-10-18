import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_1000 = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)
df_2000 = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=2000, use_cache=True)

print("==================== ETH/USDT PRECISION CALIBRATION ====================")

def test_spread_filter(df, df_name, min_spread, min_impulse, allow_bridge=True):
    cfg = BacktestConfig(
        risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
        indicators=IndicatorConfig(
            adx_enabled=True,
            min_adx=12.0,
            allow_bridge_trades=allow_bridge,
            max_bridge_bars=5,
            min_bridge_candle_atr=min_impulse
        )
    )
    b = Backtester(cfg)
    orig_eval = b.algorithm.evaluate_bar
    
    def custom_eval(df, current_idx, active_position_dir=0, bars_since_last_loss=9999):
        res = orig_eval(df, current_idx, active_position_dir, bars_since_last_loss)
        if res.is_valid and active_position_dir == 0:
            row = df.iloc[current_idx]
            spread = abs(float(row["vortex_plus"]) - float(row["vortex_minus"]))
            if spread < min_spread:
                from nnfx.algorithm import SignalDecision, SignalType
                return SignalDecision(
                    bar_index=current_idx, timestamp=df.index[current_idx], signal=SignalType.NONE,
                    is_valid=False, direction=0, rejection_reason=f"C2 Vortex spread too weak ({spread:.3f} < {min_spread:.3f})",
                    baseline_val=row["baseline"], baseline_dist_atr=0.0, ssl_direction=row["ssl_dir"],
                    volume_active=False, exit_triggered=False
                )
        return res
        
    b.algorithm.evaluate_bar = custom_eval
    m = b.run(df)
    lbl = f"Spread>={min_spread:.2f}, Imp>={min_impulse:.2f}, Bridge={allow_bridge}"
    print(f"{df_name:<11} | {lbl:<40} | Trades={m.total_trades:<2} Wins={m.winning_trades:<2} Losses={m.losing_trades:<2} WinRate={m.win_rate_pct:>5.1f}% PnL=${m.total_net_profit:+,.2f} MaxDD={m.max_drawdown_pct:.2f}% PF={m.profit_factor:.2f}")

# 1,000 bars tests
test_spread_filter(df_1000, "1,000 bars", min_spread=0.00, min_impulse=0.8, allow_bridge=True)
test_spread_filter(df_1000, "1,000 bars", min_spread=0.15, min_impulse=1.0, allow_bridge=True)
test_spread_filter(df_1000, "1,000 bars", min_spread=0.18, min_impulse=1.0, allow_bridge=True)
test_spread_filter(df_1000, "1,000 bars", min_spread=0.15, min_impulse=1.0, allow_bridge=False)

print("-" * 100)
# 2,000 bars tests
test_spread_filter(df_2000, "2,000 bars", min_spread=0.00, min_impulse=0.8, allow_bridge=True)
test_spread_filter(df_2000, "2,000 bars", min_spread=0.15, min_impulse=1.0, allow_bridge=True)
test_spread_filter(df_2000, "2,000 bars", min_spread=0.18, min_impulse=1.0, allow_bridge=True)
test_spread_filter(df_2000, "2,000 bars", min_spread=0.15, min_impulse=1.0, allow_bridge=False)
