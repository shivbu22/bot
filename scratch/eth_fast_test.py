import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_full = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=2000, use_cache=True)

print(f"Loaded {len(df_full)} bars of ETH/USDT from {df_full.index[0]} to {df_full.index[-1]}")

def run_eth_test(name, sl, tp1, min_adx, strict_c1, max_b, min_imp, loss_cool, min_bars_between_flips=0):
    cfg = BacktestConfig(
        risk=RiskConfig(sl_atr_multiplier=sl, tp1_atr_multiplier=tp1, loss_cooldown_bars=loss_cool),
        indicators=IndicatorConfig(
            adx_enabled=True,
            min_adx=min_adx,
            allow_bridge_trades=not strict_c1,
            max_bridge_bars=max_b,
            min_bridge_candle_atr=min_imp,
        )
    )
    b = Backtester(cfg)
    
    if min_bars_between_flips > 0:
        orig_eval = b.algorithm.evaluate_bar
        def custom_eval(df, current_idx, active_position_dir=0, bars_since_last_loss=9999):
            res = orig_eval(df, current_idx, active_position_dir, bars_since_last_loss)
            if res.is_valid and active_position_dir == 0:
                row = df.iloc[current_idx]
                target_dir = res.direction
                # Check when the PREVIOUS opposite flip occurred
                last_opp_flip = 9999
                for back in range(current_idx - 1, max(-1, current_idx - 50), -1):
                    if int(df.iloc[back]["ssl_flip"]) == -target_dir:
                        last_opp_flip = current_idx - back
                        break
                if last_opp_flip < min_bars_between_flips:
                    from nnfx.algorithm import SignalDecision, SignalType
                    return SignalDecision(
                        bar_index=current_idx, timestamp=df.index[current_idx], signal=SignalType.NONE,
                        is_valid=False, direction=0, rejection_reason=f"Chop filter: opposite C1 flip only {last_opp_flip} bars ago (< {min_bars_between_flips})",
                        baseline_val=row["baseline"], baseline_dist_atr=0.0, ssl_direction=row["ssl_dir"],
                        volume_active=False, exit_triggered=False
                    )
            return res
        b.algorithm.evaluate_bar = custom_eval

    m = b.run(df_full)
    print(f"{name:<45}: Trades={m.total_trades:<2} Wins={m.winning_trades:<2} Losses={m.losing_trades:<2} WinRate={m.win_rate_pct:>5.1f}% PnL=${m.total_net_profit:+,.2f} MaxDD={m.max_drawdown_pct:.2f}% PF={m.profit_factor:.2f}")
    return m

print("\n--- ETH/USDT 2,000 BARS STRATEGY COMPARISON ---")
# 1. Baseline High-Win-Rate
run_eth_test("1. Default High-Win-Rate (Calibrated Bridges)", 2.2, 0.8, 12.0, False, 5, 0.8, 2)

# 2. Strict C1 (No Bridge Trades)
run_eth_test("2. Strict C1 Only (Zero Bridge Trades)", 2.2, 0.8, 12.0, True, 0, 0.0, 2)

# 3. Strict C1 with ADX >= 15
run_eth_test("3. Strict C1 (ADX >= 15)", 2.2, 0.8, 15.0, True, 0, 0.0, 2)

# 4. Strict C1 with ADX >= 18
run_eth_test("4. Strict C1 (ADX >= 18)", 2.2, 0.8, 18.0, True, 0, 0.0, 2)

# 5. Strict C1 with SL=2.5, TP1=0.7, ADX>=14
run_eth_test("5. Strict C1 (SL=2.5, TP1=0.7, ADX>=14)", 2.5, 0.7, 14.0, True, 0, 0.0, 2)

# 6. Bridge Mode with Higher Impulse Threshold (>= 1.25 ATR)
run_eth_test("6. High-Impulse Bridges (Candle >= 1.25 ATR)", 2.2, 0.8, 12.0, False, 5, 1.25, 2)

# 7. Bridge Mode with C1 Stability (min 8 bars since opposite flip)
run_eth_test("7. Bridges + C1 Stability (>= 8 bars chop filter)", 2.2, 0.8, 12.0, False, 5, 0.8, 2, min_bars_between_flips=8)

# 8. Bridge Mode with C1 Stability (min 10 bars) + Impulse >= 1.0 ATR
run_eth_test("8. Bridges + Stability (>=10 bars) + Impulse 1.0", 2.2, 0.8, 12.0, False, 5, 1.0, 2, min_bars_between_flips=10)
