import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
import pandas as pd
from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df_full = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=2000, use_cache=True)

# Train on first 1,200 bars (In-Sample / Training), Test on last 800 bars (Out-of-Sample / Test)
train_df = df_full.iloc[:1200]
test_df = df_full.iloc[1200:]

print(f"Total ETH/USDT bars: {len(df_full)}")
print(f"Train period: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
print(f"Test period:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

# Define focused parameter grid for ETH training
param_grid = {
    "sl_mult": [1.8, 2.0, 2.2, 2.5],
    "tp1_mult": [0.6, 0.8, 1.0],
    "min_adx": [12.0, 15.0, 18.0],
    "ssl_period": [10, 12],
    "vortex_period": [14],
    "wae_dead_zone": [0.75, 1.0],
    "allow_bridge": [True, False],
    "min_bridge_impulse": [0.8, 1.0],
    "loss_cooldown": [1, 2],
}

# Run combinatorial grid search on Training Data
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Total parameter combinations to evaluate: {len(combinations)}")

results = []

for i, p in enumerate(combinations):
    risk_cfg = RiskConfig(
        sl_atr_multiplier=p["sl_mult"],
        tp1_atr_multiplier=p["tp1_mult"],
        loss_cooldown_bars=p["loss_cooldown"]
    )
    ind_cfg = IndicatorConfig(
        ssl_period=p["ssl_period"],
        vortex_period=p["vortex_period"],
        adx_enabled=True,
        min_adx=p["min_adx"],
        wae_dead_zone_atr_mult=p["wae_dead_zone"],
        allow_bridge_trades=p["allow_bridge"],
        max_bridge_bars=5,
        min_bridge_candle_atr=p["min_bridge_impulse"]
    )
    b = Backtester(BacktestConfig(risk=risk_cfg, indicators=ind_cfg))
    try:
        m = b.run(train_df)
        # Filter for candidates with high win rate and positive profit
        if m.total_trades >= 4 and m.win_rate_pct >= 80.0 and m.total_net_profit > 0:
            results.append({
                "params": p,
                "train_trades": m.total_trades,
                "train_wr": m.win_rate_pct,
                "train_pnl": m.total_net_profit,
                "train_pf": m.profit_factor,
                "train_dd": m.max_drawdown_pct
            })
    except Exception:
        continue

# Sort by win rate, then PnL
results.sort(key=lambda x: (x["train_wr"], x["train_pf"], x["train_pnl"]), reverse=True)
print(f"\n[+] Found {len(results)} candidate configurations with >= 80% Win Rate on Training Data!")

print("\n--- TOP 10 CANDIDATES EVALUATED ON OUT-OF-SAMPLE TEST DATA ---")
print(f"{'Rank':<5} {'Win Rate':<10} {'Trn PnL':<12} {'Trn Trades':<12} {'Test WR':<10} {'Test PnL':<12} {'Test Trades':<12} {'Test PF':<8} {'Config'}")
print("-" * 110)

validated_results = []

for rank, r in enumerate(results[:25]):
    p = r["params"]
    risk_cfg = RiskConfig(
        sl_atr_multiplier=p["sl_mult"],
        tp1_atr_multiplier=p["tp1_mult"],
        loss_cooldown_bars=p["loss_cooldown"]
    )
    ind_cfg = IndicatorConfig(
        ssl_period=p["ssl_period"],
        vortex_period=p["vortex_period"],
        adx_enabled=True,
        min_adx=p["min_adx"],
        wae_dead_zone_atr_mult=p["wae_dead_zone"],
        allow_bridge_trades=p["allow_bridge"],
        max_bridge_bars=5,
        min_bridge_candle_atr=p["min_bridge_impulse"]
    )
    b_test = Backtester(BacktestConfig(risk=risk_cfg, indicators=ind_cfg))
    m_test = b_test.run(test_df)
    
    # Also evaluate full 2000 bars
    b_full = Backtester(BacktestConfig(risk=risk_cfg, indicators=ind_cfg))
    m_full = b_full.run(df_full)

    validated_results.append({
        **r,
        "test_trades": m_test.total_trades,
        "test_wr": m_test.win_rate_pct,
        "test_pnl": m_test.total_net_profit,
        "test_pf": m_test.profit_factor,
        "full_trades": m_full.total_trades,
        "full_wr": m_full.win_rate_pct,
        "full_pnl": m_full.total_net_profit,
        "full_pf": m_full.profit_factor,
        "full_dd": m_full.max_drawdown_pct
    })
    
    if rank < 10:
        cfg_str = f"SL={p['sl_mult']}, TP1={p['tp1_mult']}, ADX={p['min_adx']}, SSL={p['ssl_period']}, WAE_DZ={p['wae_dead_zone']}, Bridge={p['allow_bridge']}"
        print(f"#{rank+1:<4} {r['train_wr']:>5.1f}%     ${r['train_pnl']:>+9.2f}  {r['train_trades']:<12} {m_test.win_rate_pct:>5.1f}%     ${m_test.total_net_profit:>+9.2f}  {m_test.total_trades:<12} {m_test.profit_factor:>6.2f}   {cfg_str}")

# Sort validated results by overall full 2000-bar performance
validated_results.sort(key=lambda x: (x["full_wr"], x["full_pnl"]), reverse=True)
print("\n" + "=" * 110)
print("                     TOP 5 ROBUST CONFIGURATIONS ACROSS ENTIRE 2,000 BARS OF ETH/USDT                    ")
print("=" * 110)
print(f"{'Rank':<5} {'2k WR':<10} {'2k PnL':<14} {'2k Trades':<10} {'2k PF':<8} {'2k MaxDD':<10} {'Parameters'}")
print("-" * 110)
for rank, vr in enumerate(validated_results[:5]):
    p = vr["params"]
    cfg_str = f"SL={p['sl_mult']}, TP1={p['tp1_mult']}, ADX>={p['min_adx']}, SSL={p['ssl_period']}, WAE_DZ={p['wae_dead_zone']}, Bridge={p['allow_bridge']}, Impulse={p['min_bridge_impulse']}"
    print(f"#{rank+1:<4} {vr['full_wr']:>5.1f}%     ${vr['full_pnl']:>+10.2f}    {vr['full_trades']:<10} {vr['full_pf']:>6.2f}   {vr['full_dd']:>5.2f}%     {cfg_str}")
