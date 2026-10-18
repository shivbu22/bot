"""Diagnostic script to autopsy losing trades across BTC and ETH full 2000 bars."""
import pandas as pd
from nnfx.data_loader import DataLoader
from nnfx.backtester import Backtester
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig

def analyze_symbol(symbol_csv):
    print(f"\n========================================================")
    print(f"ANALYZING: {symbol_csv}")
    print(f"========================================================")
    df = DataLoader.load(symbol_csv)
    
    cfg = BacktestConfig(
        risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
        indicators=IndicatorConfig(
            min_c2_spread=0.15,
            min_bridge_candle_atr=1.0,
            adx_enabled=True,
            min_adx=12.0,
            allow_bridge_trades=True,
            max_bridge_bars=5,
        )
    )
    
    bt = Backtester(cfg)
    metrics = bt.run(df)
    trades = bt.portfolio.completed_trades
    df_ind = bt.indicator_engine.compute_all(df)
    
    print(f"Total Trades: {len(trades)} | Win Rate: {metrics.win_rate_pct:.1f}% | Net Profit: ${metrics.total_net_profit:,.2f}")
    
    for t in trades:
        entry_ts = t.entry_time
        # Find row in df_ind
        matching_rows = df_ind[df_ind.index == entry_ts]
        if matching_rows.empty:
            continue
        entry_idx = df_ind.index.get_loc(entry_ts)
        sig_idx = entry_idx - 1
        sig_row = df_ind.iloc[sig_idx]
        
        c2_spread = abs(sig_row["vortex_plus"] - sig_row["vortex_minus"])
        candle_range = sig_row["high"] - sig_row["low"]
        candle_atr_ratio = candle_range / sig_row["atr"] if sig_row["atr"] > 0 else 0
        adx_val = sig_row.get("adx", 0.0)
        base_prev = df_ind.iloc[max(0, sig_idx - 5)]["baseline"]
        base_slope = (sig_row["baseline"] - base_prev) / 5
        side_str = "LONG" if t.direction == 1 else "SHORT"
        status = "WIN" if t.total_pnl_dollars > 0 else "LOSS"
        
        wae_power = sig_row["wae_trend_power"]
        wae_exp = sig_row["wae_explosion"]
        wae_dz = sig_row["wae_dead_zone"]
        
        print(f"\nTrade #{t.trade_id} [{status}] {side_str} | PnL: ${t.total_pnl_dollars:.2f} ({t.return_pct:.2f}%)")
        print(f"  Signal Time: {sig_row.name} | Entry: {t.entry_price:.2f} at {t.entry_time}")
        print(f"  Exit Time: {t.exit_time} | Exits: {[tr.exit_reason.value for tr in t.tranches]}")
        print(f"  ATR: {sig_row['atr']:.2f} | Candle/ATR: {candle_atr_ratio:.3f}")
        print(f"  ADX: {adx_val:.2f}")
        print(f"  C2 Vortex: VI+={sig_row['vortex_plus']:.3f}, VI-={sig_row['vortex_minus']:.3f}, Spread={c2_spread:.3f}")
        print(f"  WAE: power={wae_power:.2f}, explosion={wae_exp:.2f}, dead_zone={wae_dz:.2f}")
        print(f"  Baseline: {sig_row['baseline']:.2f} | Dist: {abs(sig_row['close'] - sig_row['baseline'])/sig_row['atr']:.2f} ATR | Slope(5): {base_slope:.4f}")

if __name__ == "__main__":
    analyze_symbol("data/binance_ETH_USDT_1h.csv")
    analyze_symbol("data/binance_BTC_USDT_1h.csv")
    analyze_symbol("data/binance_SOL_USDT_1h.csv")
