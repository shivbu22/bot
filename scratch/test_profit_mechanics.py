import sys
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig
from nnfx.backtester import Backtester
from nnfx.data_loader import DataLoader
from nnfx.indicator_engine import IndicatorEngine

def test_profit_mechanics():
    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "ADA_USDT", "XRP_USDT"]
    base_ind = IndicatorConfig(adx_enabled=True)
    engine = IndicatorEngine(base_ind)
    
    # Precompute enriched dataframes
    datasets = {}
    for tf in ["1h", "4h"]:
        for s in symbols:
            p = Path(f"data/binance_{s}_{tf}.csv")
            if p.exists():
                df = DataLoader.load(p)
                datasets[f"{s}_{tf}"] = (df, engine.compute_all(df))
                
    print(f"Precomputed {len(datasets)} datasets. Starting search...")
    
    candidates = []
    
    for sl_mult in [2.5, 2.8, 3.0]:
        for tp1_mult in [0.8, 1.0, 1.2]:
            for tranche_split in [(0.20, 0.80), (0.25, 0.75), (0.33, 0.67)]:
                for min_c2 in [0.15, 0.18, 0.20]:
                    for min_adx in [14.0, 16.0, 18.0]:
                        for slope_filter in [True, False]:
                            cfg = BacktestConfig(
                                initial_capital=100_000.0,
                                risk=RiskConfig(
                                    risk_pct=0.02,
                                    sl_atr_multiplier=sl_mult,
                                    tp1_atr_multiplier=tp1_mult,
                                    tranche_1_pct=tranche_split[0],
                                    tranche_2_pct=tranche_split[1],
                                    move_to_be_on_tp1=True,
                                    loss_cooldown_bars=2,
                                ),
                                indicators=IndicatorConfig(
                                    baseline_type="EMA",
                                    baseline_period=200,
                                    baseline_max_atr_dist=1.2,
                                    ssl_period=10,
                                    c2_enabled=True,
                                    vortex_period=14,
                                    min_c2_spread=min_c2,
                                    adx_enabled=True,
                                    adx_period=14,
                                    min_adx=min_adx,
                                    baseline_slope_filter=slope_filter,
                                    allow_bridge_trades=False,   # Strict C1 Flips only
                                    exit_oscillator_period=14,
                                    exit_ob_level=0.88,
                                    exit_os_level=0.12,
                                    runner_trailing_atr_mult=2.2,
                                ),
                                execution=ExecutionConfig(spread_pips=0.00015, slippage_pips=0.00005),
                            )
                            
                            wins, losses = 0, 0
                            profit = 0.0
                            trade_count = 0
                            high_profit_count = 0
                            
                            for k, (df, enriched) in datasets.items():
                                bt = Backtester(cfg)
                                m = bt.run(df, enriched_df=enriched)
                                wins += m.winning_trades
                                losses += m.losing_trades
                                trade_count += m.total_trades
                                profit += m.total_net_profit
                                for t in bt.portfolio.completed_trades:
                                    if t.return_pct >= 4.0:
                                        high_profit_count += 1
                                        
                            wr = (wins / trade_count * 100) if trade_count > 0 else 0
                            if wr >= 85.0 and trade_count >= 5 and profit > 0:
                                candidates.append({
                                    "sl": sl_mult, "tp1": tp1_mult, "tranche": tranche_split,
                                    "c2": min_c2, "adx": min_adx, "slope": slope_filter,
                                    "wins": wins, "losses": losses, "total": trade_count,
                                    "wr": wr, "profit": profit, "high_profit": high_profit_count
                                })

    candidates.sort(key=lambda x: (x["wr"], x["profit"]), reverse=True)
    print(f"\nFound {len(candidates)} high-accuracy configurations! Top 10:")
    for c in candidates[:10]:
        print(f"WR={c['wr']:.1f}% ({c['wins']}W/{c['losses']}L in {c['total']}) | Net Profit: {c['profit']:+,.2f} | >4% Trades: {c['high_profit']} | SL={c['sl']}x | TP1={c['tp1']}x | Tranche={c['tranche']} | C2>={c['c2']} | ADX>={c['adx']} | Slope={c['slope']}")

if __name__ == "__main__":
    test_profit_mechanics()
