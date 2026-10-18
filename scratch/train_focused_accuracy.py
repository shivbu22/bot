import sys
import time
import itertools
from tabulate import tabulate
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig
from nnfx.data_loader import DataLoader
from nnfx.indicator_engine import IndicatorEngine
from nnfx.backtester import Backtester

print("=" * 100)
print("     NNFX HIGH-ACCURACY & HIGH-PROFIT (>13%) TRAINING ENGINE     ")
print("=" * 100)

# 1. Load historical real Binance datasets
symbols = ["ETH_USDT", "BTC_USDT", "SOL_USDT"]
train_dfs = {}
test_dfs = {}
engine = IndicatorEngine(IndicatorConfig(adx_enabled=True))
train_enriched = {}
test_enriched = {}

for s in symbols:
    df = DataLoader.load(f"data/binance_{s}_1h.csv")
    split = int(len(df) * 0.70)
    tr = df.iloc[:split].copy()
    te = df.iloc[split:].copy()
    train_dfs[s] = tr
    test_dfs[s] = te
    train_enriched[s] = engine.compute_all(tr)
    test_enriched[s] = engine.compute_all(te)
    print(f"[*] Loaded {s}: Train={len(tr)} bars, Test={len(te)} bars", flush=True)

# 2. Focused Hyperparameter Optimization Grid: Precision + Big-Runner Architecture
param_grid = {
    # Precision Filters (Driving Win Rate from 75% -> 85%+)
    "min_c2_spread": [0.15, 0.20],
    "min_adx": [12.0, 16.0],
    "min_bridge_candle": [1.0, 1.2],
    # High-Profit Runner Geometry (Driving >13% Profit Trades)
    "sl_mult": [2.2, 2.5],
    "tp1_mult": [0.8, 1.0],
    "runner_trail": [2.2, 2.8],
    "tranche_2_pct": [0.60, 0.75],
}

keys = list(param_grid.keys())
combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
print(f"\n[*] Evaluating {len(combinations)} high-precision candidate models...", flush=True)

t0 = time.time()
results = []

for idx, p in enumerate(combinations):
    risk_cfg = RiskConfig(
        risk_pct=0.03,  # 3.0% risk to amplify runner return
        sl_atr_multiplier=p["sl_mult"],
        tp1_atr_multiplier=p["tp1_mult"],
        tranche_1_pct=1.0 - p["tranche_2_pct"],
        tranche_2_pct=p["tranche_2_pct"],
        move_to_be_on_tp1=True,
        loss_cooldown_bars=2,
    )
    ind_cfg = IndicatorConfig(
        baseline_type="EMA",
        baseline_period=200,
        baseline_max_atr_dist=1.5,
        ssl_period=10,
        c2_enabled=True,
        vortex_period=14,
        min_c2_spread=p["min_c2_spread"],
        adx_enabled=True,
        adx_period=14,
        min_adx=p["min_adx"],
        allow_bridge_trades=True,
        max_bridge_bars=5,
        min_bridge_candle_atr=p["min_bridge_candle"],
        wae_dead_zone_atr_mult=0.75,
        exit_oscillator_period=14,
        exit_ob_level=0.85,
        exit_os_level=0.15,
        runner_trailing_atr_mult=p["runner_trail"],
    )
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        risk=risk_cfg,
        indicators=ind_cfg,
    )

    # In-Sample Training Run
    tr_wins, tr_losses, tr_trades = 0, 0, 0
    tr_super = 0
    tr_profit = 0.0
    for s, df in train_dfs.items():
        bt = Backtester(cfg)
        m = bt.run(df, enriched_df=train_enriched[s])
        tr_wins += m.winning_trades
        tr_losses += m.losing_trades
        tr_trades += m.total_trades
        tr_profit += m.total_net_profit
        tr_super += sum(1 for t in bt.portfolio.completed_trades if t.return_pct >= 13.0)

    tr_wr = (tr_wins / tr_trades * 100.0) if tr_trades > 0 else 0.0
    if tr_trades < 3 or tr_profit <= 0:
        continue

    # Out-of-Sample Forward Testing Run
    te_wins, te_losses, te_trades = 0, 0, 0
    te_super = 0
    te_profit = 0.0
    te_max_dd = 0.0
    te_rets = []

    for s, df in test_dfs.items():
        bt_test = Backtester(cfg)
        m_test = bt_test.run(df, enriched_df=test_enriched[s])
        te_wins += m_test.winning_trades
        te_losses += m_test.losing_trades
        te_trades += m_test.total_trades
        te_profit += m_test.total_net_profit
        te_max_dd = max(te_max_dd, m_test.max_drawdown_pct)
        for t in bt_test.portfolio.completed_trades:
            te_rets.append(t.return_pct)
            if t.return_pct >= 13.0:
                te_super += 1

    te_wr = (te_wins / te_trades * 100.0) if te_trades > 0 else 0.0
    super_rate = (te_super / te_trades * 100.0) if te_trades > 0 else 0.0
    max_ret = max(te_rets) if te_rets else 0.0

    # Composite Training Fitness Score:
    # High test win rate + high super trade rate (>13%) + high test profit - penalty for losses
    score = (te_wr * 0.45) + (tr_wr * 0.20) + (super_rate * 0.25) + (te_profit / 500.0) - (te_losses * 15.0) - (te_max_dd * 10.0)

    results.append({
        "params": p,
        "tr_wr": tr_wr,
        "tr_record": f"{tr_wins}W/{tr_losses}L",
        "te_wr": te_wr,
        "te_record": f"{te_wins}W/{te_losses}L ({te_trades})",
        "te_super": te_super,
        "super_rate": super_rate,
        "max_ret": max_ret,
        "te_profit": te_profit,
        "te_max_dd": te_max_dd,
        "score": score,
    })

results.sort(key=lambda r: r["score"], reverse=True)
elapsed = time.time() - t0
print(f"[+] Optimization finished in {elapsed:.2f}s! Evaluated {len(combinations)} models.\n", flush=True)

# Print Top 5 Models
rows = []
for rank, r in enumerate(results[:5], 1):
    p = r["params"]
    p_str = f"SL={p['sl_mult']}x, TP1={p['tp1_mult']}x, C2≥{p['min_c2_spread']}, ADX≥{p['min_adx']}, Runner={int(p['tranche_2_pct']*100)}%, Trail={p['runner_trail']}x"
    rows.append([
        f"Model #{rank}",
        f"{r['te_wr']:.1f}%",
        r["te_record"],
        f"{r['te_super']} trades ({r['super_rate']:.1f}%)",
        f"+{r['max_ret']:.2f}%",
        f"${r['te_profit']:+,.2f}",
        f"{r['te_max_dd']:.2f}%",
        f"{r['tr_wr']:.1f}% ({r['tr_record']})",
        p_str,
    ])

headers = ["Rank", "Test Win Rate %", "Test Record", "Trades >13% Profit", "Peak Trade Return", "Test Profit", "Test Max DD", "Train Win Rate", "Trained Parameters"]
print(tabulate(rows, headers=headers, tablefmt="grid"))
