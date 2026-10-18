"""Comprehensive Multi-Asset NNFX Accuracy & Precision Training Grid Search."""
import itertools
import time
import pandas as pd
from tabulate import tabulate
from nnfx.data_loader import DataLoader
from nnfx.indicator_engine import IndicatorEngine
from nnfx.backtester import Backtester
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig

def evaluate_params(symbols, param_grid):
    # Pre-load datasets and precompute indicators
    data_map = {}
    enriched_map = {}
    base_ind = IndicatorConfig(adx_enabled=True)
    engine = IndicatorEngine(base_ind)
    
    for s in symbols:
        df = DataLoader.load(f"data/binance_{s}_1h.csv")
        data_map[s] = df
        enriched_map[s] = engine.compute_all(df)
    
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    print(f"[*] Total configurations to evaluate: {len(combinations)} across {len(symbols)} symbols")
    
    t0 = time.time()
    results = []
    
    for idx, p in enumerate(combinations):
        total_wins = 0
        total_losses = 0
        total_trades = 0
        total_profit = 0.0
        max_dd = 0.0
        symbol_details = {}
        
        cfg = BacktestConfig(
            risk=RiskConfig(
                sl_atr_multiplier=p["sl_mult"],
                tp1_atr_multiplier=p["tp1_mult"],
                loss_cooldown_bars=p["loss_cooldown"],
            ),
            indicators=IndicatorConfig(
                min_c2_spread=p["min_c2_spread"],
                min_bridge_candle_atr=p["min_bridge_candle_atr"],
                adx_enabled=True,
                min_adx=p["min_adx"],
                allow_bridge_trades=p.get("allow_bridge_trades", True),
                max_bridge_bars=p.get("max_bridge_bars", 5),
                wae_dead_zone_atr_mult=p.get("wae_dead_zone_atr_mult", 0.75),
                baseline_max_atr_dist=p.get("baseline_max_atr_dist", 1.5),
                ssl_period=p.get("ssl_period", 10),
            )
        )
        
        for s, df in data_map.items():
            bt = Backtester(cfg)
            m = bt.run(df, enriched_df=enriched_map[s])
            total_wins += m.winning_trades
            total_losses += m.losing_trades
            total_trades += m.total_trades
            total_profit += m.total_net_profit
            max_dd = max(max_dd, m.max_drawdown_pct)
            symbol_details[s] = (m.winning_trades, m.losing_trades, m.total_net_profit)
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        
        # Scoring function: Prioritize Win Rate (accuracy), profit, and penalize losses & drawdown
        # Require at least 6 total trades across symbols to prevent over-filtering to trivial sample
        if total_trades >= 5 and total_losses <= 2 and total_profit > 0:
            score = (win_rate * 3.0) + (total_profit / 50.0) - (max_dd * 15.0) - (total_losses * 100.0)
            results.append({
                "params": p,
                "win_rate": win_rate,
                "wins": total_wins,
                "losses": total_losses,
                "trades": total_trades,
                "net_profit": total_profit,
                "max_dd": max_dd,
                "score": score,
                "details": symbol_details,
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    elapsed = time.time() - t0
    print(f"[+] Evaluated {len(combinations)} configurations in {elapsed:.2f}s! Top {min(10, len(results))} setups:")
    
    rows = []
    for r in results[:10]:
        p = r["params"]
        p_str = f"SL={p['sl_mult']}, TP={p['tp1_mult']}, C2={p['min_c2_spread']}, Bridge={p['min_bridge_candle_atr']}, ADX={p['min_adx']}, CD={p['loss_cooldown']}"
        det_str = " | ".join(f"{s}:{w}W/{l}L" for s, (w, l, pnl) in r["details"].items())
        rows.append([
            f"{r['win_rate']:.1f}%",
            f"{r['wins']}W / {r['losses']}L ({r['trades']})",
            f"${r['net_profit']:+,.2f}",
            f"{r['max_dd']:.2f}%",
            det_str,
            p_str,
        ])
    
    headers = ["Win Rate", "Record", "Net Profit", "Max DD", "Symbol Breakdown", "Parameters"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    return results

if __name__ == "__main__":
    symbols = ["ETH_USDT", "BTC_USDT", "BNB_USDT"]
    param_grid = {
        "sl_mult": [2.0, 2.2, 2.5],
        "tp1_mult": [0.7, 0.8, 1.0],
        "min_c2_spread": [0.15, 0.18, 0.20],
        "min_bridge_candle_atr": [1.0, 1.1, 1.2],
        "min_adx": [12.0, 14.0, 16.0],
        "loss_cooldown": [2, 3],
    }
    evaluate_params(symbols, param_grid)
