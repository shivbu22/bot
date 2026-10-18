import sys
from pathlib import Path
from tabulate import tabulate
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig
from nnfx.backtester import Backtester
from nnfx.data_loader import DataLoader
from nnfx.indicator_engine import IndicatorEngine

def test_hyper_preset():
    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "ADA_USDT", "XRP_USDT"]
    timeframes = ["1h", "4h"]
    
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        risk=RiskConfig(
            risk_pct=0.02,
            sl_atr_multiplier=2.5,
            tp1_atr_multiplier=1.0,
            tranche_1_pct=0.25,
            tranche_2_pct=0.75,
            move_to_be_on_tp1=True,
            loss_cooldown_bars=2,
        ),
        indicators=IndicatorConfig(
            baseline_type="EMA",
            baseline_period=200,
            baseline_max_atr_dist=1.5,
            ssl_period=10,
            c2_enabled=True,
            vortex_period=14,
            min_c2_spread=0.18,
            adx_enabled=True,
            adx_period=14,
            min_adx=14.0,
            allow_bridge_trades=True,
            max_bridge_bars=5,
            min_bridge_candle_atr=1.1,
            wae_dead_zone_atr_mult=0.75,
            exit_oscillator_period=14,
            exit_ob_level=0.85,
            exit_os_level=0.15,
            runner_trailing_atr_mult=2.5,
        ),
        execution=ExecutionConfig(spread_pips=0.00015, slippage_pips=0.00005),
    )
    
    engine = IndicatorEngine(cfg.indicators)
    
    all_trades = []
    summary_rows = []
    
    for tf in ["1h", "4h"]:
        for s in symbols:
            p = Path(f"data/binance_{s}_{tf}.csv")
            if not p.exists():
                continue
            df = DataLoader.load(p)
            enriched = engine.compute_all(df)
            bt = Backtester(cfg)
            m = bt.run(df, enriched_df=enriched)
            
            for t in bt.portfolio.completed_trades:
                all_trades.append({
                    "asset": f"{s} {tf}",
                    "id": t.trade_id,
                    "direction": "LONG" if t.direction == 1 else "SHORT",
                    "pnl": t.total_pnl_dollars,
                    "ret_pct": t.return_pct,
                    "bars": t.tranches[-1].holding_bars if t.tranches else 0,
                    "tp1_hit": t.tp1_hit,
                    "exit_reason": t.tranches[-1].exit_reason.value if t.tranches else "N/A"
                })
                
            summary_rows.append([
                f"{s} {tf}",
                f"{m.winning_trades}W / {m.losing_trades}L",
                f"{m.win_rate_pct:.1f}%",
                f"${m.total_net_profit:+,.2f}",
                f"{m.profit_factor:.2f}" if m.profit_factor < 900 else "INF",
                f"{m.max_drawdown_pct:.2f}%",
            ])
            
    print("=" * 80)
    print(" CURRENT TRAINED HYPER-ACCURACY PERFORMANCE ACROSS ALL BINANCE PAIRS & TIMEFRAMES")
    print("=" * 80)
    headers = ["Asset & TF", "Record (W/L)", "Win Rate %", "Net Profit", "Profit Factor", "Max DD %"]
    print(tabulate(summary_rows, headers=headers, tablefmt="grid"))
    
    total_w = sum(1 for t in all_trades if t["pnl"] > 0)
    total_l = sum(1 for t in all_trades if t["pnl"] <= 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    high_profit_trades = [t for t in all_trades if t["ret_pct"] >= 10.0]
    mega_profit_trades = [t for t in all_trades if t["ret_pct"] >= 13.0]
    
    print(f"\nPortfolio Total: {total_w} Wins / {total_l} Losses ({total_w/(total_w+total_l)*100:.1f}% Win Rate)")
    print(f"Total Portfolio Net Profit: ${total_pnl:+,.2f}")
    print(f"Trades with >10% return: {len(high_profit_trades)} / {len(all_trades)}")
    print(f"Trades with >13% return: {len(mega_profit_trades)} / {len(all_trades)}")
    
    print("\nIndividual Trade Details:")
    trade_table = []
    for t in all_trades:
        trade_table.append([
            t["asset"], t["direction"], f"${t['pnl']:+,.2f}", f"{t['ret_pct']:+.2f}%",
            t["bars"], t["exit_reason"]
        ])
    print(tabulate(trade_table, headers=["Asset", "Dir", "PnL ($)", "Return (%)", "Holding Bars", "Exit Reason"], tablefmt="simple"))

if __name__ == "__main__":
    test_hyper_preset()
