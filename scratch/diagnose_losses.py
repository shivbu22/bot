import sys
from pathlib import Path
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig, ExecutionConfig
from nnfx.backtester import Backtester
from nnfx.data_loader import DataLoader
from nnfx.indicator_engine import IndicatorEngine

def diagnose_losses():
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
    )
    
    engine = IndicatorEngine(cfg.indicators)
    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "ADA_USDT", "XRP_USDT"]
    
    loss_details = []
    
    for tf in ["1h", "4h"]:
        for s in symbols:
            p = Path(f"data/binance_{s}_{tf}.csv")
            if not p.exists():
                continue
            df = DataLoader.load(p)
            enriched = engine.compute_all(df)
            bt = Backtester(cfg)
            bt.run(df, enriched_df=enriched)
            
            for t in bt.portfolio.completed_trades:
                if t.total_pnl_dollars <= 0:
                    entry_bar = t.tranches[0].entry_time if t.tranches else "N/A"
                    # Find bar in enriched
                    try:
                        idx = enriched.index.get_loc(entry_bar)
                        row = enriched.iloc[idx]
                        c2_spread = abs(row["vortex_plus"] - row["vortex_minus"])
                        adx = row["adx"]
                        slope = row["baseline_slope"]
                        dist = row["baseline_atr_dist"]
                        candle_atr = row["candle_atr_ratio"]
                        c1_flip = row["ssl_flip"]
                        loss_details.append([
                            f"{s} {tf}",
                            str(entry_bar),
                            "LONG" if t.direction == 1 else "SHORT",
                            f"{t.total_pnl_dollars:,.2f}",
                            f"{adx:.1f}",
                            f"{c2_spread:.3f}",
                            f"{dist:.2f}",
                            f"{candle_atr:.2f}",
                            f"{slope:.4f}",
                            "C1 Flip" if c1_flip != 0 else "Bridge",
                        ])
                    except Exception as e:
                        loss_details.append([f"{s} {tf}", str(entry_bar), str(t.direction), f"{t.total_pnl_dollars:,.2f}", str(e), "", "", "", "", ""])

    print("=" * 100)
    print(" DIAGNOSTIC ANALYSIS OF ALL LOSING TRADES IN PORTFOLIO")
    print("=" * 100)
    headers = ["Asset & TF", "Entry Time", "Dir", "Loss ($)", "ADX", "C2 Spread", "Base Dist", "Candle/ATR", "Slope", "Setup Type"]
    print(tabulate(loss_details, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    diagnose_losses()
