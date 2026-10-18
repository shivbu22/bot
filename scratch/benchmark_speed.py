import sys
import time
sys.path.insert(0, ".")
from nnfx.config import BacktestConfig, IndicatorConfig, RiskConfig
from nnfx.synthetic_data import generate_synthetic_forex_data
from nnfx.backtester import Backtester

risk_cfg = RiskConfig(
    risk_pct=0.015,
    sl_atr_multiplier=2.2,
    tp1_atr_multiplier=0.8,
    loss_cooldown_bars=2,
)
ind_cfg = IndicatorConfig(
    baseline_type="EMA",
    baseline_period=200,
    baseline_max_atr_dist=1.5,
    ssl_period=10,
    c2_enabled=True,
    vortex_period=14,
    min_c2_spread=0.15,
    adx_enabled=True,
    min_adx=12.0,
    allow_bridge_trades=True,
    max_bridge_bars=5,
    min_bridge_candle_atr=1.0,
    wae_dead_zone_atr_mult=0.75,
    exit_oscillator_period=14,
)
bt_cfg = BacktestConfig(initial_capital=100_000.0, risk=risk_cfg, indicators=ind_cfg)

print("[*] Generating 10,000 bars...")
t0 = time.time()
df = generate_synthetic_forex_data(n_bars=10000, start_price=2500.0, seed=42)
t1 = time.time()
print(f"Data generated in {t1 - t0:.2f}s")

print("[*] Running Backtester on 10,000 bars...")
bt = Backtester(bt_cfg)
metrics = bt.run(df)
t2 = time.time()
print(f"Backtest completed in {t2 - t1:.2f}s")
print(f"Total Trades: {metrics.total_trades} | Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades}")
print(f"Win Rate: {metrics.win_rate_pct:.2f}% | Profit: ${metrics.total_net_profit:+,.2f} | PF: {metrics.profit_factor:.2f}")
