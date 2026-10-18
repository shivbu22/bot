import sys
sys.path.insert(0, ".")
from nnfx.ccxt_feed import CCXTDataFeed
from nnfx.config import ExchangeConfig, IndicatorConfig, RiskConfig, BacktestConfig
from nnfx.indicator_engine import IndicatorEngine
from nnfx.algorithm import NNFXAlgorithm
import pandas as pd

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", paper_trading=True))

risk_cfg = RiskConfig(
    risk_pct=0.015,
    atr_period=14,
    sl_atr_multiplier=2.2,
    tp1_atr_multiplier=0.8,
    tranche_1_pct=0.50,
    tranche_2_pct=0.50,
    move_to_be_on_tp1=True,
    loss_cooldown_bars=2,
)

indicator_cfg = IndicatorConfig(
    baseline_type="EMA",
    baseline_period=200,
    baseline_max_atr_dist=1.5,
    ssl_period=10,
    c2_enabled=True,
    vortex_period=14,
    min_c2_spread=0.15,
    adx_enabled=True,
    adx_period=14,
    min_adx=12.0,
    max_candle_atr_ratio=2.0,
    allow_bridge_trades=True,
    max_bridge_bars=5,
    min_bridge_candle_atr=1.0,
    wae_dead_zone_atr_mult=0.75,
    exit_oscillator_period=14,
    exit_ob_level=0.70,
    exit_os_level=0.30,
    runner_trailing_atr_mult=1.5,
)

ind_engine = IndicatorEngine(indicator_cfg)
algo = NNFXAlgorithm(indicator_cfg, risk_cfg)

print("=" * 80)
print("             REAL-TIME BINANCE TELEMETRY & NNFX STATUS REPORT             ")
print("=" * 80)

for symbol in ["ETH/USDT", "BTC/USDT"]:
    df = feed.fetch_historical_ohlcv(symbol=symbol, timeframe="1h", limit=250, force_download=True)
    enriched = ind_engine.compute_all(df)
    
    # -2 is latest closed candle, -1 is forming candle
    closed_idx = len(enriched) - 2
    forming_idx = len(enriched) - 1
    
    closed_bar = enriched.iloc[closed_idx]
    forming_bar = enriched.iloc[forming_idx]
    
    decision = algo.evaluate_bar(enriched, current_idx=closed_idx, active_position_dir=0, bars_since_last_loss=999)
    
    c1_dir = "BULLISH (+1)" if closed_bar["ssl_dir"] == 1 else ("BEARISH (-1)" if closed_bar["ssl_dir"] == -1 else "NEUTRAL (0)")
    c2_spread = abs(closed_bar["vortex_plus"] - closed_bar["vortex_minus"])
    c2_dir = "BULLISH" if closed_bar["vortex_plus"] > closed_bar["vortex_minus"] else "BEARISH"
    wae_status = "BULL VOL" if closed_bar.get("wae_bull_vol", False) else ("BEAR VOL" if closed_bar.get("wae_bear_vol", False) else "NO EXPLOSION")
    
    print(f"\n[ASSET: {symbol}]")
    print(f"  Closed Candle Time:     {closed_bar.name} UTC")
    print(f"  Closed Candle Close:    ${closed_bar['close']:,.2f}")
    print(f"  Current Live Price:     ${forming_bar['close']:,.2f} (Forming candle open: ${forming_bar['open']:,.2f})")
    print(f"  ATR (14):               ${closed_bar['atr']:,.2f}")
    print(f"  1. Baseline (200 EMA):  ${closed_bar['baseline']:,.2f} | Distance: {closed_bar['baseline_atr_dist']:.2f} ATR (Max Allowed: 1.50 ATR)")
    print(f"  2. Confirmation 1 (SSL):{c1_dir} | Flip: {closed_bar['ssl_flip']}")
    print(f"  3. Confirmation 2 (VTX):{c2_dir} (Spread: {c2_spread:.3f} vs Min Req: {indicator_cfg.min_c2_spread:.3f})")
    print(f"  4. Trend Strength (ADX):{closed_bar['adx']:.1f} (Min Req: {indicator_cfg.min_adx:.1f})")
    print(f"  5. Volume/Explosion(WAE):{wae_status} (Power: {closed_bar['wae_trend_power']:.1f} vs Explosion: {closed_bar['wae_explosion']:.1f})")
    print(f"  6. Exit Oscillator (DeM):{closed_bar['exit_oscillator']:.3f}")
    print(f"  -------------------------------------------------------------")
    print(f"  Algorithm Decision:     {'VALID ENTRY SIGNAL' if decision.is_valid else 'STAND ASIDE / NO ENTRY'}")
    if not decision.is_valid:
        print(f"  Reason For No Trade:    {decision.rejection_reason}")
    else:
        print(f"  Signal Direction:       {'LONG' if decision.direction == 1 else 'SHORT'}")
print("\n" + "=" * 80)
