import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed
import pandas as pd

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)

b = Backtester(BacktestConfig(
    risk=RiskConfig(sl_atr_multiplier=2.2, tp1_atr_multiplier=0.8, loss_cooldown_bars=2),
    indicators=IndicatorConfig(adx_enabled=True, min_adx=12.0)
))
ind_df = b.indicator_engine.compute_all(df)
m = b.run(df)

trades_info = []
for t in b.portfolio.completed_trades:
    e_idx = df.index.get_loc(t.entry_time)
    sig_idx = e_idx - 1
    sig = ind_df.iloc[sig_idx]
    prev_sig = ind_df.iloc[sig_idx - 1]
    
    # Calculate additional technical factors:
    # 1. RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 2. Distance from baseline in ATR
    dist_atr = abs(sig["close"] - sig["baseline"]) / sig["atr"]
    
    # 3. Baseline 200 vs Baseline 100 / 50 EMA
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    ema100 = df["close"].ewm(span=100, adjust=False).mean()
    
    # 4. Vortex spread
    vx_spread = sig["vortex_plus"] - sig["vortex_minus"]
    
    # 5. Direction of candle (bullish or bearish close vs open)
    candle_body_dir = 1 if sig["close"] > sig["open"] else (-1 if sig["close"] < sig["open"] else 0)
    
    # 6. Distance from close to high/low wick (rejection wick)
    upper_wick = sig["high"] - max(sig["open"], sig["close"])
    lower_wick = min(sig["open"], sig["close"]) - sig["low"]
    
    trades_info.append({
        "id": t.trade_id,
        "res": "WIN" if t.total_pnl_dollars > 0 else "LOSS",
        "dir": "LONG" if t.direction == 1 else "SHORT",
        "entry_time": t.entry_time,
        "ssl_flip": sig["ssl_flip"],
        "dist_atr": dist_atr,
        "rsi": rsi.iloc[sig_idx],
        "vx_spread": vx_spread,
        "candle_body_dir": candle_body_dir,
        "upper_wick_ratio": upper_wick / sig["atr"],
        "lower_wick_ratio": lower_wick / sig["atr"],
        "ema50_vs_baseline": 1 if ema50.iloc[sig_idx] > sig["baseline"] else -1,
        "ema100_vs_baseline": 1 if ema100.iloc[sig_idx] > sig["baseline"] else -1,
        "demarker": sig["exit_oscillator"]
    })

res_df = pd.DataFrame(trades_info)
print(res_df.to_string())
