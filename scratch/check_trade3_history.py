import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nnfx.config import BacktestConfig, RiskConfig, IndicatorConfig, ExchangeConfig
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed

feed = CCXTDataFeed(ExchangeConfig(exchange_id="binance", symbol="ETH/USDT", timeframe="1h"))
df = feed.fetch_historical_ohlcv(symbol="ETH/USDT", timeframe="1h", limit=1000, use_cache=True)

b = Backtester()
ind_df = b.indicator_engine.compute_all(df)

idx_766 = 766
print(f"Candle 766 ({df.index[766]}): Close={ind_df.iloc[766]['close']}, Baseline={ind_df.iloc[766]['baseline']:.2f}")

for back in range(766, 740, -1):
    r = ind_df.iloc[back]
    print(f"  Bar {back} ({df.index[back]}): Close={r['close']:.2f}, Baseline={r['baseline']:.2f}, SSL_Dir={r['ssl_dir']}, SSL_Flip={r['ssl_flip']}")
