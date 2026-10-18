# No Nonsense Forex (NNFX) Algorithmic Trading Framework & Backtester

A complete, production-ready quantitative algorithmic trading system and event-driven backtester built strictly upon the **No Nonsense Forex (NNFX)** methodology.

---

## 1. System Architecture

The codebase is engineered with strict separation of concerns into decoupled, modular components:

```
bot/
├── nnfx/
│   ├── __init__.py           # Package namespace and clean public exports
│   ├── config.py             # Typed dataclasses for risk, indicators, execution, and backtest configs
│   ├── data_loader.py        # Validates OHLCV columns, timestamp monotonicity, and data integrity
│   ├── indicator_engine.py   # Vectorized indicator math (ATR, EMA, SSL Channel, WAE, DeMarker)
│   ├── risk_manager.py       # ATR-based dynamic position sizing, SL/TP1, and 50/50 tranche splits
│   ├── algorithm.py          # Multi-layer signal evaluation, distance filter, and rejection logs
│   ├── execution.py          # Bar t+1 open fills, intrabar stop/limit execution, BE & trailing stops
│   ├── metrics.py            # Quantitative performance metrics (Sharpe, Sortino, Calmar, Drawdown)
│   ├── synthetic_data.py     # Realistic regime-switching Forex OHLCV time-series generator
│   └── backtester.py         # Sequential, lookahead-free event-driven backtesting driver
├── main.py                   # Out-of-the-box CLI runner with tabular reporting and plotting
├── test_nnfx.py              # Unit & integration test suite
└── README.md                 # System documentation
```

---

## 2. NNFX Rules & Strategy Logic

1. **Baseline Filter (EMA 200 / Kijun-sen)**
   - **Directionality:** Long trades permitted only when price closes above the baseline (`Close > Baseline`). Short trades permitted only when price closes below (`Close < Baseline`).
   - **Extension Rule (Distance Filter):** Rejects entries if price is extended further than **$1.5 \times \text{ATR}(14)$** from the baseline.

2. **Confirmation Indicator (C1 - SSL Channel)**
   - Tracks dynamic price highs and lows using rolling channels.
   - Signals a trend entry when the channel flips direction (or when price crosses the baseline while C1 is already aligned).

3. **Volume / Volatility Filter (Waddah Attar Explosion - WAE)**
   - Evaluates MACD trend acceleration against Bollinger Band width (Explosion Line) and an ATR-based Dead Zone threshold.
   - Blocks trade execution during low-volume or sideways consolidation.

4. **Exit Indicator (DeMarker Oscillator / Fast Reversal Flip)**
   - Triggers an early market exit for the Scale 2 Runner tranche when momentum is exhausted before hitting an adverse stop.

---

## 3. Money Management & Execution Engine

Every trade is partitioned into two equal tranches:
- **Tranche 1 (`TP1_Tranche` - 50% size):** Take Profit target set at exactly **$1.0 \times \text{ATR}(14)$**.
- **Tranche 2 (`Runner_Tranche` - 50% size):** Trend-following runner.
- **Initial Stop Loss:** Set at exactly **$1.5 \times \text{ATR}(14)$** from the fill price.
- **Risk Sizing:** Units are calculated dynamically such that if the initial Stop Loss is hit, total dollar loss equals exactly **$1.5\%$** (configurable $1-2\%$) of current account equity.
- **Breakeven Stop Adjustment:** As soon as Tranche 1 reaches TP1, the Stop Loss of Tranche 2 is automatically moved to **Breakeven** (`Entry Price`).
- **Trailing Stop:** While in profit, the Runner trails the market at $1.5 \times \text{ATR}$ or closes immediately if the Exit Indicator fires.

---

## 4. Zero Lookahead Bias

- **Signal Timestamp:** Indicators and signals are calculated strictly on the close of bar $t$.
- **Execution Timestamp:** Order fill occurs at the **Open of bar $t+1$** with realistic spread and slippage.
- **Intrabar Testing:** Bar $t+1$ High and Low are evaluated against Stop Loss and Take Profit levels with conservative worst-case conflict resolution.

---

## 5. Quickstart & Usage

### 1. Fetch Real Market Data via CCXT & Backtest
Download real historical candles directly from exchanges (Binance, Kraken, Bybit, Coinbase, OKX) with automated caching:
```bash
# Download 1,000 bars of real BTC/USDT 1h data from Binance and run backtest
python main.py --fetch-real --symbol BTC/USDT --exchange binance --timeframe 1h --bars 1000 --plot

# Or test ETH/USDT on Kraken
python main.py --fetch-real --symbol ETH/USD --exchange kraken --timeframe 1h --bars 1000 --plot
```

### 2. Real-Time Live / Paper Trading Loop
Monitor live bar closes and execute simulated two-tranche paper trades:
```bash
python main.py --live --paper --exchange binance --symbol BTC/USDT --timeframe 1h --poll-interval 10
```

### 3. Running with Synthetic Market Data
```bash
python main.py --bars 2000 --verbose --plot
```

### 4. Running with Custom Historical CSV Data
```bash
python main.py --data path/to/market_data.csv --capital 100000 --risk 0.015 --plot
```

### 5. Trained Hyper-Accuracy & Dynamic Profit Ratchet Preset
Run the trained hyper-accuracy and profit-maximizing engine across single or multiple assets simultaneously:
```bash
# Trained Hyper-Accuracy Preset (SL=2.5x, TP1=0.8x, Dynamic Profit Ratchet, 25% TP1 / 75% Runner, ADX>=14, C2>=0.18)
python main.py --fetch-real --symbols BTC/USDT,ETH/USDT,SOL/USDT --exchange binance --bars 2000 --hyper-accuracy --plot

# Continuous 1,000 Paper Trades Evaluation
uv run python scratch/run_1000_paper_trade_test.py

# High-Win-Rate Calibrated Preset (SL=2.2, TP1=0.8, ADX>=12, Bridge bars<=5, Candle>=0.8 ATR)
python main.py --fetch-real --symbols BTC/USDT,ETH/USDT,SOL/USDT --exchange binance --bars 1000 --high-win-rate --plot
```

### 6. Precision Filter Options
- `--strict-c1`: Enforces pure C1 momentum flips, eliminating false baseline bridge crosses.
- `--max-bridge-bars <N>`: Maximum bars elapsed since C1 flip for a bridge entry to remain valid (default: 5).
- `--min-bridge-candle-atr <N>`: Minimum candle impulse range (High - Low) / ATR for bridge entry (default: 0.8).
- `--min-adx <N>`: Wilder's ADX regime threshold to filter choppy markets (default: 12.0 in high-win-rate mode).
- `--loss-cooldown <N>`: Pauses new entries for N bars following an initial stop loss to avoid whipsaws.

### 7. Strategy Parameter Optimization & Tuning
Run automated grid search with out-of-sample walk-forward validation:
```bash
python main.py --fetch-real --symbol BTC/USDT --exchange binance --timeframe 1h --bars 1000 --optimize
```

### 8. Live Real-Time & Paper Trading on Binance
Run the live continuous monitoring and execution loop with real-time indicator telemetry:
```bash
# Live Paper Trading on Binance (Real-time live market feed, simulated zero-risk execution)
uv run python main.py --live --paper --symbol ETH/USDT --high-win-rate --poll-interval 10

# Live Real Money Trading on Binance (Using API credentials configured in .env)
uv run python main.py --live --real --symbol ETH/USDT --high-win-rate --poll-interval 10
```

### 9. Live Model Trade Accuracy & Parity Verification
Verify 100% mathematical parity and win-rate accuracy between the offline backtester and the live trading engine:
```bash
uv run python test_live_accuracy.py
```

### 10. Running Automated Unit & Integration Tests
```bash
uv run python -m unittest test_nnfx.py test_ccxt_integration.py
```


