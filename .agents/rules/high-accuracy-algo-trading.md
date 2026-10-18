# High-Accuracy Algorithmic Trading Invariants

When developing, calibrating, or optimizing trading algorithms and live execution bots:

1. **Bridge Trade Impulse Invariant**:
   - Never allow entry on baseline crosses without checking candle impulse momentum.
   - For bridge entries where C1 is already active, require candle range >= 1.0 * ATR and reject if more than 5 bars have elapsed since the initial flip. Sluggish drift across the baseline is the primary cause of false breakouts in crypto.

2. **Secondary Confirmation Spread Invariant**:
   - Secondary confirmation filters (e.g., Vortex C2, Aroon) must not merely check polarity (VI+ > VI-). They must require a minimum spread |VI+ - VI-| >= 0.15 to reject indecisive consolidation chop.

3. **Zero-Lookahead Live Execution Parity**:
   - In CCXT live stream loops, always treat index [-2] (penultimate candle) as the finalized closed bar for indicator calculation, and index [-1] (forming candle open) for simulated or real execution fills.
   - Maintain 100% mathematical parity between backtester logic and live step evaluation.

4. **Performance & Vectorization Invariant**:
   - Avoid pandas `.iloc` inside tight lookback loops across thousands of bars. Extract columns to pure NumPy arrays (`df['col'].values`) before looping to prevent microsecond-per-call overhead.

5. **Risk & Capital Preservation Invariant**:
   - Enforce fixed fractional risk sizing (1.5% max account equity risk per trade).
   - Implement post-loss cooldown windows (e.g., 2 bars) to prevent entering immediate counter-trend whipsaws.
   - Enforce two-tranche scaling (Scale 1 for TP1 profit lock, Scale 2 runner with Breakeven shift and trailing stops).
