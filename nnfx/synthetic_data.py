"""Synthetic market data generator simulating realistic financial OHLCV dynamics."""

from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


def generate_synthetic_forex_data(
    n_bars: int = 1500,
    start_price: float = 1.1000,
    timeframe_minutes: int = 60,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic OHLCV price series for Forex/Crypto pairs.
    Features:
    - 3 market regimes: Bullish trend, Bearish trend, and Ranging consolidation
    - Volatility clustering with autoregressive volatility
    - Valid intrabar OHLC relationships: High >= max(Open, Close), Low <= min(Open, Close)
    - Realistic volume proportional to price range
    - Monotonically increasing DatetimeIndex

    Parameters
    ----------
    n_bars : int
        Number of candles to simulate (default: 1500 bars).
    start_price : float
        Initial base price (e.g., 1.1000 for EUR/USD).
    timeframe_minutes : int
        Bar interval in minutes (default: 60 min / H1).
    seed : Optional[int]
        Random seed for deterministic reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', 'volume' and DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Simulate Markov regime states (0: Bull Trend, 1: Bear Trend, 2: Ranging)
    transition_matrix = np.array([
        [0.96, 0.02, 0.02],  # Bull trend persistence
        [0.02, 0.96, 0.02],  # Bear trend persistence
        [0.05, 0.05, 0.90],  # Ranging persistence
    ])

    regimes = np.zeros(n_bars, dtype=int)
    curr_regime = 0
    for i in range(n_bars):
        regimes[i] = curr_regime
        curr_regime = np.random.choice([0, 1, 2], p=transition_matrix[curr_regime])

    # Drift and baseline volatility per regime
    drift_map = {0: 0.00025, 1: -0.00025, 2: 0.00000}
    vol_base_map = {0: 0.0018, 1: 0.0022, 2: 0.0010}

    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    closes = np.zeros(n_bars)
    volumes = np.zeros(n_bars)

    current_close = start_price
    current_vol = 0.0015

    for i in range(n_bars):
        regime = regimes[i]
        target_drift = drift_map[regime]
        target_vol = vol_base_map[regime]

        # AR(1) volatility clustering
        current_vol = 0.85 * current_vol + 0.15 * target_vol + np.random.normal(0, 0.0002)
        current_vol = max(current_vol, 0.0005)

        # Candle open with tiny potential gap
        gap = np.random.normal(0, current_vol * 0.1) * current_close
        open_price = current_close + gap

        # Candle close log-return
        ret = target_drift + np.random.normal(0, current_vol)
        close_price = open_price * np.exp(ret)

        # Intrabar excursions
        candle_body = abs(close_price - open_price)
        upper_wick = abs(np.random.normal(0, current_vol * 0.75)) * current_close
        lower_wick = abs(np.random.normal(0, current_vol * 0.75)) * current_close

        high_price = max(open_price, close_price) + upper_wick
        low_price = min(open_price, close_price) - lower_wick

        # Realistic volume
        candle_range = high_price - low_price
        base_volume = 1000 + (candle_range / current_close) * 200_000
        vol_noise = np.random.uniform(0.8, 1.2)
        volume = max(100.0, base_volume * vol_noise)

        opens[i] = open_price
        highs[i] = high_price
        lows[i] = low_price
        closes[i] = close_price
        volumes[i] = volume

        current_close = close_price

    # Generate timestamp index
    start_time = datetime(2025, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=i * timeframe_minutes) for i in range(n_bars)]

    df = pd.DataFrame(
        {
            "open": np.round(opens, 5),
            "high": np.round(highs, 5),
            "low": np.round(lows, 5),
            "close": np.round(closes, 5),
            "volume": np.round(volumes, 1),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    # Final sanity checks
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df
