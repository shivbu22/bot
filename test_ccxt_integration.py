"""Tests for CCXT data feed, real-time fetching, and live trader integration."""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from nnfx.config import ExchangeConfig, BacktestConfig
from nnfx.ccxt_feed import CCXTDataFeed
from nnfx.ccxt_live_trader import CCXTLiveTrader


class TestCCXTDataFeed(unittest.TestCase):
    """Tests CCXT feed schema, timeframe conversion, and mock pagination."""

    def setUp(self):
        self.config = ExchangeConfig(
            exchange_id="kraken",
            symbol="BTC/USD",
            timeframe="1h",
            data_cache_dir="data_test",
        )

    def test_timeframe_to_ms(self):
        feed = CCXTDataFeed(self.config)
        self.assertEqual(feed._timeframe_to_ms("1m"), 60_000)
        self.assertEqual(feed._timeframe_to_ms("15m"), 15 * 60_000)
        self.assertEqual(feed._timeframe_to_ms("1h"), 3_600_000)
        self.assertEqual(feed._timeframe_to_ms("1d"), 86_400_000)

    def test_mock_fetch_historical_ohlcv(self):
        # Generate 50 mock CCXT candles: [timestamp_ms, open, high, low, close, volume]
        base_ts = 1735689600000 # 2025-01-01 00:00:00 UTC
        mock_candles = []
        price = 90000.0
        for i in range(50):
            ts = base_ts + (i * 3600000)
            mock_candles.append([ts, price, price + 500, price - 500, price + 100, 15.0])
            price += 100

        feed = CCXTDataFeed(self.config)
        feed.exchange.fetch_ohlcv = MagicMock(return_value=mock_candles)
        df = feed.fetch_historical_ohlcv(limit=50, use_cache=False)

        self.assertEqual(len(df), 50)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("open", df.columns)
        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)
        self.assertTrue((df["high"] >= df["open"]).all())
        self.assertTrue((df["low"] <= df["open"]).all())


class TestCCXTLiveTrader(unittest.TestCase):
    """Tests CCXT live trader single-step evaluation and signal handling."""

    @patch("nnfx.ccxt_feed.CCXTDataFeed.fetch_historical_ohlcv")
    def test_single_poll_step_evaluation(self, mock_fetch):
        # Generate 250 mock bars so baseline (200) has sufficient warmup
        base_time = datetime(2025, 1, 1, 0, 0)
        dates = [base_time + timedelta(hours=i) for i in range(250)]
        price = 90000.0
        records = []
        for i, dt in enumerate(dates):
            records.append({
                "open": price,
                "high": price + 200,
                "low": price - 200,
                "close": price + 50,
                "volume": 100.0,
            })
            price += 20.0

        mock_df = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
        mock_fetch.return_value = mock_df

        trader = CCXTLiveTrader()
        trader.initialize(warmup_bars=250)

        # Mock the next poll step returning recent 5 bars with one new bar
        recent_dates = [dates[-1] + timedelta(hours=i + 1) for i in range(5)]
        recent_records = []
        for dt in recent_dates:
            recent_records.append({
                "open": price,
                "high": price + 200,
                "low": price - 200,
                "close": price + 50,
                "volume": 100.0,
            })
            price += 20.0

        mock_recent_df = pd.DataFrame(recent_records, index=pd.DatetimeIndex(recent_dates))
        mock_fetch.return_value = mock_recent_df

        step_result = trader.poll_and_evaluate_step()
        self.assertIn(step_result["status"], ["bar_processed", "candle_still_forming"])


if __name__ == "__main__":
    unittest.main()
