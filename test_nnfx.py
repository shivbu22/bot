"""Comprehensive unit and system test suite for the NNFX Algorithmic Trading Framework."""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from nnfx.config import RiskConfig, IndicatorConfig, ExecutionConfig, BacktestConfig
from nnfx.data_loader import DataLoader, DataValidationError
from nnfx.indicator_engine import IndicatorEngine
from nnfx.risk_manager import RiskManager
from nnfx.algorithm import NNFXAlgorithm, SignalType
from nnfx.execution import PortfolioTracker, ExitReason
from nnfx.synthetic_data import generate_synthetic_forex_data
from nnfx.backtester import Backtester
from nnfx.metrics import MetricsCalculator, PerformanceMetrics
from nnfx.optimizer import StrategyOptimizer


class TestDataLoader(unittest.TestCase):
    """Tests data ingestion and OHLC validation rules."""

    def test_valid_dataframe(self):
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(10)]
        df = pd.DataFrame({
            "open": [1.10] * 10,
            "high": [1.12] * 10,
            "low": [1.08] * 10,
            "close": [1.11] * 10,
            "volume": [500] * 10,
        }, index=pd.DatetimeIndex(dates))

        loaded = DataLoader.load(df)
        self.assertEqual(len(loaded), 10)
        self.assertIn("open", loaded.columns)
        self.assertIn("close", loaded.columns)

    def test_invalid_high_raises(self):
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)]
        # High is below Open
        df = pd.DataFrame({
            "open": [1.10, 1.15, 1.10, 1.10, 1.10],
            "high": [1.12, 1.11, 1.12, 1.12, 1.12], # index 1 invalid: high 1.11 < open 1.15
            "low": [1.08] * 5,
            "close": [1.09] * 5,
        }, index=pd.DatetimeIndex(dates))

        with self.assertRaises(DataValidationError):
            DataLoader.load(df)

    def test_missing_column_raises(self):
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)]
        df = pd.DataFrame({
            "open": [1.10] * 5,
            "high": [1.12] * 5,
            "low": [1.08] * 5,
            # missing close
        }, index=pd.DatetimeIndex(dates))

        with self.assertRaises(DataValidationError):
            DataLoader.load(df)


class TestRiskManager(unittest.TestCase):
    """Tests NNFX money management, ATR position sizing, and two-tranche allocation."""

    def setUp(self):
        self.cfg = RiskConfig(
            risk_pct=0.015,         # 1.5%
            sl_atr_multiplier=1.5,
            tp1_atr_multiplier=1.0,
            tranche_1_pct=0.5,
            tranche_2_pct=0.5,
        )
        self.rm = RiskManager(self.cfg)

    def test_long_position_sizing_and_targets(self):
        equity = 100_000.0
        entry_price = 1.1000
        atr = 0.0050 # 50 pips
        direction = 1 # Long

        specs = self.rm.calculate_position(equity, entry_price, atr, direction)

        # Expected risk in dollars = $1,500
        self.assertAlmostEqual(specs.risk_amount_dollars, 1500.0)

        # SL Distance = 1.5 * ATR = 0.0075
        expected_sl = 1.1000 - 0.0075 # 1.0925
        self.assertAlmostEqual(specs.initial_sl, expected_sl)

        # TP1 Distance = 1.0 * ATR = 0.0050
        expected_tp1 = 1.1000 + 0.0050 # 1.1050
        self.assertAlmostEqual(specs.tp1_price, expected_tp1)

        # Total Units = $1,500 / 0.0075 = 200,000 units
        self.assertAlmostEqual(specs.total_units, 200_000.0)
        self.assertAlmostEqual(specs.tranche_1_units, 100_000.0)
        self.assertAlmostEqual(specs.tranche_2_units, 100_000.0)

        # Verify exact risk loss if SL hit
        loss_at_sl = (entry_price - specs.initial_sl) * specs.total_units
        self.assertAlmostEqual(loss_at_sl, 1500.0)

    def test_short_position_sizing_and_targets(self):
        equity = 50_000.0
        entry_price = 1.2000
        atr = 0.0040
        direction = -1 # Short

        specs = self.rm.calculate_position(equity, entry_price, atr, direction)
        expected_risk = 50_000.0 * 0.015 # $750
        self.assertAlmostEqual(specs.risk_amount_dollars, expected_risk)

        # SL = entry + 1.5 * ATR
        expected_sl = 1.2000 + (1.5 * 0.0040) # 1.2060
        self.assertAlmostEqual(specs.initial_sl, expected_sl)

        # TP1 = entry - 1.0 * ATR
        expected_tp1 = 1.2000 - (1.0 * 0.0040) # 1.1960
        self.assertAlmostEqual(specs.tp1_price, expected_tp1)

        # Loss at SL
        loss = (specs.initial_sl - entry_price) * specs.total_units
        self.assertAlmostEqual(loss, expected_risk)

    def test_trailing_stop_monotonicity(self):
        # Long trailing stop
        current_sl = 1.1000
        current_atr = 0.0050
        # If price reaches higher high 1.1200, trail is 1.1200 - 1.5*0.0050 = 1.1125 > 1.1000 -> moves up
        sl1 = self.rm.calculate_trailing_stop(1, current_sl, 1.1200, 1.0900, current_atr)
        self.assertAlmostEqual(sl1, 1.1125)

        # If price pulls back, highest high remains 1.1200, stop must never move down
        sl2 = self.rm.calculate_trailing_stop(1, sl1, 1.1200, 1.0900, current_atr)
        self.assertAlmostEqual(sl2, 1.1125)

    def test_dynamic_profit_ratchet(self):
        entry_price = 100.0
        current_atr = 2.0
        # Tier 1 (+2 ATR = 104.0): locks +0.5 ATR (101.0)
        sl1 = self.rm.calculate_ratchet_stop(
            direction=1, entry_price=entry_price, current_sl=entry_price,
            highest_high_since_tp1=104.5, lowest_low_since_tp1=100.0, current_atr=current_atr
        )
        self.assertGreaterEqual(sl1, 101.0)

        # Tier 2 (+4 ATR = 108.0): locks +2.0 ATR (104.0)
        sl2 = self.rm.calculate_ratchet_stop(
            direction=1, entry_price=entry_price, current_sl=sl1,
            highest_high_since_tp1=109.0, lowest_low_since_tp1=100.0, current_atr=current_atr
        )
        self.assertGreaterEqual(sl2, 104.0)

        # Tier 3 (+8 ATR = 116.0): locks +5.0 ATR (110.0)
        sl3 = self.rm.calculate_ratchet_stop(
            direction=1, entry_price=entry_price, current_sl=sl2,
            highest_high_since_tp1=117.0, lowest_low_since_tp1=100.0, current_atr=current_atr
        )
        self.assertGreaterEqual(sl3, 110.0)


class TestIndicatorEngine(unittest.TestCase):
    """Tests technical indicator calculations and boundaries."""

    def test_indicators_on_synthetic_data(self):
        df = generate_synthetic_forex_data(n_bars=300, seed=123)
        engine = IndicatorEngine()
        enriched = engine.compute_all(df)

        self.assertIn("atr", enriched.columns)
        self.assertIn("baseline", enriched.columns)
        self.assertIn("ssl_dir", enriched.columns)
        self.assertIn("wae_trend_power", enriched.columns)
        self.assertIn("exit_oscillator", enriched.columns)

        # Check ATR is positive
        valid_atr = enriched["atr"].dropna()
        self.assertTrue((valid_atr > 0).all())

        # Check DeMarker bounds [0, 1]
        valid_demarker = enriched["exit_oscillator"].dropna()
        self.assertTrue((valid_demarker >= 0.0).all() and (valid_demarker <= 1.0).all())

    def test_adx_calculation(self):
        df = generate_synthetic_forex_data(n_bars=300, seed=42)
        cfg = IndicatorConfig(adx_enabled=True, adx_period=14, min_adx=15.0)
        engine = IndicatorEngine(cfg)
        enriched = engine.compute_all(df)

        self.assertIn("adx", enriched.columns)
        self.assertIn("plus_di", enriched.columns)
        self.assertIn("minus_di", enriched.columns)
        self.assertIn("adx_valid", enriched.columns)

        valid_adx = enriched["adx"].dropna()
        self.assertTrue((valid_adx >= 0.0).all() and (valid_adx <= 100.0).all())
        valid_pdi = enriched["plus_di"].dropna()
        self.assertTrue((valid_pdi >= 0.0).all() and (valid_pdi <= 100.0).all())


class TestNNFXAlgorithm(unittest.TestCase):
    """Tests NNFX multi-layer entry rules and distance filters."""

    def setUp(self):
        self.cfg = IndicatorConfig(baseline_max_atr_dist=1.5, baseline_slope_filter=True)
        self.algo = NNFXAlgorithm(self.cfg)

    def test_baseline_distance_filter_rejection(self):
        # Create single row where baseline distance is 2.0 ATR (> 1.5 ATR limit)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.12, 1.12, 1.12],
            "low": [1.08, 1.08, 1.08],
            "close": [1.10, 1.10, 1.12],
            "baseline": [1.10, 1.10, 1.08],
            "baseline_dir": [1, 1, 1],
            "atr": [0.01, 0.01, 0.01],
            "baseline_atr_dist": [0.0, 0.0, 4.0],  # 4.0 ATR distance!
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],                # Bullish flip
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = self.algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Price extended too far from baseline", decision.rejection_reason)

    def test_volume_filter_rejection(self):
        # Bullish flip and baseline pass, but volume filter is False
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.12, 1.12, 1.12],
            "low": [1.08, 1.08, 1.08],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "atr": [0.01, 0.01, 0.01],
            "baseline_atr_dist": [0.0, 0.0, 0.5], # 0.5 ATR distance (passes)
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],               # Bullish flip
            "wae_bull_vol": [False, False, False],# Volume filter FAIL
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = self.algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Volume filter invalid", decision.rejection_reason)

    def test_c2_filter_rejection(self):
        # C1 is bullish, but C2 is bearish (-1)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.0],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],               # Bullish flip
            "c2_dir": [-1, -1, -1],              # C2 is BEARISH
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = self.algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("C2 confirmation indicator is not bullish", decision.rejection_reason)

    def test_c2_spread_filter_rejection(self):
        cfg = IndicatorConfig(c2_enabled=True, min_c2_spread=0.15)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.0],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],
            "c2_dir": [1, 1, 1],
            "vortex_plus": [1.02, 1.03, 1.04],
            "vortex_minus": [0.98, 0.97, 0.95],   # Spread is 1.04 - 0.95 = 0.09 (< 0.15 limit)
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("C2 Vortex spread too weak", decision.rejection_reason)

    def test_giant_candle_filter_rejection(self):
        # Candle range is 3.5x ATR (> 2.0x limit)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.13, 1.13, 1.135],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [False, False, False], # False due to 3.5x ATR
            "candle_atr_ratio": [3.5, 3.5, 3.5],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = self.algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Giant exhaustion candle", decision.rejection_reason)

    def test_baseline_slope_filter_rejection(self):
        # Long setup, but baseline is sloping downward (-0.005)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.095],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, -0.005], # Downward slope!
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.0],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = self.algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Baseline is sloping downward", decision.rejection_reason)

    def test_adx_filter_rejection(self):
        cfg = IndicatorConfig(adx_enabled=True, min_adx=20.0)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.0],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "adx": [10.0, 12.0, 14.0],           # ADX 14.0 < min_adx 20.0!
            "adx_valid": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("ADX trend strength too low", decision.rejection_reason)

    def test_loss_cooldown_rejection(self):
        risk_cfg = RiskConfig(loss_cooldown_bars=4)
        algo = NNFXAlgorithm(IndicatorConfig(), risk_cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.105],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.0],
            "baseline_atr_dist": [0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [0, 0, 1],
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "adx": [25.0, 25.0, 25.0],
            "adx_valid": [True, True, True],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        # Cooldown active: only 2 bars since last SL hit (limit is 4)
        decision = algo.evaluate_bar(df, current_idx=2, active_position_dir=0, bars_since_last_loss=2)
        self.assertFalse(decision.is_valid)
        self.assertIn("Post-loss whipsaw cooldown active", decision.rejection_reason)

        # Cooldown expired: 5 bars since last SL hit
        decision_ok = algo.evaluate_bar(df, current_idx=2, active_position_dir=0, bars_since_last_loss=5)
        self.assertTrue(decision_ok.is_valid)

    def test_bridge_trade_strict_c1_rejection(self):
        cfg = IndicatorConfig(allow_bridge_trades=False)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(4)]
        df = pd.DataFrame({
            "open": [1.09, 1.09, 1.09, 1.10],
            "high": [1.11, 1.11, 1.11, 1.12],
            "low": [1.08, 1.08, 1.08, 1.09],
            "close": [1.09, 1.095, 1.098, 1.105],     # Bar 2 <= baseline (1.10), Bar 3 > baseline
            "baseline": [1.10, 1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01, 0.01],
            "candle_valid": [True, True, True, True],
            "candle_atr_ratio": [1.2, 1.2, 1.2, 1.2],
            "baseline_atr_dist": [0.0, 0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1, 1],
            "ssl_flip": [1, 0, 0, 0],                 # Flip at bar 0, no flip at bar 3
            "c2_dir": [1, 1, 1, 1],
            "wae_bull_vol": [True, True, True, True],
            "wae_bear_vol": [False, False, False, False],
            "exit_long_signal": [False, False, False, False],
            "exit_short_signal": [False, False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=3, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Bridge trade rejected (Strict C1 entry mode enabled)", decision.rejection_reason)

    def test_bridge_trade_expired_rejection(self):
        cfg = IndicatorConfig(allow_bridge_trades=True, max_bridge_bars=2)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)]
        # C1 flip at bar 0, bridge cross at bar 4 (4 bars elapsed > max 2)
        df = pd.DataFrame({
            "open": [1.09, 1.09, 1.09, 1.09, 1.10],
            "high": [1.11, 1.11, 1.11, 1.11, 1.12],
            "low": [1.08, 1.08, 1.08, 1.08, 1.09],
            "close": [1.09, 1.092, 1.094, 1.096, 1.105], # Crosses above 1.10 at bar 4
            "baseline": [1.10, 1.10, 1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01, 0.01, 0.01],
            "candle_valid": [True, True, True, True, True],
            "candle_atr_ratio": [1.2, 1.2, 1.2, 1.2, 1.2],
            "baseline_atr_dist": [0.0, 0.0, 0.0, 0.0, 0.5],
            "ssl_dir": [1, 1, 1, 1, 1],
            "ssl_flip": [1, 0, 0, 0, 0],
            "c2_dir": [1, 1, 1, 1, 1],
            "wae_bull_vol": [True, True, True, True, True],
            "wae_bear_vol": [False, False, False, False, False],
            "exit_long_signal": [False, False, False, False, False],
            "exit_short_signal": [False, False, False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=4, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Bridge trade expired", decision.rejection_reason)

    def test_bridge_trade_weak_impulse_rejection(self):
        cfg = IndicatorConfig(allow_bridge_trades=True, min_bridge_candle_atr=1.5)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.09, 1.09, 1.10],
            "high": [1.11, 1.11, 1.105],
            "low": [1.08, 1.08, 1.098],
            "close": [1.09, 1.095, 1.102],
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 0.7],     # 0.7 < min 1.5!
            "baseline_atr_dist": [0.0, 0.0, 0.2],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [1, 0, 0],
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertFalse(decision.is_valid)
        self.assertIn("Bridge candle impulse too weak", decision.rejection_reason)

    def test_bridge_trade_valid(self):
        cfg = IndicatorConfig(allow_bridge_trades=True, max_bridge_bars=5, min_bridge_candle_atr=0.8)
        algo = NNFXAlgorithm(cfg)
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(3)]
        df = pd.DataFrame({
            "open": [1.09, 1.09, 1.098],
            "high": [1.11, 1.11, 1.115],
            "low": [1.08, 1.08, 1.095],
            "close": [1.09, 1.095, 1.108],             # Crosses baseline (1.10)
            "baseline": [1.10, 1.10, 1.10],
            "baseline_dir": [1, 1, 1],
            "baseline_slope": [0.0, 0.0, 0.001],
            "atr": [0.01, 0.01, 0.01],
            "candle_valid": [True, True, True],
            "candle_atr_ratio": [1.0, 1.0, 1.3],       # 1.3 >= 0.8
            "baseline_atr_dist": [0.0, 0.0, 0.8],
            "ssl_dir": [1, 1, 1],
            "ssl_flip": [1, 0, 0],                     # 2 bars elapsed <= 5
            "c2_dir": [1, 1, 1],
            "wae_bull_vol": [True, True, True],
            "wae_bear_vol": [False, False, False],
            "exit_long_signal": [False, False, False],
            "exit_short_signal": [False, False, False],
        }, index=pd.DatetimeIndex(dates))

        decision = algo.evaluate_bar(df, current_idx=2, active_position_dir=0)
        self.assertTrue(decision.is_valid)
        self.assertEqual(decision.signal, SignalType.BUY)


class TestExecutionEngine(unittest.TestCase):
    """Tests two-tranche execution, breakeven stop moves, and runner trailing."""

    def test_tp1_hit_moves_runner_to_breakeven(self):
        tracker = PortfolioTracker(initial_capital=100_000.0)
        atr = 0.0050

        # Open Long trade at 1.1000
        trade = tracker.open_trade(
            direction=1,
            market_open_price=1.1000,
            current_atr=atr,
            timestamp=datetime(2025, 1, 1, 10, 0),
            bar_idx=1,
        )

        entry_px = trade.entry_price
        initial_sl = trade.current_sl
        tp1_px = trade.tp1_price

        self.assertTrue(trade.t1_open)
        self.assertTrue(trade.t2_open)
        self.assertFalse(trade.tp1_hit)

        # Bar 2 hits TP1: High reaches tp1_px, Low stays above SL
        tracker.process_bar(
            bar_idx=2,
            timestamp=datetime(2025, 1, 1, 11, 0),
            open_price=entry_px,
            high_price=tp1_px + 0.0010,
            low_price=entry_px - 0.0010,
            close_price=tp1_px,
            current_atr=atr,
            exit_indicator_signal=False,
        )

        # Tranche 1 should be closed with profit
        self.assertFalse(trade.t1_open)
        self.assertTrue(trade.tp1_hit)
        # Runner stop must now be at breakeven (entry_price)
        self.assertAlmostEqual(trade.current_sl, entry_px)

        # Bar 3 pulls back to entry_price (breakeven)
        tracker.process_bar(
            bar_idx=3,
            timestamp=datetime(2025, 1, 1, 12, 0),
            open_price=tp1_px,
            high_price=tp1_px,
            low_price=entry_px - 0.0005,
            close_price=entry_px,
            current_atr=atr,
            exit_indicator_signal=False,
        )

        # Both tranches now closed, trade completed
        self.assertIsNone(tracker.active_trade)
        self.assertEqual(len(tracker.completed_trades), 1)
        completed = tracker.completed_trades[0]
        # Net profit should be strictly positive because TP1 was banked and runner was breakeven
        self.assertGreater(completed.total_pnl_dollars, 0.0)


class TestSystemIntegration(unittest.TestCase):
    """Full end-to-end integration test of the Backtester."""

    def test_backtest_runs_cleanly(self):
        df = generate_synthetic_forex_data(n_bars=1500, seed=42)
        backtester = Backtester()
        metrics = backtester.run(df, verbose=False)

        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.total_trades, 0)
        self.assertGreaterEqual(metrics.win_rate_pct, 0.0)
        self.assertLessEqual(metrics.win_rate_pct, 100.0)


class TestStrategyOptimizer(unittest.TestCase):
    """Tests grid search optimizer and out-of-sample split logic."""

    def test_optimizer_runs(self):
        df = generate_synthetic_forex_data(n_bars=1500, seed=42)
        base_cfg = BacktestConfig()
        optimizer = StrategyOptimizer(base_cfg)

        small_grid = {
            "sl_atr_multiplier": [1.4, 1.6],
            "tp1_atr_multiplier": [1.0],
        }

        results = optimizer.optimize(df, param_grid=small_grid, train_ratio=0.70, top_n=2, verbose=False)
        self.assertGreater(len(results), 0)
        self.assertIn("sl_atr_multiplier", results[0].params)


if __name__ == "__main__":
    unittest.main()
