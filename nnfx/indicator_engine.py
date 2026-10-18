"""Technical indicator engine implementing NNFX algorithms using vectorized Pandas & NumPy."""

import numpy as np
import pandas as pd
from typing import Optional
from nnfx.config import IndicatorConfig


class IndicatorEngine:
    """
    Computes all standard NNFX technical indicators in a fully vectorized pipeline:
    1. Average True Range (ATR)
    2. Baseline (EMA 200 / Kijun-sen) & Baseline Distance
    3. Confirmation Indicator (SSL Channel)
    4. Volume / Volatility Filter (Waddah Attar Explosion)
    5. Exit Indicator (DeMarker Oscillator)
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all indicators and appends them as columns to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with indicator columns.
        """
        data = df.copy()

        # 1. ATR (Wilder's Smoothing)
        data = self.compute_atr(data, period=14)

        # 2. Baseline & Slope
        data = self.compute_baseline(data)

        # 3. Confirmation Indicator 1 (C1) - SSL Channel
        data = self.compute_ssl_channel(data, period=self.config.ssl_period)

        # 4. Confirmation Indicator 2 (C2) - Vortex Indicator
        if self.config.c2_enabled:
            data = self.compute_vortex(data, period=self.config.vortex_period)
        else:
            data["c2_dir"] = data["ssl_dir"]
            data["c2_flip"] = data["ssl_flip"]

        # 5. Trend Strength Filter (Wilder's ADX)
        if self.config.adx_enabled:
            data = self.compute_adx(data, period=self.config.adx_period)
        else:
            data["adx"] = 25.0
            data["adx_valid"] = True

        # 6. Volume / Volatility Filter (Waddah Attar Explosion)
        data = self.compute_waddah_attar_explosion(data)

        # 7. Exit Indicator (DeMarker Oscillator)
        data = self.compute_exit_oscillator(data, period=self.config.exit_oscillator_period)

        return data

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculates True Range (TR), Average True Range (ATR), and candle size ratio."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["tr"] = tr

        # Wilder's Smoothing: ewm with alpha = 1 / period
        df["atr"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Candle range vs ATR ratio (to filter out climax giant candles)
        df["candle_range"] = high - low
        df["candle_atr_ratio"] = df["candle_range"] / (df["atr"] + 1e-12)
        df["candle_valid"] = df["candle_atr_ratio"] <= self.config.max_candle_atr_ratio
        return df

    def compute_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes baseline (EMA 200), direction (+1/-1), ATR distance, and slope."""
        close = df["close"]
        period = self.config.baseline_period

        if self.config.baseline_type.upper() == "EMA":
            baseline = close.ewm(span=period, adjust=False).mean()
        elif self.config.baseline_type.upper() == "SMA":
            baseline = close.rolling(window=period, min_periods=period).mean()
        elif self.config.baseline_type.upper() == "KIJUN":
            high_roll = df["high"].rolling(window=period, min_periods=period).max()
            low_roll = df["low"].rolling(window=period, min_periods=period).min()
            baseline = (high_roll + low_roll) / 2.0
        else:
            baseline = close.ewm(span=period, adjust=False).mean()

        df["baseline"] = baseline

        # Baseline Direction: +1 if close > baseline, -1 if close < baseline, 0 if equal
        df["baseline_dir"] = np.where(df["close"] > df["baseline"], 1, np.where(df["close"] < df["baseline"], -1, 0))

        # Baseline Distance in ATR units
        df["baseline_atr_dist"] = (df["close"] - df["baseline"]).abs() / (df["atr"] + 1e-12)

        # Baseline Qualification: Distance <= max allowed ATR distance
        df["baseline_distance_valid"] = df["baseline_atr_dist"] <= self.config.baseline_max_atr_dist

        # Baseline Slope over lookback bars
        lookback = self.config.baseline_slope_lookback
        df["baseline_slope"] = baseline - baseline.shift(lookback)

        return df

    @staticmethod
    def compute_ssl_channel(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Computes the SSL Channel confirmation indicator.
        SSL Channel uses rolling SMAs of High and Low prices.
        When Close crosses above High SMA, the state turns Bullish (+1).
        When Close crosses below Low SMA, the state turns Bearish (-1).
        """
        high_sma = df["high"].rolling(window=period, min_periods=period).mean()
        low_sma = df["low"].rolling(window=period, min_periods=period).mean()

        close = df["close"].values
        high_sma_vals = high_sma.values
        low_sma_vals = low_sma.values
        n = len(df)

        ssl_dir = np.zeros(n, dtype=int)
        current_dir = 0

        # Vectorized state machine execution
        for i in range(n):
            if np.isnan(high_sma_vals[i]) or np.isnan(low_sma_vals[i]):
                ssl_dir[i] = 0
                continue
            if close[i] > high_sma_vals[i]:
                current_dir = 1
            elif close[i] < low_sma_vals[i]:
                current_dir = -1
            ssl_dir[i] = current_dir

        df["ssl_dir"] = ssl_dir
        # Flip signal: +1 on bullish flip, -1 on bearish flip, 0 otherwise
        prev_ssl_dir = pd.Series(ssl_dir, index=df.index).shift(1).fillna(0).astype(int)
        df["ssl_flip"] = np.where(
            (df["ssl_dir"] == 1) & (prev_ssl_dir != 1),
            1,
            np.where((df["ssl_dir"] == -1) & (prev_ssl_dir != -1), -1, 0)
        )
        return df

    def compute_waddah_attar_explosion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Waddah Attar Explosion (WAE) Volume/Volatility indicator.
        Trend Power = Difference between Fast and Slow MACD multiplied by sensitivity.
        Explosion Line = Bollinger Bands bandwidth (Upper BB - Lower BB).
        Dead Zone = Multiplier * ATR.
        """
        close = df["close"]
        fast_p = self.config.wae_fast_period
        slow_p = self.config.wae_slow_period
        bb_p = self.config.wae_bb_period
        bb_mult = self.config.wae_bb_mult

        # MACD trend difference
        fast_ema = close.ewm(span=fast_p, adjust=False).mean()
        slow_ema = close.ewm(span=slow_p, adjust=False).mean()
        macd = fast_ema - slow_ema

        # Trend difference (speed/acceleration of MACD)
        macd_diff = macd - macd.shift(1)
        # Sensitivity scaling
        trend_power = macd_diff * 150.0

        # Bollinger Band width for explosion line
        bb_mid = close.rolling(window=bb_p, min_periods=bb_p).mean()
        bb_std = close.rolling(window=bb_p, min_periods=bb_p).std()
        bb_up = bb_mid + (bb_std * bb_mult)
        bb_down = bb_mid - (bb_std * bb_mult)
        explosion_line = bb_up - bb_down

        # Dead Zone threshold based on ATR
        dead_zone = df["atr"] * self.config.wae_dead_zone_atr_mult

        df["wae_trend_power"] = trend_power
        df["wae_explosion"] = explosion_line
        df["wae_dead_zone"] = dead_zone

        # Bullish volume: trend_power > 0 and trend_power > explosion_line and trend_power > dead_zone
        df["wae_bull_vol"] = (
            (trend_power > 0) &
            (trend_power > explosion_line) &
            (trend_power > dead_zone)
        )

        # Bearish volume: trend_power < 0 and abs(trend_power) > explosion_line and abs(trend_power) > dead_zone
        df["wae_bear_vol"] = (
            (trend_power < 0) &
            (trend_power.abs() > explosion_line) &
            (trend_power.abs() > dead_zone)
        )

        return df

    def compute_exit_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Computes the DeMarker indicator as an exit oscillator.
        DeMax = High - PrevHigh if High > PrevHigh else 0
        DeMin = PrevLow - Low if Low < PrevLow else 0
        DeMarker = SMA(DeMax, period) / (SMA(DeMax, period) + SMA(DeMin, period))
        """
        high = df["high"]
        low = df["low"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)

        demax = np.where(high > prev_high, high - prev_high, 0.0)
        demin = np.where(low < prev_low, prev_low - low, 0.0)

        demax_series = pd.Series(demax, index=df.index)
        demin_series = pd.Series(demin, index=df.index)

        demax_sma = demax_series.rolling(window=period, min_periods=period).mean()
        demin_sma = demin_series.rolling(window=period, min_periods=period).mean()

        denominator = demax_sma + demin_sma
        demarker = np.where(denominator > 0, demax_sma / denominator, 0.5)

        df["exit_oscillator"] = pd.Series(demarker, index=df.index).fillna(0.5)

        # Exit Signals:
        # For Long: Oscillator crossing down below Overbought level or flipping downward
        prev_osc = df["exit_oscillator"].shift(1)
        df["exit_long_signal"] = (
            (prev_osc >= self.config.exit_ob_level) &
            (df["exit_oscillator"] < self.config.exit_ob_level)
        ) | (df["baseline_dir"] == -1)  # Or if price closes back below baseline

        # For Short: Oscillator crossing up above Oversold level or flipping upward
        df["exit_short_signal"] = (
            (prev_osc <= self.config.exit_os_level) &
            (df["exit_oscillator"] > self.config.exit_os_level)
        ) | (df["baseline_dir"] == 1)   # Or if price closes back above baseline

        return df

    @staticmethod
    def compute_vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Computes the Vortex Indicator (VI+ and VI-) as Confirmation Indicator 2 (C2).
        VM+ = |Current High - Prior Low|
        VM- = |Current Low - Prior High|
        VI+ = Rolling Sum(VM+, period) / Rolling Sum(TR, period)
        VI- = Rolling Sum(VM-, period) / Rolling Sum(TR, period)
        Bullish when VI+ > VI-, Bearish when VI- > VI+.
        """
        high = df["high"]
        low = df["low"]
        tr = df["tr"]

        prev_low = low.shift(1)
        prev_high = high.shift(1)

        vm_plus = (high - prev_low).abs()
        vm_minus = (low - prev_high).abs()

        sum_vm_plus = vm_plus.rolling(window=period, min_periods=period).sum()
        sum_vm_minus = vm_minus.rolling(window=period, min_periods=period).sum()
        sum_tr = tr.rolling(window=period, min_periods=period).sum()

        vi_plus = sum_vm_plus / (sum_tr + 1e-12)
        vi_minus = sum_vm_minus / (sum_tr + 1e-12)

        df["vortex_plus"] = vi_plus
        df["vortex_minus"] = vi_minus

        # C2 Direction: +1 Bullish, -1 Bearish
        c2_dir = np.where(vi_plus > vi_minus, 1, np.where(vi_minus > vi_plus, -1, 0))
        df["c2_dir"] = c2_dir

        prev_c2_dir = pd.Series(c2_dir, index=df.index).shift(1).fillna(0).astype(int)
        df["c2_flip"] = np.where(
            (df["c2_dir"] == 1) & (prev_c2_dir != 1),
            1,
            np.where((df["c2_dir"] == -1) & (prev_c2_dir != -1), -1, 0)
        )

        return df

    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Computes Wilder's Average Directional Index (ADX), +DI, and -DI.
        Used as a regime filter to reject entries during low-momentum whipsaw/consolidation.
        """
        high = df["high"]
        low = df["low"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        pos_dm_s = pd.Series(pos_dm, index=df.index)
        neg_dm_s = pd.Series(neg_dm, index=df.index)

        alpha = 1.0 / period
        tr_smooth = df["tr"].ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        pos_di = 100.0 * (pos_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (tr_smooth + 1e-12))
        neg_di = 100.0 * (neg_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (tr_smooth + 1e-12))

        di_sum = pos_di + neg_di
        dx = 100.0 * (pos_di - neg_di).abs() / (di_sum + 1e-12)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        df["plus_di"] = pos_di
        df["minus_di"] = neg_di
        df["adx"] = adx
        df["adx_valid"] = adx >= self.config.min_adx

        return df

