"""Data loader and validator for OHLCV financial time series data."""

from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np


class DataValidationError(Exception):
    """Raised when time series data fails financial validation checks."""
    pass


class DataLoader:
    """Loads, sanitizes, and validates historical OHLCV market data."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close"]

    @classmethod
    def load(
        cls,
        source: Union[str, Path, pd.DataFrame],
        datetime_col: str = "timestamp",
        date_format: str = None,
    ) -> pd.DataFrame:
        """
        Loads OHLCV data from a CSV file path or an existing DataFrame.

        Parameters
        ----------
        source : Union[str, Path, pd.DataFrame]
            Path to the CSV file or a pandas DataFrame.
        datetime_col : str
            Name of the datetime column if reading from CSV or DataFrame without DatetimeIndex.
        date_format : str, optional
            Explicit strftime format for parsing timestamps faster and reliably.

        Returns
        -------
        pd.DataFrame
            Sanitized, validated OHLCV DataFrame with DatetimeIndex and normalized lowercase column names.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found at: {path.resolve()}")
            df = pd.read_csv(path)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}. Must be str, Path, or pd.DataFrame.")

        return cls._sanitize_and_validate(df, datetime_col=datetime_col, date_format=date_format)

    @classmethod
    def _sanitize_and_validate(
        cls,
        df: pd.DataFrame,
        datetime_col: str = "timestamp",
        date_format: str = None,
    ) -> pd.DataFrame:
        """Sanitizes columns, parses timestamps, and executes data validation rules."""
        if df.empty:
            raise DataValidationError("Supplied DataFrame is empty.")

        # Normalize column names to lowercase and strip whitespace
        df.columns = [str(c).strip().lower() for c in df.columns]
        target_dt_col = datetime_col.strip().lower()

        # Handle Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            if target_dt_col in df.columns:
                df[target_dt_col] = pd.to_datetime(df[target_dt_col], format=date_format, errors="coerce")
                if df[target_dt_col].isna().any():
                    raise DataValidationError(f"Column '{target_dt_col}' contains unparseable datetime values.")
                df.set_index(target_dt_col, inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
                if df["date"].isna().any():
                    raise DataValidationError("Column 'date' contains unparseable datetime values.")
                df.set_index("date", inplace=True)
            elif "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], format=date_format, errors="coerce")
                if df["time"].isna().any():
                    raise DataValidationError("Column 'time' contains unparseable datetime values.")
                df.set_index("time", inplace=True)
            else:
                # Attempt to convert current index if possible
                try:
                    df.index = pd.to_datetime(df.index, format=date_format)
                except Exception as err:
                    raise DataValidationError(
                        f"No valid datetime column found and index could not be parsed as DatetimeIndex: {err}"
                    )

        # Check required OHLC columns
        missing_cols = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required OHLC columns: {missing_cols}")

        # Add dummy volume if not provided
        if "volume" not in df.columns:
            df["volume"] = 1.0

        # Enforce numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values in OHLC
        if df[cls.REQUIRED_COLUMNS].isna().any().any():
            nan_counts = df[cls.REQUIRED_COLUMNS].isna().sum().to_dict()
            raise DataValidationError(f"NaN values detected in OHLC data: {nan_counts}")

        # Check timestamp ordering
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
            if not df.index.is_monotonic_increasing:
                raise DataValidationError("Timestamps are not strictly ascending even after sorting.")

        # Check for duplicate timestamps
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        # OHLC logical consistency validation
        # High must be >= Open, High >= Close, Low <= Open, Low <= Close
        invalid_high = (df["high"] < df["open"] - 1e-9) | (df["high"] < df["close"] - 1e-9)
        if invalid_high.any():
            violating_count = int(invalid_high.sum())
            raise DataValidationError(
                f"Data inconsistency: {violating_count} bars have High price strictly lower than Open or Close."
            )

        invalid_low = (df["low"] > df["open"] + 1e-9) | (df["low"] > df["close"] + 1e-9)
        if invalid_low.any():
            violating_count = int(invalid_low.sum())
            raise DataValidationError(
                f"Data inconsistency: {violating_count} bars have Low price strictly higher than Open or Close."
            )

        # Non-negative prices check
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise DataValidationError("Non-positive prices detected in OHLC data.")

        return df[["open", "high", "low", "close", "volume"]].copy()
