"""CCXT data feed engine for downloading, paginating, caching, and validating real market data."""

import os
import time
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import ccxt

from nnfx.data_loader import DataLoader, DataValidationError
from nnfx.config import ExchangeConfig


class CCXTDataFeed:
    """
    Ingests live and historical market data across 100+ exchanges using CCXT.
    Handles pagination, rate limiting, local CSV caching, and schema validation.
    """

    def __init__(self, config: Optional[ExchangeConfig] = None):
        self.config = config or ExchangeConfig()
        self.exchange_id = self.config.exchange_id.lower()
        self.exchange = self._init_exchange(self.exchange_id)

    def _init_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Instantiates and configures a CCXT exchange instance."""
        if not hasattr(ccxt, exchange_id):
            available = [e for e in ccxt.exchanges if not e.startswith("_")]
            raise ValueError(
                f"Exchange '{exchange_id}' is not supported by CCXT. Examples of valid exchanges: "
                f"{available[:10]}"
            )

        exchange_class = getattr(ccxt, exchange_id)
        params = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        # CCXTDataFeed fetches public market data (OHLCV), which does not require credentials
        return exchange_class(params)

    def fetch_historical_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 2000,
        use_cache: bool = True,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data with pagination and local caching.

        Parameters
        ----------
        symbol : str, optional
            Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT').
        timeframe : str, optional
            Candle timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d').
        limit : int
            Total target number of bars to retrieve.
        use_cache : bool
            Whether to load from or store to local CSV cache.
        force_download : bool
            If True, bypasses local cache and downloads fresh data from exchange.

        Returns
        -------
        pd.DataFrame
            Sanitized, validated OHLCV DataFrame with DatetimeIndex.
        """
        target_symbol = symbol or self.config.symbol
        target_tf = timeframe or self.config.timeframe

        clean_symbol = target_symbol.replace("/", "_").replace(":", "_")
        cache_dir = Path(self.config.data_cache_dir)
        cache_file = cache_dir / f"{self.exchange_id}_{clean_symbol}_{target_tf}.csv"

        # 1. Check local cache first
        if use_cache and not force_download and cache_file.exists():
            try:
                cached_df = DataLoader.load(cache_file)
                if len(cached_df) >= limit:
                    print(f"[*] Loaded {len(cached_df):,} bars for {target_symbol} from local cache: {cache_file}")
                    return cached_df.tail(limit).copy()
            except Exception as err:
                print(f"[!] Warning: Failed reading cache {cache_file}: {err}. Re-downloading...")

        # 2. Download from exchange with pagination
        print(f"[*] Fetching {limit:,} bars of real {target_symbol} ({target_tf}) from {self.exchange_id.upper()}...")
        all_candles = []
        batch_limit = 1000

        # Calculate step in milliseconds for the given timeframe
        tf_ms = self._timeframe_to_ms(target_tf)
        now_ms = int(time.time() * 1000)
        # Approximate start time to cover `limit` bars
        target_since_ms = now_ms - (limit * tf_ms)

        current_since = target_since_ms
        fetched_count = 0

        while fetched_count < limit:
            chunk_limit = min(batch_limit, limit - fetched_count + 100)
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=target_symbol,
                    timeframe=target_tf,
                    since=current_since,
                    limit=chunk_limit,
                )
            except Exception as e:
                # If geoblocked or rate-limited, attempt fallback to kraken if default was binance
                if self.exchange_id == "binance" and ("restricted" in str(e).lower() or "blocked" in str(e).lower()):
                    print(f"[!] Binance access restricted ({e}). Switching to Kraken fallback...")
                    self.exchange_id = "kraken"
                    self.exchange = self._init_exchange("kraken")
                    continue
                raise e

            if not candles:
                break

            all_candles.extend(candles)
            fetched_count = len(all_candles)

            # Advance current_since beyond last received candle timestamp
            last_ts = candles[-1][0]
            if last_ts <= current_since:
                current_since += tf_ms
            else:
                current_since = last_ts + 1

            if len(candles) < chunk_limit or current_since >= now_ms:
                break

            time.sleep(self.config.rate_limit_delay_sec)

        if not all_candles:
            raise RuntimeError(f"No OHLCV data could be retrieved for {target_symbol} from {self.exchange_id}.")

        # 3. Format into DataFrame
        df_raw = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        # Drop duplicates by timestamp and sort
        df_raw.drop_duplicates(subset=["timestamp"], inplace=True)
        df_raw.sort_values(by="timestamp", inplace=True)

        # Convert epoch ms to DatetimeIndex
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
        df_raw.set_index("timestamp", inplace=True)

        # 4. Validate through DataLoader
        validated_df = DataLoader.load(df_raw)

        # Trim to requested limit if exceeded
        if len(validated_df) > limit:
            validated_df = validated_df.tail(limit).copy()

        # 5. Save to local cache
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            validated_df.to_csv(cache_file)
            print(f"[+] Cached {len(validated_df):,} bars to: {cache_file}")

        return validated_df

    def fetch_latest_ticker(self, symbol: Optional[str] = None) -> dict:
        """Fetches current live bid, ask, and last price for real-time spread simulation."""
        target_symbol = symbol or self.config.symbol
        return self.exchange.fetch_ticker(target_symbol)

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """Converts human timeframe string ('1m', '1h', '1d') into milliseconds."""
        unit = timeframe[-1].lower()
        amount = int(timeframe[:-1])
        multipliers = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }
        if unit not in multipliers:
            return 60 * 60 * 1000
        return amount * multipliers[unit]
