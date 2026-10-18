"""Configuration dataclasses for the NNFX Algorithmic Trading Framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    """Money Management and Risk Engine configuration."""
    risk_pct: float = 0.015          # 1.5% account equity risk per trade
    atr_period: int = 14             # Standard ATR period for sizing and stops
    sl_atr_multiplier: float = 1.5   # Stop loss placed at 1.5 * ATR
    tp1_atr_multiplier: float = 1.0  # TP1 placed at 1.0 * ATR
    tranche_1_pct: float = 0.50      # 50% allocated to Scale 1 (TP1)
    tranche_2_pct: float = 0.50      # 50% allocated to Scale 2 (Runner)
    move_to_be_on_tp1: bool = True   # Automatically move runner SL to breakeven on TP1 hit
    profit_ratchet_enabled: bool = False # Enable dynamic multi-tier profit ratcheting
    ratchet_tiers: tuple = ((2.0, 0.5), (4.0, 2.0), (8.0, 5.0)) # (profit_atr_threshold, locked_profit_atr)
    runner_buffer_atr: float = 0.5       # Cushion ATR distance below/above entry when moving to breakeven
    runner_trailing_atr_mult: float = 1.5 # ATR trailing multiplier for runner once in profit
    max_open_trades: int = 1         # Single-pair default max active position
    account_currency: str = "USD"
    min_position_size: float = 0.01  # Minimum lot or fractional unit size
    loss_cooldown_bars: int = 0      # Bars to pause entries after an initial stop loss hit


@dataclass(frozen=True)
class IndicatorConfig:
    """Technical Indicator Engine configuration."""
    # Baseline
    baseline_type: str = "EMA"       # "EMA", "SMA", "KIJUN"
    baseline_period: int = 200
    baseline_max_atr_dist: float = 1.5  # Max distance from baseline to allow entry (NNFX Rule)

    # Confirmation 1 (C1) - SSL Channel
    ssl_period: int = 10

    # Confirmation 2 (C2) - Secondary Confirmation Filter
    c2_enabled: bool = True
    c2_type: str = "VORTEX"          # "VORTEX", "AROON"
    vortex_period: int = 14
    min_c2_spread: float = 0.0       # Minimum spread |VI+ - VI-| required for entry (e.g. 0.15 for high precision)

    # Trend Strength Filter (Wilder's ADX)
    adx_enabled: bool = False
    adx_period: int = 14
    min_adx: float = 18.0            # Rejects entries in flat/choppy consolidation (< 18.0)

    # Volume / Volatility Filter - Waddah Attar Explosion (WAE) style
    wae_fast_period: int = 20
    wae_slow_period: int = 40
    wae_bb_period: int = 20
    wae_bb_mult: float = 2.0
    wae_dead_zone_atr_mult: float = 0.75  # Dead zone threshold = mult * ATR

    # Exit Indicator - DeMarker / Fast Stochastic
    exit_oscillator_period: int = 14
    exit_ob_level: float = 0.70      # Overbought / exit threshold for Long
    exit_os_level: float = 0.30      # Oversold / exit threshold for Short

    # Precision Filters (NNFX Advanced Rules)
    max_candle_atr_ratio: float = 2.0      # Reject entries on giant exhaustion candles (> 2.0 * ATR)
    baseline_slope_filter: bool = False    # Baseline slope filter (optional, default False per NNFX)
    baseline_slope_lookback: int = 5       # Lookback bars to determine baseline direction

    # Bridge Trade Controls (NNFX Bridge Setup)
    allow_bridge_trades: bool = True       # Whether to allow baseline cross entries when C1 already active
    max_bridge_bars: int = 5               # Max bars elapsed since C1 flip for valid bridge entry (default: 5)
    min_bridge_candle_atr: float = 0.8     # Minimum candle impulse range (High - Low) / ATR for bridge entry

    # Trailing Runner Stop
    runner_trailing_atr_mult: float = 1.5  # ATR trailing distance for runner once in profit



@dataclass(frozen=True)
class ExecutionConfig:
    """Execution model configuration."""
    spread_pips: float = 0.00015     # 1.5 pips fixed spread simulation
    slippage_pips: float = 0.00005   # 0.5 pips slippage simulation
    commission_per_unit: float = 0.0 # Per-unit commission (if any)


@dataclass(frozen=True)
class BacktestConfig:
    """Backtester runtime configuration."""
    initial_capital: float = 100_000.0
    risk: RiskConfig = field(default_factory=RiskConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    annualization_factor: int = 252  # 252 for daily, 252*24 for hourly


@dataclass(frozen=True)
class ExchangeConfig:
    """CCXT Exchange and Live/Paper Trading configuration."""
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    paper_trading: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    password: Optional[str] = None   # Required for KuCoin/OKX
    rate_limit_delay_sec: float = 0.2
    data_cache_dir: str = "data"
    poll_interval_sec: float = 10.0

