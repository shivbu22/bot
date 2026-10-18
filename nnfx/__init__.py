"""No Nonsense Forex (NNFX) Algorithmic Trading Bot & Backtester Package."""

from nnfx.config import RiskConfig, IndicatorConfig, ExecutionConfig, BacktestConfig, ExchangeConfig
from nnfx.data_loader import DataLoader, DataValidationError
from nnfx.indicator_engine import IndicatorEngine
from nnfx.risk_manager import RiskManager, PositionSpecs
from nnfx.algorithm import NNFXAlgorithm, SignalDecision, SignalType
from nnfx.execution import PortfolioTracker, ActiveTrade, CompletedTrade, TrancheRecord, ExitReason
from nnfx.metrics import MetricsCalculator, PerformanceMetrics
from nnfx.synthetic_data import generate_synthetic_forex_data
from nnfx.backtester import Backtester
from nnfx.ccxt_feed import CCXTDataFeed
from nnfx.ccxt_live_trader import CCXTLiveTrader
from nnfx.optimizer import StrategyOptimizer, OptimizationResult

__all__ = [
    "RiskConfig",
    "IndicatorConfig",
    "ExecutionConfig",
    "BacktestConfig",
    "ExchangeConfig",
    "DataLoader",
    "DataValidationError",
    "IndicatorEngine",
    "RiskManager",
    "PositionSpecs",
    "NNFXAlgorithm",
    "SignalDecision",
    "SignalType",
    "PortfolioTracker",
    "ActiveTrade",
    "CompletedTrade",
    "TrancheRecord",
    "ExitReason",
    "MetricsCalculator",
    "PerformanceMetrics",
    "generate_synthetic_forex_data",
    "Backtester",
    "CCXTDataFeed",
    "CCXTLiveTrader",
    "StrategyOptimizer",
    "OptimizationResult",
]

