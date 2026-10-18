"""CCXT Live and Paper Trading Engine for real-time NNFX execution."""

import time
import signal
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import ccxt
from nnfx.config import ExchangeConfig, BacktestConfig
from nnfx.ccxt_feed import CCXTDataFeed
from nnfx.indicator_engine import IndicatorEngine
from nnfx.algorithm import NNFXAlgorithm, SignalType
from nnfx.risk_manager import RiskManager
from nnfx.execution import PortfolioTracker


class CCXTLiveTrader:
    """
    Live and Paper Trading Execution Engine:
    - Streams live OHLCV data from CCXT exchanges.
    - Monitors bar close events in real-time.
    - Evaluates NNFX multi-layer indicator conditions.
    - Manages two-tranche execution (TP1 and Runner) with Breakeven and Trailing Stop updates.
    """

    def __init__(
        self,
        exchange_config: Optional[ExchangeConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        self.exchange_cfg = exchange_config or ExchangeConfig()
        self.bt_cfg = backtest_config or BacktestConfig()

        self.feed = CCXTDataFeed(self.exchange_cfg)
        self.indicator_engine = IndicatorEngine(self.bt_cfg.indicators)
        self.algorithm = NNFXAlgorithm(self.bt_cfg.indicators, self.bt_cfg.risk)
        self.portfolio = PortfolioTracker(
            initial_capital=self.bt_cfg.initial_capital,
            risk_config=self.bt_cfg.risk,
            exec_config=self.bt_cfg.execution,
        )

        self.trade_exchange = self._init_trade_exchange()
        self.historical_df: Optional[pd.DataFrame] = None
        self.last_processed_timestamp: Optional[Any] = None
        self.is_running = False

    def _init_trade_exchange(self) -> Optional[ccxt.Exchange]:
        """Instantiates authenticated exchange for live order placement and wallet balance management."""
        if self.exchange_cfg.paper_trading:
            return None
        exchange_id = self.exchange_cfg.exchange_id.lower()
        if not hasattr(ccxt, exchange_id):
            return None
        exchange_class = getattr(ccxt, exchange_id)
        params = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if self.exchange_cfg.api_key:
            params["apiKey"] = self.exchange_cfg.api_key
        if self.exchange_cfg.api_secret:
            params["secret"] = self.exchange_cfg.api_secret
        if self.exchange_cfg.password:
            params["password"] = self.exchange_cfg.password
        return exchange_class(params)

    def initialize(self, warmup_bars: int = 300):
        """Initializes the historical candle buffer for indicator warmup and syncs live wallet balance."""
        print(f"[*] Initializing CCXT Live Trader on {self.exchange_cfg.exchange_id.upper()}...")
        print(f"[*] Symbol: {self.exchange_cfg.symbol} | Timeframe: {self.exchange_cfg.timeframe}")
        mode_str = "PAPER TRADING (Simulated Fills)" if self.exchange_cfg.paper_trading else "LIVE TRADING (Real Funds)"
        print(f"[*] Execution Mode: {mode_str}")

        # In real trading mode, authenticate and sync live wallet balance
        if not self.exchange_cfg.paper_trading and self.trade_exchange:
            print("[*] Validating Binance API credentials and fetching wallet balance...")
            try:
                bal = self.trade_exchange.fetch_balance()
                usdt_free = float(bal.get("USDT", {}).get("free", 0.0))
                usdt_total = float(bal.get("USDT", {}).get("total", 0.0))
                print(f"[+] Binance Authentication SUCCESSFUL!")
                print(f"[+] Available Free USDT: ${usdt_free:,.2f} | Total Account Value: ${usdt_total:,.2f}")
                if usdt_free > 0:
                    self.portfolio.current_equity = usdt_free
                    print(f"[+] Risk position sizing synchronized to actual wallet equity (${self.portfolio.current_equity:,.2f}).")
                else:
                    print("[!] Notice: Available USDT is $0.00. Real orders cannot execute without funds.")
            except Exception as ex:
                print(f"[!] Binance Authentication / Balance Notice: {ex}")
                print("[!] Action Required for Live Real Orders: Verify API Key & Secret permissions in Binance API Management.")

        self.historical_df = self.feed.fetch_historical_ohlcv(
            symbol=self.exchange_cfg.symbol,
            timeframe=self.exchange_cfg.timeframe,
            limit=warmup_bars,
            use_cache=True,
        )
        self.last_processed_timestamp = self.historical_df.index[-2]
        print(f"[+] Initialized buffer with {len(self.historical_df)} bars up to {self.historical_df.index[-1]}.")

    def poll_and_evaluate_step(self) -> Dict[str, Any]:
        """
        Single polling evaluation step:
        1. Fetches latest completed candle.
        2. Detects if a new bar has finalized.
        3. Runs indicator calculations and NNFX algorithm evaluation.
        4. Updates two-tranche positions, breakeven stops, and exits.
        """
        if self.historical_df is None:
            self.initialize()

        # Fetch recent 5 bars to catch the latest closed candle
        recent_df = self.feed.fetch_historical_ohlcv(
            symbol=self.exchange_cfg.symbol,
            timeframe=self.exchange_cfg.timeframe,
            limit=5,
            use_cache=False,
            force_download=True,
        )

        # In CCXT, the very last candle (index -1) is typically the active forming candle.
        # The penultimate candle (index -2) is the latest closed candle.
        if len(recent_df) < 2:
            return {"status": "waiting_for_candles"}

        latest_closed_idx = -2
        latest_closed_bar = recent_df.iloc[latest_closed_idx]
        latest_closed_ts = recent_df.index[latest_closed_idx]

        active_forming_bar = recent_df.iloc[-1]
        current_market_price = float(active_forming_bar["open"])

        # Check if this bar has already been evaluated
        if self.last_processed_timestamp is not None and latest_closed_ts <= self.last_processed_timestamp:
            # Intrabar update of active position on currently forming bar
            if self.portfolio.active_trade is not None:
                self.portfolio.process_bar(
                    bar_idx=len(self.historical_df),
                    timestamp=recent_df.index[-1],
                    open_price=float(active_forming_bar["open"]),
                    high_price=float(active_forming_bar["high"]),
                    low_price=float(active_forming_bar["low"]),
                    close_price=float(active_forming_bar["close"]),
                    current_atr=float(self.historical_df["atr"].iloc[-1]) if "atr" in self.historical_df else 0.001,
                    exit_indicator_signal=False,
                )
                trade = self.portfolio.active_trade
                unrealized_pnl = trade.get_unrealized_pnl(current_market_price)
                tp1_status = "HIT (Scale 1 Banked)" if trade.tp1_hit else f"PENDING (Target: ${trade.specs.tp1_price:.2f})"
                print(
                    f"[LIVE POSITION] Trade #{trade.trade_id} {'LONG' if trade.direction == 1 else 'SHORT'} @ ${trade.entry_price:.2f} "
                    f"| Current: ${current_market_price:.2f} | Unr PnL: ${unrealized_pnl:+,.2f} "
                    f"| TP1: {tp1_status} | Current SL: ${trade.current_sl:.2f}",
                    flush=True
                )
            return {"status": "candle_still_forming", "timestamp": latest_closed_ts}

        # New completed candle detected!
        print("=" * 80)
        print(f"[+] NEW CANDLE CLOSED: {latest_closed_ts} | Symbol: {self.exchange_cfg.symbol} | Close: ${latest_closed_bar['close']:,.2f}")
        print("=" * 80)

        # Update historical DataFrame
        self.historical_df = pd.concat([self.historical_df, recent_df.iloc[[latest_closed_idx]]])
        self.historical_df = self.historical_df[~self.historical_df.index.duplicated(keep="last")].sort_index()

        # Recompute indicators on updated historical buffer
        enriched_df = self.indicator_engine.compute_all(self.historical_df)
        self.historical_df = enriched_df

        curr_bar_idx = len(enriched_df) - 1
        active_dir = self.portfolio.active_trade.direction if self.portfolio.active_trade else 0
        bars_since_loss = self.portfolio.get_bars_since_last_sl(curr_bar_idx)

        # Diagnostic readout of all 6 NNFX indicator layers
        row = enriched_df.iloc[curr_bar_idx]
        current_atr = float(row["atr"])
        c1_dir = "BULLISH (+1)" if row["ssl_dir"] == 1 else ("BEARISH (-1)" if row["ssl_dir"] == -1 else "NEUTRAL (0)")
        c2_spread = abs(row["vortex_plus"] - row["vortex_minus"])
        c2_dir = "BULLISH" if row["vortex_plus"] > row["vortex_minus"] else "BEARISH"
        wae_status = "BULL VOL" if row.get("wae_bull_vol", False) else ("BEAR VOL" if row.get("wae_bear_vol", False) else "NO EXPLOSION")
        adx_val = float(row.get("adx", 0.0))
        dist_atr = float(row["baseline_atr_dist"])

        print(f"[*] --- NNFX LIVE INDICATOR TELEMETRY ---")
        print(f"    1. Baseline (200 EMA):    ${row['baseline']:,.2f} (Dist: {dist_atr:.2f} ATR | Valid: {dist_atr <= self.bt_cfg.indicators.baseline_max_atr_dist})")
        print(f"    2. Confirmation 1 (SSL):  {c1_dir} (Flip: {row['ssl_flip']})")
        print(f"    3. Confirmation 2 (VTX):  {c2_dir} (Spread: {c2_spread:.3f} | Min Req: {self.bt_cfg.indicators.min_c2_spread:.3f})")
        print(f"    4. Trend Strength (ADX):  {adx_val:.1f} (Min Req: {self.bt_cfg.indicators.min_adx:.1f})")
        print(f"    5. Volume/Volatility (WAE): {wae_status} (Power: {row['wae_trend_power']:.1f} | Explosion: {row['wae_explosion']:.1f})")
        print(f"    6. Exit Oscillator:       {row['exit_oscillator']:.3f}")

        # Evaluate rules on closed bar
        decision = self.algorithm.evaluate_bar(
            df=enriched_df,
            current_idx=curr_bar_idx,
            active_position_dir=active_dir,
            bars_since_last_loss=bars_since_loss,
        )

        if decision.is_valid:
            side_str = "BUY (LONG)" if decision.direction == 1 else "SELL (SHORT)"
            print(f"[+] >>> NNFX SIGNAL CONFIRMED: {side_str} @ ${current_market_price:,.2f} <<<")
        else:
            print(f"[-] Signal Status: NEUTRAL / REJECTED ({decision.rejection_reason})")

        # If entry setup is confirmed and no position open, trigger entry on current open price
        if decision.is_valid and self.portfolio.active_trade is None:
            side_str = "BUY" if decision.direction == 1 else "SELL"
            if self.exchange_cfg.paper_trading:
                trade = self.portfolio.open_trade(
                    direction=decision.direction,
                    market_open_price=current_market_price,
                    current_atr=current_atr,
                    timestamp=recent_df.index[-1],
                    bar_idx=curr_bar_idx + 1,
                )
                print(f"[+] [PAPER FILL] Trade #{trade.trade_id} {side_str} @ ${trade.entry_price:,.2f}")
                print(f"    Initial SL: ${trade.specs.initial_sl:,.2f} | TP1: ${trade.specs.tp1_price:,.2f} | Units: {trade.specs.total_units:.4f}")
            else:
                print(f"[!] Processing REAL LIVE ORDER on {self.exchange_cfg.exchange_id.upper()}...")
                order_side = "buy" if decision.direction == 1 else "sell"
                
                # Refresh available wallet balance if available
                if self.trade_exchange:
                    try:
                        bal = self.trade_exchange.fetch_balance()
                        usdt_free = float(bal.get("USDT", {}).get("free", 0.0))
                        if usdt_free > 0:
                            self.portfolio.current_equity = usdt_free
                    except Exception:
                        pass

                specs = self.portfolio.risk_manager.calculate_position(
                    current_equity=self.portfolio.current_equity,
                    entry_price=current_market_price,
                    atr=current_atr,
                    direction=decision.direction,
                )
                
                # Minimum notional safety check (Binance min ~5-10 USDT)
                notional = specs.total_units * current_market_price
                if notional < 5.0:
                    print(f"[!] Order notional (${notional:.2f}) is below Binance minimum (5.00 USDT). Cannot execute order.")
                elif self.trade_exchange:
                    try:
                        formatted_amount = float(
                            self.trade_exchange.amount_to_precision(self.exchange_cfg.symbol, specs.total_units)
                        )
                        order = self.trade_exchange.create_market_order(
                            symbol=self.exchange_cfg.symbol,
                            side=order_side,
                            amount=formatted_amount,
                        )
                        print(f"[+] REAL LIVE ORDER PLACED! Order ID: {order.get('id')} | Amount: {formatted_amount} | Status: {order.get('status')}")
                        
                        # Open tracking trade in portfolio to manage SL/TP and runner exit
                        self.portfolio.open_trade(
                            direction=decision.direction,
                            market_open_price=current_market_price,
                            current_atr=current_atr,
                            timestamp=recent_df.index[-1],
                            bar_idx=curr_bar_idx + 1,
                        )
                    except Exception as ex:
                        print(f"[!] Error placing real exchange order: {ex}")
                else:
                    print("[!] Error: Authenticated exchange instance is not initialized.")

        # Intrabar processing of active position
        exit_flag = decision.exit_triggered
        prior_trade = self.portfolio.active_trade
        self.portfolio.process_bar(
            bar_idx=curr_bar_idx + 1,
            timestamp=recent_df.index[-1],
            open_price=float(active_forming_bar["open"]),
            high_price=float(active_forming_bar["high"]),
            low_price=float(active_forming_bar["low"]),
            close_price=float(active_forming_bar["close"]),
            current_atr=current_atr,
            exit_indicator_signal=exit_flag,
        )

        # Check if trade closed on this bar
        if prior_trade is not None and self.portfolio.active_trade is None:
            completed = self.portfolio.completed_trades[-1]
            status = "WIN" if completed.total_pnl_dollars > 0 else "LOSS"
            n_closed = len(self.portfolio.completed_trades)
            n_wins = sum(1 for t in self.portfolio.completed_trades if t.total_pnl_dollars > 0)
            wr = (n_wins / n_closed * 100.0) if n_closed > 0 else 0.0
            print(f"[+] [TRADE CLOSED] Trade #{completed.trade_id} [{status}] PnL: ${completed.total_pnl_dollars:+,.2f} ({completed.return_pct:+.2f}%)")
            print(f"    Account Equity: ${self.portfolio.current_equity:,.2f} | Realized Win Rate: {wr:.1f}% ({n_wins}/{n_closed} Wins)")

        self.last_processed_timestamp = latest_closed_ts
        return {
            "status": "bar_processed",
            "timestamp": latest_closed_ts,
            "decision": decision,
            "active_trade": self.portfolio.active_trade is not None,
            "equity": self.portfolio.current_equity,
        }

    def run_loop(self, max_iterations: Optional[int] = None):
        """Continuous live execution loop with graceful shutdown support."""
        self.is_running = True

        def handle_shutdown(signum, frame):
            print("\n[*] Received shutdown signal. Halting live trader cleanly...")
            self.is_running = False

        try:
            signal.signal(signal.SIGINT, handle_shutdown)
            signal.signal(signal.SIGTERM, handle_shutdown)
        except Exception:
            pass

        print(f"[*] Starting live polling loop (poll interval: {self.exchange_cfg.poll_interval_sec}s)...")
        iterations = 0
        while self.is_running:
            try:
                self.poll_and_evaluate_step()
            except Exception as e:
                print(f"[!] Warning during poll step: {e}")

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                print(f"[*] Reached maximum iterations ({max_iterations}). Stopping.")
                break

            time.sleep(self.exchange_cfg.poll_interval_sec)

        print("[+] Live trader stopped.")
