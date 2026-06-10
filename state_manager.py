"""
State Manager
=============
Manages:
- Bot running state (position tracking)
- Daily P&L tracking (USD and INR)
- Trade log (list of completed trades)
- Daily trade counter
- Midnight IST reset logic
- Persistent storage (JSON files)
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict

from config import (
    TRADE_LOG_FILE,
    STATE_FILE,
    DEFAULT_LEVERAGE,
    USD_INR_RATE,
    DAILY_LOSS_LIMIT_INR,
    MAX_TRADES_PER_DAY,
    SYMBOL,
)
from risk_manager import get_ist_today, is_new_day, usd_to_inr

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class TradeRecord:
    """A single completed trade record."""
    trade_id: str                   # Unique trade ID (timestamp-based)
    entry_time: str                 # ISO format entry time
    exit_time: str                  # ISO format exit time
    direction: str                  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl_usd: float
    pnl_inr: float
    exit_reason: str                # "TRAILING_STOP", "MANUAL_CLOSE", "SESSION_END", "STOP_LOSS"
    entry_signal: str = ""          # Description of the entry signal
    trade_duration_seconds: float = 0.0
    fees_usd: float = 0.0


@dataclass
class PositionState:
    """Current open position state."""
    is_open: bool = False
    direction: str = ""             # "LONG" or "SHORT"
    entry_price: float = 0.0
    entry_time: str = ""
    quantity: float = 0.0
    leverage: int = DEFAULT_LEVERAGE
    notional_usd: float = 0.0
    trailing_stop_level: float = 0.0
    initial_stop_loss: float = 0.0
    stop_loss_order_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    entry_client_order_id: Optional[str] = None


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str = ""                  # ISO date string
    trade_count: int = 0
    total_pnl_usd: float = 0.0
    total_pnl_inr: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown_inr: float = 0.0   # Peak to trough in INR
    peak_pnl_inr: float = 0.0       # Highest P&L point of the day


@dataclass
class BotState:
    """Overall bot state for persistence."""
    position: PositionState = field(default_factory=PositionState)
    daily_stats: DailyStats = field(default_factory=DailyStats)
    last_reset_date: str = ""       # ISO date of last daily reset
    last_update_time: str = ""      # ISO timestamp of last state update
    running: bool = True            # Bot running status
    emergency_stop: bool = False    # Emergency stop flag


# =============================================================================
# State Manager Class
# =============================================================================
class StateManager:
    """
    Manages bot state with persistence to JSON files.

    Handles:
    - Position tracking (open/close)
    - Trade logging
    - Daily P&L accumulation
    - Daily trade counter with midnight IST reset
    - Load/save state from disk
    """

    def __init__(self, state_file: str = STATE_FILE, trade_log_file: str = TRADE_LOG_FILE):
        self.state_file = state_file
        self.trade_log_file = trade_log_file
        self.state: BotState = BotState()
        self.trades: List[TradeRecord] = []
        self._load_state()
        self._load_trades()

    # -------------------------------------------------------------------------
    # Daily Reset
    # -------------------------------------------------------------------------
    def check_daily_reset(self) -> bool:
        """
        Check if a new IST day has started and reset daily counters if needed.
        Returns True if a reset was performed.
        """
        today = get_ist_today()
        last_reset = (
            date.fromisoformat(self.state.last_reset_date)
            if self.state.last_reset_date
            else None
        )
        if is_new_day(last_reset):
            logger.info(f"Daily reset: new IST day {today.isoformat()}")
            self.state.daily_stats = DailyStats(date=today.isoformat())
            self.state.last_reset_date = today.isoformat()
            self._save_state()
            return True
        return False

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------
    def has_open_position(self) -> bool:
        """Check if there's currently an open position."""
        return self.state.position.is_open

    def get_position(self) -> PositionState:
        """Get the current position state."""
        return self.state.position

    def open_position(
        self,
        direction: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        notional_usd: float,
        entry_time: Optional[str] = None,
        initial_stop_loss: float = 0.0,
        trailing_stop_level: float = 0.0,
        entry_order_id: Optional[str] = None,
        entry_client_order_id: Optional[str] = None,
    ) -> None:
        """
        Record a new open position.
        """
        if self.state.position.is_open:
            logger.warning("Attempted to open position while one is already open - closing previous first")
            self.close_position(exit_price=entry_price, exit_reason="REPLACED")

        self.state.position = PositionState(
            is_open=True,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now().isoformat(),
            quantity=quantity,
            leverage=leverage,
            notional_usd=notional_usd,
            trailing_stop_level=trailing_stop_level,
            initial_stop_loss=initial_stop_loss,
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
        )
        self.state.daily_stats.trade_count += 1
        self.state.last_update_time = datetime.now().isoformat()
        self._save_state()
        logger.info(
            f"Position opened: {direction} {quantity} @ {entry_price:.2f} "
            f"(notional=${notional_usd:.2f}, leverage={leverage}x)"
        )

    def close_position(
        self,
        exit_price: float,
        exit_reason: str = "MANUAL",
        exit_time: Optional[str] = None,
        stop_loss_order_id: Optional[str] = None,
    ) -> Optional[TradeRecord]:
        """
        Close the current position and record the trade.

        Args:
            exit_price: Price at which position was closed
            exit_reason: TRAILING_STOP, MANUAL_CLOSE, SESSION_END, STOP_LOSS
            exit_time: ISO timestamp of exit
            stop_loss_order_id: ID of the stop-loss order that triggered

        Returns:
            TradeRecord if position was closed, None if no position was open
        """
        if not self.state.position.is_open:
            logger.warning("Attempted to close position but none is open")
            return None

        pos = self.state.position
        exit_ts = exit_time or datetime.now().isoformat()

        # Calculate P&L
        from risk_manager import calculate_trade_pnl
        pnl_usd = calculate_trade_pnl(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            direction=pos.direction,
        )
        pnl_inr = usd_to_inr(pnl_usd)

        # Calculate trade duration
        try:
            entry_dt = datetime.fromisoformat(pos.entry_time)
            exit_dt = datetime.fromisoformat(exit_ts)
            duration = (exit_dt - entry_dt).total_seconds()
        except Exception:
            duration = 0.0

        # Create trade record
        trade = TradeRecord(
            trade_id=f"TRD-{exit_ts.replace(':', '').replace('-', '').replace('T', '-')}",
            entry_time=pos.entry_time,
            exit_time=exit_ts,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            leverage=pos.leverage,
            pnl_usd=pnl_usd,
            pnl_inr=pnl_inr,
            exit_reason=exit_reason,
            trade_duration_seconds=duration,
        )

        # Update daily stats
        self.state.daily_stats.total_pnl_usd += pnl_usd
        self.state.daily_stats.total_pnl_inr += pnl_inr
        if pnl_usd > 0:
            self.state.daily_stats.winning_trades += 1
        elif pnl_usd < 0:
            self.state.daily_stats.losing_trades += 1

        # Track peak P&L and drawdown
        if self.state.daily_stats.total_pnl_inr > self.state.daily_stats.peak_pnl_inr:
            self.state.daily_stats.peak_pnl_inr = self.state.daily_stats.total_pnl_inr
        dd = self.state.daily_stats.total_pnl_inr - self.state.daily_stats.peak_pnl_inr
        if dd < self.state.daily_stats.max_drawdown_inr:
            self.state.daily_stats.max_drawdown_inr = dd

        # Save trade log
        self.trades.append(trade)
        self._save_trades()

        # Reset position
        self.state.position = PositionState()
        self.state.last_update_time = exit_ts
        self._save_state()

        logger.info(
            f"Position closed: {pos.direction} @ {exit_price:.2f} | "
            f"P&L: ₹{pnl_inr:.2f} (${pnl_usd:.2f}) | "
            f"Reason: {exit_reason} | Duration: {duration:.0f}s"
        )

        return trade

    def update_trailing_stop(self, new_level: float) -> None:
        """Update the trailing stop level for the current position."""
        if self.state.position.is_open:
            old_level = self.state.position.trailing_stop_level
            self.state.position.trailing_stop_level = new_level
            logger.debug(f"Trailing stop updated: {old_level:.2f} -> {new_level:.2f}")
            self._save_state()

    def set_stop_loss_order_id(self, order_id: str) -> None:
        """Record the stop-loss order ID."""
        self.state.position.stop_loss_order_id = order_id
        self._save_state()

    # -------------------------------------------------------------------------
    # Daily Stats Accessors
    # -------------------------------------------------------------------------
    def get_daily_pnl_inr(self) -> float:
        """Get current daily P&L in INR."""
        return self.state.daily_stats.total_pnl_inr

    def get_daily_pnl_usd(self) -> float:
        """Get current daily P&L in USD."""
        return self.state.daily_stats.total_pnl_usd

    def get_daily_trade_count(self) -> int:
        """Get number of trades executed today."""
        return self.state.daily_stats.trade_count

    def can_open_trade(
        self,
        loss_limit_inr: Optional[float] = None,
        max_trades: Optional[int] = None,
        sessions: Optional[list] = None,
    ) -> tuple:
        """
        Check if a new trade can be opened based on risk limits.
        
        Args:
            loss_limit_inr: Daily loss limit override (uses config default if None)
            max_trades: Max trades override (uses config default if None)
            sessions: Session time override (uses config default if None)
        """
        from risk_manager import can_trade
        return can_trade(
            daily_pnl_inr=self.state.daily_stats.total_pnl_inr,
            daily_trade_count=self.state.daily_stats.trade_count,
            loss_limit_inr=loss_limit_inr,
            max_trades=max_trades,
            sessions=sessions,
        )

    # -------------------------------------------------------------------------
    # Trade Log
    # -------------------------------------------------------------------------
    def get_recent_trades(self, count: int = 50) -> List[TradeRecord]:
        """Get the most recent N trades."""
        return self.trades[-count:]

    def get_all_trades(self) -> List[TradeRecord]:
        """Get all recorded trades."""
        return self.trades

    def get_trades_today(self) -> List[TradeRecord]:
        """Get trades from today only."""
        today = get_ist_today().isoformat()
        return [
            t for t in self.trades
            if t.entry_time[:10] == today or t.exit_time[:10] == today
        ]

    # -------------------------------------------------------------------------
    # Emergency Controls
    # -------------------------------------------------------------------------
    def set_emergency_stop(self) -> None:
        """Set emergency stop flag (prevents new trades)."""
        self.state.emergency_stop = True
        self.state.running = False
        self._save_state()
        logger.warning("EMERGENCY STOP ACTIVATED - No new trades will be placed")

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop flag."""
        self.state.emergency_stop = False
        self.state.running = True
        self._save_state()
        logger.info("Emergency stop cleared - Trading resumed")

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active."""
        return self.state.emergency_stop

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def _save_state(self) -> None:
        """Save bot state to JSON file."""
        try:
            state_dict = {
                "position": asdict(self.state.position),
                "daily_stats": asdict(self.state.daily_stats),
                "last_reset_date": self.state.last_reset_date,
                "last_update_time": self.state.last_update_time,
                "running": self.state.running,
                "emergency_stop": self.state.emergency_stop,
            }
            with open(self.state_file, "w") as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load bot state from JSON file."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            pos_data = data.get("position", {})
            self.state.position = PositionState(
                is_open=pos_data.get("is_open", False),
                direction=pos_data.get("direction", ""),
                entry_price=pos_data.get("entry_price", 0.0),
                entry_time=pos_data.get("entry_time", ""),
                quantity=pos_data.get("quantity", 0.0),
                leverage=pos_data.get("leverage", DEFAULT_LEVERAGE),
                notional_usd=pos_data.get("notional_usd", 0.0),
                trailing_stop_level=pos_data.get("trailing_stop_level", 0.0),
                initial_stop_loss=pos_data.get("initial_stop_loss", 0.0),
                stop_loss_order_id=pos_data.get("stop_loss_order_id"),
                entry_order_id=pos_data.get("entry_order_id"),
                entry_client_order_id=pos_data.get("entry_client_order_id"),
            )

            stats_data = data.get("daily_stats", {})
            self.state.daily_stats = DailyStats(
                date=stats_data.get("date", ""),
                trade_count=stats_data.get("trade_count", 0),
                total_pnl_usd=stats_data.get("total_pnl_usd", 0.0),
                total_pnl_inr=stats_data.get("total_pnl_inr", 0.0),
                winning_trades=stats_data.get("winning_trades", 0),
                losing_trades=stats_data.get("losing_trades", 0),
                max_drawdown_inr=stats_data.get("max_drawdown_inr", 0.0),
                peak_pnl_inr=stats_data.get("peak_pnl_inr", 0.0),
            )

            self.state.last_reset_date = data.get("last_reset_date", "")
            self.state.last_update_time = data.get("last_update_time", "")
            self.state.running = data.get("running", True)
            self.state.emergency_stop = data.get("emergency_stop", False)

            logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def _save_trades(self) -> None:
        """Save trade log to JSON file."""
        try:
            trades_list = [asdict(t) for t in self.trades]
            with open(self.trade_log_file, "w") as f:
                json.dump(trades_list, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trade log: {e}")

    def _load_trades(self) -> None:
        """Load trade log from JSON file."""
        if not os.path.exists(self.trade_log_file):
            return
        try:
            with open(self.trade_log_file, "r") as f:
                data = json.load(f)
            self.trades = [TradeRecord(**t) for t in data]
            logger.info(f"Loaded {len(self.trades)} trades from {self.trade_log_file}")
        except Exception as e:
            logger.error(f"Failed to load trade log: {e}")

    # -------------------------------------------------------------------------
    # Status Summary for Dashboard
    # -------------------------------------------------------------------------
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a comprehensive status summary for the dashboard."""
        return {
            "running": self.state.running,
            "emergency_stop": self.state.emergency_stop,
            "position_open": self.state.position.is_open,
            "position_direction": self.state.position.direction,
            "position_entry_price": self.state.position.entry_price,
            "position_quantity": self.state.position.quantity,
            "position_notional_usd": self.state.position.notional_usd,
            "position_leverage": self.state.position.leverage,
            "trailing_stop_level": self.state.position.trailing_stop_level,
            "initial_stop_loss": self.state.position.initial_stop_loss,
            "daily_pnl_inr": self.state.daily_stats.total_pnl_inr,
            "daily_pnl_usd": self.state.daily_stats.total_pnl_usd,
            "daily_trade_count": self.state.daily_stats.trade_count,
            "daily_winning_trades": self.state.daily_stats.winning_trades,
            "daily_losing_trades": self.state.daily_stats.losing_trades,
            "daily_max_drawdown_inr": self.state.daily_stats.max_drawdown_inr,
            "daily_peak_pnl_inr": self.state.daily_stats.peak_pnl_inr,
            "last_reset_date": self.state.last_reset_date,
            "last_update_time": self.state.last_update_time,
            "total_trades_logged": len(self.trades),
        }