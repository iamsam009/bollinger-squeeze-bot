"""
Risk Manager
============
Handles:
- IST trading session filtering
- Daily loss limit (₹3,000)
- Maximum trade count per day (30 trades)
- Midnight IST daily reset
- Position quantity calculation from INR trade size
"""

import logging
from datetime import datetime, date, time as dt_time
from typing import Optional, Tuple, List
import pytz

from config import (
    IST_SESSIONS,
    TIMEZONE_IST,
    TRADE_SIZE_INR,
    DEFAULT_LEVERAGE,
    USD_INR_RATE,
    DAILY_LOSS_LIMIT_INR,
    MAX_TRADES_PER_DAY,
    SYMBOL,
    MIN_POSITION_SIZE_USD,
)

logger = logging.getLogger(__name__)

# Cache the timezone object
_ist_tz = pytz.timezone(TIMEZONE_IST)


# =============================================================================
# Session Management
# =============================================================================
def get_ist_now() -> datetime:
    """Get current datetime in Asia/Kolkata timezone."""
    return datetime.now(_ist_tz)


def is_trading_session(
    dt_now: Optional[datetime] = None,
    sessions: Optional[List[Tuple[dt_time, dt_time]]] = None,
) -> bool:
    """
    Check if current IST time falls within any trading session window.
    
    Args:
        dt_now: Datetime to check (defaults to now in IST)
        sessions: Optional override list of (start, end) time tuples.
                  If None, uses IST_SESSIONS from config.
    """
    if dt_now is None:
        dt_now = get_ist_now()

    if sessions is None:
        sessions = IST_SESSIONS

    current_time = dt_now.time()

    for start, end in sessions:
        if start <= current_time <= end:
            return True
    return False


def get_current_session_name(dt_now: Optional[datetime] = None) -> str:
    """Get the name of the current trading session (for display)."""
    if dt_now is None:
        dt_now = get_ist_now()

    current_time = dt_now.time()
    session_names = ["Morning (09:30-12:00)", "Afternoon (13:00-15:30)", "Evening (19:00-22:00)"]

    for (start, end), name in zip(IST_SESSIONS, session_names):
        if start <= current_time <= end:
            return name
    return "Outside Trading Hours"


def seconds_until_next_session(dt_now: Optional[datetime] = None) -> float:
    """
    Calculate seconds until the next trading session starts.
    Returns 0 if currently in a session.
    """
    if dt_now is None:
        dt_now = get_ist_now()

    if is_trading_session(dt_now):
        return 0.0

    current_time = dt_now.time()
    today = dt_now.date()
    from datetime import timedelta

    # Check each session today
    for start, end in IST_SESSIONS:
        if current_time < start:
            session_start_dt = _ist_tz.localize(datetime.combine(today, start))
            delta = session_start_dt - dt_now
            return max(delta.total_seconds(), 0)

    # All sessions passed for today, return time until first session tomorrow
    tomorrow = today + timedelta(days=1)
    first_start = IST_SESSIONS[0][0]
    tomorrow_start = _ist_tz.localize(datetime.combine(tomorrow, first_start))
    delta = tomorrow_start - dt_now
    return max(delta.total_seconds(), 0)


# =============================================================================
# Position Sizing
# =============================================================================
def calculate_position_size(
    entry_price: float,
    trade_size_inr: float = TRADE_SIZE_INR,
    leverage: int = DEFAULT_LEVERAGE,
    usd_inr_rate: float = USD_INR_RATE,
) -> Tuple[float, float]:
    """
    Calculate position size in contracts and notional value.

    Trade size: ₹20,000 (configurable)
    Leverage: 10x default
    USD/INR rate: 83.0 default

    Returns:
        (quantity_in_contracts, notional_value_in_usd)

    Formula:
        notional_usd = trade_size_inr / usd_inr_rate * leverage
        quantity = notional_usd / entry_price
    """
    notional_inr = trade_size_inr * leverage
    notional_usd = notional_inr / usd_inr_rate
    quantity = notional_usd / entry_price

    # Round quantity to appropriate precision (BTC typically 3 decimal places for contracts)
    # But SharkEx may use different precision; we round to 3dp for safety
    quantity = round(quantity, 3)

    if quantity <= 0:
        return 0.0, 0.0

    return quantity, notional_usd


def validate_min_position(notional_usd: float, min_notional: float = MIN_POSITION_SIZE_USD) -> bool:
    """Check if the position meets minimum notional requirements."""
    return notional_usd >= min_notional


# =============================================================================
# Daily Risk Limits
# =============================================================================
def get_ist_today() -> date:
    """Get current date in IST timezone."""
    return get_ist_now().date()


def is_new_day(last_reset_date: Optional[date] = None) -> bool:
    """
    Check if a new day has started in IST (daily reset at midnight IST).
    Returns True if last_reset_date is None or different from today.
    """
    today = get_ist_today()
    return last_reset_date is None or last_reset_date != today


def check_daily_loss_limit(
    daily_pnl_inr: float,
    loss_limit_inr: float = DAILY_LOSS_LIMIT_INR,
) -> bool:
    """
    Check if daily loss limit has been reached.
    Returns True if trading is allowed (loss within limit).
    """
    # Daily loss limit is negative: if P&L <= -3000, stop trading
    return daily_pnl_inr > -loss_limit_inr


def check_max_trades(
    daily_trade_count: int,
    max_trades: int = MAX_TRADES_PER_DAY,
) -> bool:
    """
    Check if max trades per day limit has been hit.
    Returns True if trading is allowed (under limit).
    """
    return daily_trade_count < max_trades


def can_trade(
    daily_pnl_inr: float,
    daily_trade_count: int,
    dt_now: Optional[datetime] = None,
    loss_limit_inr: Optional[float] = None,
    max_trades: Optional[int] = None,
    sessions: Optional[List[Tuple[dt_time, dt_time]]] = None,
) -> Tuple[bool, str]:
    """
    Master check: can the bot place a new trade?

    Args:
        daily_pnl_inr: Current daily P&L in INR
        daily_trade_count: Number of trades so far today
        dt_now: Datetime to check (defaults to IST now)
        loss_limit_inr: Daily loss limit override (uses config default if None)
        max_trades: Max trades override (uses config default if None)
        sessions: Session time override (uses config default if None)

    Returns:
        (allowed: bool, reason: str)
    """
    if loss_limit_inr is None:
        loss_limit_inr = DAILY_LOSS_LIMIT_INR
    if max_trades is None:
        max_trades = MAX_TRADES_PER_DAY

    if dt_now is None:
        dt_now = get_ist_now()

    if not is_trading_session(dt_now, sessions=sessions):
        return False, "Outside trading session hours"

    if not check_daily_loss_limit(daily_pnl_inr, loss_limit_inr=loss_limit_inr):
        return False, f"Daily loss limit reached (₹{loss_limit_inr})"

    if not check_max_trades(daily_trade_count, max_trades=max_trades):
        return False, f"Max trades per day reached ({max_trades})"

    return True, "OK"


# =============================================================================
# P&L Calculation Helpers
# =============================================================================
def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    direction: str,
    fee_pct: float = 0.0005,  # 0.05% estimated fee
) -> float:
    """
    Calculate realized P&L for a trade in USD.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position size in contracts
        direction: "LONG" or "SHORT"
        fee_pct: Trading fee percentage (default 0.05%)

    Returns:
        P&L in USD (positive = profit, negative = loss)
    """
    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * quantity
    elif direction == "SHORT":
        gross_pnl = (entry_price - exit_price) * quantity
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # Deduct fees (entry + exit)
    notional = entry_price * quantity
    fee_entry = notional * fee_pct
    fee_exit = exit_price * quantity * fee_pct

    net_pnl = gross_pnl - fee_entry - fee_exit
    return net_pnl


def usd_to_inr(usd_amount: float, rate: float = USD_INR_RATE) -> float:
    """Convert USD amount to INR."""
    return usd_amount * rate


def inr_to_usd(inr_amount: float, rate: float = USD_INR_RATE) -> float:
    """Convert INR amount to USD."""
    return inr_amount / rate