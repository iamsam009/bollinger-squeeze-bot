"""
Bollinger Squeeze Breakout Trading Bot - Streamlit Dashboard
=============================================================
BTC/USDT Perpetual Futures on SharkEx Exchange
Version 1.0.0

Main dashboard integrating:
- Real-time market data (60s auto-refresh)
- Bollinger Bands squeeze breakout strategy
- Trailing stop management
- Risk management (IST sessions, daily loss limit, max trades)
- Position management
- Emergency controls
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from config import (
    DASHBOARD_REFRESH_SECONDS,
    DASHBOARD_TITLE,
    DASHBOARD_SUBTITLE,
    SYMBOL,
    BB_PERIOD,
    BB_STD_DEV,
    BB_SQUEEZE_LOOKBACK,
    TRAILING_STOP_WINDOW,
    KLINE_INTERVAL,
    KLINE_LIMIT,
    DEFAULT_LEVERAGE,
    TRADE_SIZE_INR,
    DAILY_LOSS_LIMIT_INR,
    MAX_TRADES_PER_DAY,
    USD_INR_RATE,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_STOP_MARKET,
    ORDER_TYPE_STOP_LIMIT,
    ORDER_SIDE_BUY,
    ORDER_SIDE_SELL,
    IST_SESSIONS,
    TIMEZONE_IST,
    STOP_LIMIT_OFFSET_PCT,
)
from sharkex_client import (
    fetch_klines,
    fetch_ticker,
    fetch_depth,
    fetch_positions,
    fetch_open_orders,
    place_order,
    cancel_all_orders,
    close_all_positions,
    update_leverage,
    get_bid_ask,
    get_current_price,
    get_available_balance,
    fetch_usd_inr_rate,
)
from strategy import (
    prepare_strategy_df,
    evaluate_signal,
    calculate_trailing_stop,
    is_trailing_stop_hit,
    get_stop_limit_price,
    get_squeeze_info,
)
from risk_manager import (
    is_trading_session,
    get_ist_now,
    get_current_session_name,
    calculate_position_size,
    can_trade,
    check_daily_loss_limit,
    check_max_trades,
    usd_to_inr,
    inr_to_usd,
)
from state_manager import StateManager
import pytz

# =============================================================================
# Logger Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", mode="a"),
    ],
)
logger = logging.getLogger("BollingerBot")

# =============================================================================
# Session Config Persistence (survives browser refreshes)
# =============================================================================
SESSION_CONFIG_FILE = "session_config.json"

PERSISTED_KEYS = [
    "api_key", "api_secret",
    "bb_period", "bb_std_dev", "bb_squeeze_lookback", "trailing_stop_window",
    "leverage", "trade_size_inr", "daily_loss_limit_inr", "max_trades_per_day",
    "usd_inr_rate", "bot_running",
    "session_1_start", "session_1_end",
    "session_2_start", "session_2_end",
    "session_3_start", "session_3_end",
]


def _load_session_config() -> dict:
    """Load persisted session config from JSON file."""
    try:
        if os.path.exists(SESSION_CONFIG_FILE):
            with open(SESSION_CONFIG_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_session_config() -> None:
    """Save config keys that should survive refreshes."""
    try:
        data = {}
        for key in PERSISTED_KEYS:
            if key in st.session_state:
                val = st.session_state[key]
                # Skip non-serializable objects (like StateManager)
                if isinstance(val, (str, int, float, bool, type(None))):
                    data[key] = val
        with open(SESSION_CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save session config: {e}")


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Session State Initialization
# =============================================================================
def init_session_state() -> None:
    """Initialize all Streamlit session state variables.
    Loads persisted config from session_config.json so API keys and
    parameter edits survive browser refreshes.
    """
    # Load persisted config (API keys, strategy params) from disk
    persisted = _load_session_config()

    defaults = {
        # API Keys (loaded from disk)
        "api_key": persisted.get("api_key", ""),
        "api_secret": persisted.get("api_secret", ""),
        "api_configured": False,  # recomputed below

        # Strategy Parameters (loaded from disk if available)
        "bb_period": persisted.get("bb_period", BB_PERIOD),
        "bb_std_dev": persisted.get("bb_std_dev", BB_STD_DEV),
        "bb_squeeze_lookback": persisted.get("bb_squeeze_lookback", BB_SQUEEZE_LOOKBACK),
        "trailing_stop_window": persisted.get("trailing_stop_window", TRAILING_STOP_WINDOW),
        "leverage": persisted.get("leverage", DEFAULT_LEVERAGE),
        "trade_size_inr": persisted.get("trade_size_inr", TRADE_SIZE_INR),
        "daily_loss_limit_inr": persisted.get("daily_loss_limit_inr", DAILY_LOSS_LIMIT_INR),
        "max_trades_per_day": persisted.get("max_trades_per_day", MAX_TRADES_PER_DAY),
        "usd_inr_rate": persisted.get("usd_inr_rate", USD_INR_RATE),

        # Bot State
        "bot_running": persisted.get("bot_running", False),
        "last_cycle_time": None,
        "last_signal": None,
        "last_error": None,
        "cycle_count": 0,
        "logs": [],

        # Market Data (cached between cycles)
        "current_price": None,
        "bid_price": None,
        "ask_price": None,
        "df_klines": None,
        "squeeze_info": None,
        "available_balance": None,

        # State Manager (initialized once)
        "state_manager": StateManager(),

        # Session times (loaded from disk if available)
        "session_1_start": persisted.get("session_1_start", "09:30"),
        "session_1_end": persisted.get("session_1_end", "12:00"),
        "session_2_start": persisted.get("session_2_start", "13:00"),
        "session_2_end": persisted.get("session_2_end", "15:30"),
        "session_3_start": persisted.get("session_3_start", "19:00"),
        "session_3_end": persisted.get("session_3_end", "22:00"),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Recompute api_configured
    st.session_state.api_configured = bool(
        st.session_state.api_key and st.session_state.api_secret
    )


def add_log(message: str, level: str = "INFO") -> None:
    """Add a log message to the session state log buffer."""
    timestamp = get_ist_now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "time": timestamp,
        "level": level,
        "message": message,
    })
    # Keep only last 200 logs
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]

    # Also log to file
    log_func = {
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
    }.get(level, logger.info)
    log_func(message)


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> None:
    """Render the sidebar with API keys, strategy parameters, and controls."""
    with st.sidebar:
        st.header("🔑 API Configuration")

        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Enter SharkEx API Key",
        )
        api_secret = st.text_input(
            "API Secret",
            value=st.session_state.api_secret,
            type="password",
            placeholder="Enter SharkEx API Secret",
        )

        st.session_state.api_key = api_key
        st.session_state.api_secret = api_secret
        st.session_state.api_configured = bool(api_key and api_secret)
        # Persist API keys immediately
        _save_session_config()

        if st.session_state.api_configured:
            st.success("✅ API Keys Configured")
        else:
            st.warning("⚠️ Enter API keys to start trading")

        st.divider()

        # ---- Strategy Parameters ----
        st.header("📊 Strategy Parameters")
        with st.expander("Bollinger Bands", expanded=True):
            st.session_state.bb_period = st.number_input(
                "BB Period",
                min_value=5,
                max_value=50,
                value=st.session_state.bb_period,
                step=1,
            )
            st.session_state.bb_std_dev = st.number_input(
                "Std Deviation",
                min_value=1.0,
                max_value=4.0,
                value=st.session_state.bb_std_dev,
                step=0.1,
            )
            st.session_state.bb_squeeze_lookback = st.number_input(
                "Squeeze Lookback (N candles)",
                min_value=3,
                max_value=30,
                value=st.session_state.bb_squeeze_lookback,
                step=1,
            )

        with st.expander("Trailing Stop", expanded=False):
            st.session_state.trailing_stop_window = st.number_input(
                "Trailing Stop Window (candles)",
                min_value=2,
                max_value=20,
                value=st.session_state.trailing_stop_window,
                step=1,
            )

        st.divider()

        # ---- Risk Management ----
        st.header("⚖️ Risk Management")
        with st.expander("Position & Leverage", expanded=False):
            st.session_state.leverage = st.slider(
                "Leverage",
                min_value=1,
                max_value=125,
                value=st.session_state.leverage,
                step=1,
            )
            st.session_state.trade_size_inr = st.number_input(
                "Trade Size (₹)",
                min_value=500.0,
                max_value=100000.0,
                value=st.session_state.trade_size_inr,
                step=500.0,
            )
            st.session_state.usd_inr_rate = st.number_input(
                "USD/INR Rate",
                min_value=70.0,
                max_value=120.0,
                value=st.session_state.usd_inr_rate,
                step=0.1,
            )
            if st.button("🔄 Refresh Rate", help="Fetch live USD/INR rate from forex API"):
                with st.spinner("Fetching live rate..."):
                    live_rate = fetch_usd_inr_rate()
                    if live_rate:
                        st.session_state.usd_inr_rate = live_rate
                        st.success(f"Updated: ₹{live_rate:.2f}")
                        _save_session_config()
                        st.rerun()
                    else:
                        st.error("Could not fetch rate. Check internet connection.")

        with st.expander("Daily Limits", expanded=False):
            st.session_state.daily_loss_limit_inr = st.number_input(
                "Daily Loss Limit (₹)",
                min_value=1000.0,
                max_value=50000.0,
                value=st.session_state.daily_loss_limit_inr,
                step=500.0,
            )
            st.session_state.max_trades_per_day = st.number_input(
                "Max Trades/Day",
                min_value=1,
                max_value=100,
                value=st.session_state.max_trades_per_day,
                step=1,
            )

        st.divider()

        # ---- Session Times ----
        st.header("🕐 Trading Sessions (IST)")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.session_1_start = st.text_input(
                "S1 Start", value=st.session_state.session_1_start, key="s1s"
            )
            st.session_state.session_2_start = st.text_input(
                "S2 Start", value=st.session_state.session_2_start, key="s2s"
            )
            st.session_state.session_3_start = st.text_input(
                "S3 Start", value=st.session_state.session_3_start, key="s3s"
            )
        with col2:
            st.session_state.session_1_end = st.text_input(
                "S1 End", value=st.session_state.session_1_end, key="s1e"
            )
            st.session_state.session_2_end = st.text_input(
                "S2 End", value=st.session_state.session_2_end, key="s2e"
            )
            st.session_state.session_3_end = st.text_input(
                "S3 End", value=st.session_state.session_3_end, key="s3e"
            )

        # Persist all sidebar changes after widget rendering
        _save_session_config()

        st.divider()

        # ---- Bot Controls ----
        st.header("🎮 Controls")

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.bot_running:
                if st.button("▶️ Start Bot", type="primary", use_container_width=True):
                    if not st.session_state.api_configured:
                        st.error("Please enter API keys first!")
                    else:
                        st.session_state.bot_running = True
                        add_log("Bot started by user", "INFO")
                        st.rerun()
            else:
                if st.button("⏸️ Stop Bot", use_container_width=True):
                    st.session_state.bot_running = False
                    add_log("Bot stopped by user", "WARNING")
                    st.rerun()

        with col2:
            if st.button(
                "🛑 CLOSE ALL",
                type="secondary",
                use_container_width=True,
                help="Emergency: Close all positions and cancel all orders",
            ):
                handle_emergency_close()

        # Status indicators
        st.divider()
        ist_now = get_ist_now()
        sidebar_sessions = _parse_session_times() or None
        in_session = is_trading_session(ist_now, sessions=sidebar_sessions)
        session_name = get_current_session_name(ist_now)

        st.metric("IST Time", ist_now.strftime("%H:%M:%S"))
        st.metric("Trading Session", "🟢 " + session_name if in_session else "🔴 " + session_name)
        st.metric("Bot Status", "🟢 Running" if st.session_state.bot_running else "🔴 Stopped")

        if st.session_state.last_cycle_time:
            st.caption(f"Last cycle: {st.session_state.last_cycle_time.strftime('%H:%M:%S')} IST")

        # Quick stats
        sm = st.session_state.state_manager
        if st.session_state.bot_running or sm.has_open_position():
            st.divider()
            st.caption(f"Daily Trades: {sm.get_daily_trade_count()}/{st.session_state.max_trades_per_day}")
            st.caption(f"Daily P&L: ₹{sm.get_daily_pnl_inr():,.2f}")
            st.caption(f"Loss Limit: ₹{st.session_state.daily_loss_limit_inr:,}")


# =============================================================================
# Emergency Close
# =============================================================================
def _parse_session_times() -> list:
    """
    Parse the sidebar session time strings into a list of
    (datetime.time, datetime.time) tuples for risk_manager overrides.
    Returns an empty list if any field is malformed.
    """
    from datetime import time as dt_time
    sessions = []
    fields = [
        (st.session_state.session_1_start, st.session_state.session_1_end),
        (st.session_state.session_2_start, st.session_state.session_2_end),
        (st.session_state.session_3_start, st.session_state.session_3_end),
    ]
    for start_str, end_str in fields:
        try:
            sh, sm_val = map(int, start_str.strip().split(":"))
            start_t = dt_time(sh, sm_val)
            eh, em = map(int, end_str.strip().split(":"))
            end_t = dt_time(eh, em)
            sessions.append((start_t, end_t))
        except (ValueError, AttributeError):
            continue
    return sessions


def handle_emergency_close() -> None:
    """Emergency close all positions and cancel orders."""
    if not st.session_state.api_configured:
        st.error("API keys not configured!")
        return

    api_key = st.session_state.api_key
    api_secret = st.session_state.api_secret
    sm = st.session_state.state_manager

    try:
        # Cancel all open orders first
        add_log("EMERGENCY: Cancelling all open orders...", "WARNING")
        result = cancel_all_orders(api_key, api_secret, SYMBOL)
        add_log(f"Orders cancelled: {result}", "WARNING")

        # Close all positions
        add_log("EMERGENCY: Closing all positions...", "WARNING")
        result = close_all_positions(api_key, api_secret, SYMBOL)
        add_log(f"Positions closed: {result}", "WARNING")

        # Update local state
        if sm.has_open_position():
            current_price = get_current_price(SYMBOL) or 0
            sm.close_position(current_price, exit_reason="EMERGENCY")
            add_log(f"Local position closed at {current_price:.2f}", "WARNING")

        sm.set_emergency_stop()
        st.session_state.bot_running = False
        st.error("🚨 EMERGENCY: All positions closed, bot stopped!")
        add_log("Emergency close complete - bot halted", "ERROR")

    except Exception as e:
        st.error(f"Emergency close failed: {e}")
        add_log(f"Emergency close error: {e}", "ERROR")


# =============================================================================
# Market Data Fetching
# =============================================================================
def fetch_market_data() -> None:
    """Fetch market data and update session state."""
    try:
        # NOTE: No @st.cache_data here. Caching failed API results (empty list / None)
        # caused the bot to dead-loop with "No market data available" for 30 seconds.
        # The 60-second auto-refresh + rate limiter in sharkex_client.py prevent abuse.
        klines = fetch_klines(
            SYMBOL,
            st.session_state.get("kline_interval", KLINE_INTERVAL),
            st.session_state.get("kline_limit", KLINE_LIMIT),
        )
        price = get_current_price(SYMBOL)
        bid, ask = get_bid_ask(SYMBOL)

        if price:
            st.session_state.current_price = price
        if bid:
            st.session_state.bid_price = bid
        if ask:
            st.session_state.ask_price = ask
        if klines:
            # Prepare strategy DataFrame
            df = prepare_strategy_df(klines)
            st.session_state.df_klines = df
            st.session_state.squeeze_info = get_squeeze_info(df)

        add_log(f"Market data: price={price}, bid={bid}, ask={ask}, candles={len(klines)}")

    except Exception as e:
        add_log(f"Market data error: {e}", "ERROR")
        logger.error(traceback.format_exc())


# =============================================================================
# Core Trading Logic
# =============================================================================
def manage_open_position(sm: StateManager, df: pd.DataFrame) -> None:
    """
    Manage an existing open position:
    - Update trailing stop
    - Check if trailing stop hit
    - Exit if conditions met
    """
    if not sm.has_open_position() or df.empty:
        return

    api_key = st.session_state.api_key
    api_secret = st.session_state.api_secret
    pos = sm.get_position()

    # Calculate updated trailing stop
    trail_result = calculate_trailing_stop(
        df=df,
        direction=pos.direction,
        current_level=pos.trailing_stop_level,
        window=st.session_state.trailing_stop_window,
    )

    if trail_result.updated:
        sm.update_trailing_stop(trail_result.level)
        add_log(
            f"Trailing stop updated: {trail_result.previous_level:.2f} -> {trail_result.level:.2f}",
            "INFO",
        )

    # Check if trailing stop is hit
    current_price = st.session_state.current_price or float(df.iloc[-1]["close"])
    if is_trailing_stop_hit(current_price, pos.trailing_stop_level, pos.direction):
        add_log(
            f"Trailing stop HIT! Price={current_price:.2f}, Stop={pos.trailing_stop_level:.2f}",
            "WARNING",
        )
        exit_position(sm, current_price, reason="TRAILING_STOP")


def exit_position(sm: StateManager, exit_price: float, reason: str) -> None:
    """
    Exit the current position:
    1. Cancel the existing stop-loss order
    2. Place a market order to close
    3. Update state
    """
    api_key = st.session_state.api_key
    api_secret = st.session_state.api_secret
    pos = sm.get_position()

    try:
        # Cancel existing stop-loss order if any
        if pos.stop_loss_order_id:
            try:
                from sharkex_client import delete_order
                delete_order(api_key, api_secret, pos.stop_loss_order_id)
                add_log(f"Cancelled stop-loss order: {pos.stop_loss_order_id}")
            except Exception as e:
                add_log(f"Failed to cancel stop-loss: {e}", "WARNING")

        # Cancel all open orders for this symbol
        cancel_all_orders(api_key, api_secret, SYMBOL)

        # Place market order to close
        # Direction: if we're LONG, we SELL to close. If SHORT, we BUY to close.
        close_side = ORDER_SIDE_SELL if pos.direction == "LONG" else ORDER_SIDE_BUY

        try:
            # Use stop-limit as fallback approach: place a limit order at current price
            # For immediate execution, use a limit order slightly favoring execution
            limit_offset_pct = 0.001  # 0.1% slippage tolerance
            if pos.direction == "LONG":
                limit_price = exit_price * (1 - limit_offset_pct)
            else:
                limit_price = exit_price * (1 + limit_offset_pct)

            result = place_order(
                api_key=api_key,
                api_secret=api_secret,
                symbol=SYMBOL,
                side=close_side,
                order_type=ORDER_TYPE_LIMIT,
                quantity=pos.quantity,
                price=limit_price,
                reduce_only=True,
            )
            add_log(f"Exit order placed: {close_side} {pos.quantity} @ {limit_price:.2f} | Result: {result.get('clientOrderId', 'N/A')}")
        except Exception as e:
            add_log(f"Exit order failed, trying market order: {e}", "WARNING")
            # Fallback to market order
            result = place_order(
                api_key=api_key,
                api_secret=api_secret,
                symbol=SYMBOL,
                side=close_side,
                order_type="MARKET",
                quantity=pos.quantity,
                reduce_only=True,
            )
            add_log(f"Exit MARKET order placed: {close_side} {pos.quantity}")

        # Record trade in state manager
        trade = sm.close_position(exit_price, exit_reason=reason)
        if trade:
            add_log(
                f"Trade closed: {pos.direction} | P&L: ₹{trade.pnl_inr:,.2f} (${trade.pnl_usd:,.2f}) | "
                f"Reason: {reason} | Duration: {trade.trade_duration_seconds:.0f}s"
            )
            st.toast(
                f"Trade Closed: {trade.direction} | ₹{trade.pnl_inr:,.2f} | {reason}",
                icon="💰",
            )

    except Exception as e:
        add_log(f"Error closing position: {e}", "ERROR")
        logger.error(traceback.format_exc())


def place_stop_loss_order(sm: StateManager, stop_price: float) -> None:
    """
    Place a stop-market order with stop-limit fallback for the open position.
    
    Strategy:
    1. First attempt: STOP_MARKET order at stop_price
    2. Fallback: STOP_LIMIT order with limit price 0.1% beyond stop
    """
    api_key = st.session_state.api_key
    api_secret = st.session_state.api_secret
    pos = sm.get_position()

    # Close side is opposite of position direction
    close_side = ORDER_SIDE_SELL if pos.direction == "LONG" else ORDER_SIDE_BUY

    try:
        # Attempt 1: Stop-market order
        result = place_order(
            api_key=api_key,
            api_secret=api_secret,
            symbol=SYMBOL,
            side=close_side,
            order_type=ORDER_TYPE_STOP_MARKET,
            quantity=pos.quantity,
            stop_price=stop_price,
            reduce_only=True,
        )
        order_id = result.get("clientOrderId", result.get("orderId", ""))
        sm.set_stop_loss_order_id(order_id)
        add_log(f"Stop-market order placed: {close_side} {pos.quantity} @ stop={stop_price:.2f}")
        return

    except Exception as e:
        add_log(f"Stop-market failed, trying stop-limit: {e}", "WARNING")

    try:
        # Attempt 2: Stop-limit with 0.1% offset
        limit_price = get_stop_limit_price(stop_price, pos.direction)
        result = place_order(
            api_key=api_key,
            api_secret=api_secret,
            symbol=SYMBOL,
            side=close_side,
            order_type=ORDER_TYPE_STOP_LIMIT,
            quantity=pos.quantity,
            price=limit_price,
            stop_price=stop_price,
            reduce_only=True,
        )
        order_id = result.get("clientOrderId", result.get("orderId", ""))
        sm.set_stop_loss_order_id(order_id)
        add_log(f"Stop-limit order placed: {close_side} {pos.quantity} @ limit={limit_price:.2f}, stop={stop_price:.2f}")
        return

    except Exception as e:
        add_log(f"Failed to place stop-loss order: {e}", "ERROR")
        logger.error(traceback.format_exc())


def enter_position(sm: StateManager, signal, df: pd.DataFrame) -> None:
    """
    Enter a new position based on the signal:
    1. Set leverage
    2. Calculate position size
    3. Place limit order at bid/ask
    4. Place stop-loss order
    """
    api_key = st.session_state.api_key
    api_secret = st.session_state.api_secret
    leverage = st.session_state.leverage
    trade_size_inr = st.session_state.trade_size_inr
    usd_inr_rate = st.session_state.usd_inr_rate

    direction = signal.signal_type
    entry_price = signal.entry_price
    initial_stop = signal.stop_loss_price

    if not entry_price or not initial_stop:
        add_log("Missing entry price or stop loss - cannot enter", "ERROR")
        return

    try:
        # Step 1: Set leverage
        add_log(f"Setting leverage to {leverage}x for {SYMBOL}")
        update_leverage(api_key, api_secret, SYMBOL, leverage)
        add_log(f"Leverage set to {leverage}x")

        # Step 2: Calculate position size
        qty, notional = calculate_position_size(
            entry_price=entry_price,
            trade_size_inr=trade_size_inr,
            leverage=leverage,
            usd_inr_rate=usd_inr_rate,
        )
        add_log(f"Position size: {qty} contracts, Notional: ${notional:,.2f}")

        if qty <= 0:
            add_log("Position size is zero - cannot enter", "ERROR")
            return

        # Step 3: Place limit order
        side = ORDER_SIDE_BUY if direction == "LONG" else ORDER_SIDE_SELL
        add_log(f"Placing {direction} limit order: {side} {qty} @ {entry_price:.2f}")

        result = place_order(
            api_key=api_key,
            api_secret=api_secret,
            symbol=SYMBOL,
            side=side,
            order_type=ORDER_TYPE_LIMIT,
            quantity=qty,
            price=entry_price,
        )
        add_log(f"Entry order response: {result}")

        client_order_id = result.get("clientOrderId", "")
        order_id = result.get("orderId", "")

        # Step 4: Record position in state
        # Calculate initial trailing stop level from entry signal
        trail_result = calculate_trailing_stop(
            df=df,
            direction=direction,
            current_level=None,
            window=st.session_state.trailing_stop_window,
        )

        sm.open_position(
            direction=direction,
            entry_price=entry_price,
            quantity=qty,
            leverage=leverage,
            notional_usd=notional,
            initial_stop_loss=initial_stop,
            trailing_stop_level=trail_result.level or initial_stop,
            entry_order_id=str(order_id),
            entry_client_order_id=str(client_order_id),
        )

        # Step 5: Place stop-loss order
        place_stop_loss_order(sm, initial_stop)

        add_log(
            f"✅ {direction} position ENTERED: {qty} @ {entry_price:.2f} | "
            f"Stop: {initial_stop:.2f} | Trail: {trail_result.level:.2f} | "
            f"Notional: ${notional:,.2f}"
        )
        st.toast(f"🎯 {direction} Entry: {qty} @ {entry_price:.2f}", icon="📈")

    except Exception as e:
        add_log(f"Failed to enter position: {e}", "ERROR")
        logger.error(traceback.format_exc())


def check_session_end_exit(sm: StateManager, sessions=None) -> None:
    """
    Check if we need to exit due to session end.
    Close positions when session ends.
    """
    if not sm.has_open_position():
        return

    ist_now = get_ist_now()
    if is_trading_session(ist_now, sessions=sessions):
        return  # Still in session

    # Session ended, close position
    current_price = st.session_state.current_price
    if not current_price:
        try:
            current_price = get_current_price(SYMBOL) or 0
        except Exception:
            current_price = 0

    add_log(f"Session ended - closing position at {current_price:.2f}", "WARNING")
    exit_position(sm, current_price or 0, reason="SESSION_END")


# =============================================================================
# Main Bot Cycle
# =============================================================================
def run_bot_cycle() -> None:
    """
    Execute one cycle of the trading bot:
    1. Check daily reset
    2. Fetch market data
    3. Evaluate strategy
    4. Manage existing position OR check for entry
    5. Check session end
    """
    sm: StateManager = st.session_state.state_manager

    # Daily reset check
    sm.check_daily_reset()

    # Fetch market data
    fetch_market_data()

    df = st.session_state.df_klines
    if df is None or df.empty:
        add_log("No market data available - skipping cycle", "WARNING")
        return

    # Update available balance
    if st.session_state.api_configured:
        try:
            balance = get_available_balance(
                st.session_state.api_key,
                st.session_state.api_secret,
            )
            if balance is not None:
                st.session_state.available_balance = balance
        except Exception:
            pass

    # Parse sidebar session times for override
    sidebar_sessions = _parse_session_times() or None

    # Check trading conditions with sidebar overrides
    can_trade_result, can_trade_reason = sm.can_open_trade(
        loss_limit_inr=st.session_state.daily_loss_limit_inr,
        max_trades=st.session_state.max_trades_per_day,
        sessions=sidebar_sessions,
    )
    in_session = is_trading_session(sessions=sidebar_sessions)

    # Manage existing position
    if sm.has_open_position():
        manage_open_position(sm, df)
        check_session_end_exit(sm, sessions=sidebar_sessions)
        return

    # If no position, check for entry signals
    if not sm.has_open_position():
        # Evaluate strategy signal
        signal = evaluate_signal(
            df=df,
            bid=st.session_state.bid_price,
            ask=st.session_state.ask_price,
        )
        st.session_state.last_signal = signal

        if signal.has_signal:
            add_log(
                f"Signal detected: {signal.signal_type} | "
                f"Entry: {signal.entry_price} | "
                f"Squeeze: {signal.squeeze_detected} | "
                f"Breakout: {signal.breakout_detected}",
                "INFO",
            )

            if not st.session_state.bot_running:
                add_log("Bot is stopped - signal ignored", "WARNING")
                return

            if not in_session:
                add_log("Outside trading session - signal ignored", "WARNING")
                return

            if not can_trade_result:
                add_log(f"Risk limit reached: {can_trade_reason} - signal ignored", "WARNING")
                return

            if sm.is_emergency_stopped():
                add_log("Emergency stop active - signal ignored", "ERROR")
                return

            # Enter the trade!
            enter_position(sm, signal, df)

    st.session_state.cycle_count += 1
    st.session_state.last_cycle_time = get_ist_now()


# =============================================================================
# Dashboard Render Functions
# =============================================================================
def render_metrics_row() -> None:
    """Render the top metrics row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_price = st.session_state.current_price
        if current_price:
            price_display = f"${current_price:,.2f}"
        else:
            price_display = "Loading..."
        st.metric(
            label=f"💎 {SYMBOL} Price",
            value=price_display,
            delta=None,
        )

    with col2:
        sm = st.session_state.state_manager
        daily_pnl = sm.get_daily_pnl_inr()
        st.metric(
            label="💰 Daily P&L (₹)",
            value=f"₹{daily_pnl:,.2f}",
            delta=None,
            delta_color="normal" if daily_pnl >= 0 else "inverse",
        )

    with col3:
        trade_count = sm.get_daily_trade_count()
        max_trades = st.session_state.max_trades_per_day
        st.metric(
            label="📊 Trades Today",
            value=f"{trade_count}/{max_trades}",
        )

    with col4:
        bid = st.session_state.bid_price
        ask = st.session_state.ask_price
        if bid and ask:
            spread = ask - bid
            spread_pct = (spread / ask) * 100
            st.metric(
                label="📉 Bid / Ask",
                value=f"{bid:,.2f} / {ask:,.2f}",
                delta=f"Spread: {spread:,.2f} ({spread_pct:.3f}%)",
            )
        else:
            st.metric(label="📉 Bid / Ask", value="Loading...")

    with col5:
        balance = st.session_state.available_balance
        if balance is not None:
            st.metric(
                label="🏦 Balance (USDT)",
                value=f"${balance:,.2f}",
            )
        else:
            st.metric(label="🏦 Balance (USDT)", value="N/A")


def render_position_box() -> None:
    """Render the current position details box."""
    sm = st.session_state.state_manager

    if not sm.has_open_position():
        st.info("📭 No open position")
        return

    pos = sm.get_position()
    current_price = st.session_state.current_price or pos.entry_price

    # Calculate unrealized P&L
    if pos.direction == "LONG":
        unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
    else:
        unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100

    unrealized_pnl_pct = unrealized_pnl_pct * pos.leverage  # Adjust for leverage

    # Color based on P&L
    box_color = "#d4edda" if unrealized_pnl_pct >= 0 else "#f8d7da"
    border_color = "#28a745" if unrealized_pnl_pct >= 0 else "#dc3545"

    st.markdown(
        f"""
        <div style="
            background-color: {box_color};
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        ">
            <h4 style="margin-top:0;">
                {'🟢' if pos.direction == 'LONG' else '🔴'}
                {pos.direction} Position
                <span style="font-size:0.8em;color:gray;">
                    ({pos.entry_time[:19] if pos.entry_time else 'N/A'})
                </span>
            </h4>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entry Price", f"${pos.entry_price:,.2f}")
        st.metric("Current Price", f"${current_price:,.2f}")
    with col2:
        st.metric("Quantity", f"{pos.quantity}")
        st.metric("Notional", f"${pos.notional_usd:,.2f}")
    with col3:
        st.metric("Leverage", f"{pos.leverage}x")
        st.metric(
            "Unrealized P&L %",
            f"{unrealized_pnl_pct:+.2f}%",
        )
    with col4:
        st.metric("Trailing Stop", f"${pos.trailing_stop_level:,.2f}")
        st.metric("Initial Stop", f"${pos.initial_stop_loss:,.2f}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_chart() -> None:
    """Render the price chart with Bollinger Bands."""
    df = st.session_state.df_klines

    if df is None or df.empty:
        st.warning("📊 No chart data available")
        return

    # Get last ~50 candles for chart clarity
    chart_df = df.tail(50).copy()

    # Create subplot with volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{SYMBOL} - {st.session_state.get('kline_interval', KLINE_INTERVAL)} Chart", "BB Width %"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Bollinger Bands
    if "bb_upper" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["bb_upper"],
                name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.8)", dash="dash"),
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["bb_sma"],
                name="BB SMA",
                line=dict(color="rgba(128, 128, 128, 0.8)", dash="dot"),
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["bb_lower"],
                name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.8)", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.1)",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Squeeze markers
        squeeze_df = chart_df[chart_df.get("bb_squeeze", False)]
        if not squeeze_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=squeeze_df.index,
                    y=squeeze_df["close"],
                    name="Squeeze",
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=10,
                        color="orange",
                        line=dict(color="red", width=1),
                    ),
                    showlegend=True,
                ),
                row=1, col=1,
            )

        # Breakout markers
        long_sig = chart_df[chart_df.get("long_signal", False)]
        short_sig = chart_df[chart_df.get("short_signal", False)]

        if not long_sig.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_sig.index,
                    y=long_sig["close"],
                    name="Long Signal",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=14, color="green"),
                    showlegend=True,
                ),
                row=1, col=1,
            )
        if not short_sig.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_sig.index,
                    y=short_sig["close"],
                    name="Short Signal",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=14, color="red"),
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # BB Width subplot
    if "bb_width_pct" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["bb_width_pct"],
                name="BB Width %",
                line=dict(color="purple", width=2),
                showlegend=False,
            ),
            row=2, col=1,
        )
        # Add a horizontal line at 0
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

    # Also add highest_high_n and lowest_low_n if available
    if "highest_high_n" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["highest_high_n"],
                name=f"Highest High ({st.session_state.bb_squeeze_lookback})",
                line=dict(color="rgba(0,200,0,0.4)", width=1),
                showlegend=False,
            ),
            row=1, col=1,
        )
    if "lowest_low_n" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["lowest_low_n"],
                name=f"Lowest Low ({st.session_state.bb_squeeze_lookback})",
                line=dict(color="rgba(200,0,0,0.4)", width=1),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Position entry marker
    sm = st.session_state.state_manager
    if sm.has_open_position():
        pos = sm.get_position()
        fig.add_hline(
            y=pos.entry_price,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Entry: {pos.entry_price:.2f}",
            row=1, col=1,
        )
        fig.add_hline(
            y=pos.trailing_stop_level,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Stop: {pos.trailing_stop_level:.2f}",
            row=1, col=1,
        )

    # Layout
    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
    fig.update_yaxes(title_text="Width %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_squeeze_status() -> None:
    """Render the squeeze detection status indicators."""
    info = st.session_state.squeeze_info
    if not info:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        in_squeeze = info.get("in_squeeze", False)
        st.metric(
            "🔍 Squeeze Status",
            "🟠 ACTIVE" if in_squeeze else "⚪ Inactive",
        )

    with col2:
        width_pct = info.get("width_pct", 0)
        st.metric("📏 BB Width %", f"{width_pct:.4f}%")

    with col3:
        squeeze_rank = info.get("squeeze_rank", 0)
        st.metric("🏆 Squeeze Rank", f"#{squeeze_rank}")

    with col4:
        close = info.get("close", 0)
        highest = info.get("highest_high_n", 0)
        lowest = info.get("lowest_low_n", 0)
        if close > highest:
            st.metric("🎯 Breakout Check", "Close > HH", delta="LONG potential")
        elif close < lowest:
            st.metric("🎯 Breakout Check", "Close < LL", delta="SHORT potential")
        else:
            st.metric("🎯 Breakout Check", "No breakout")


def render_next_trade_targets() -> None:
    """Render breakout trigger prices for the next potential trade."""
    info = st.session_state.squeeze_info
    if not info:
        return

    in_squeeze = info.get("in_squeeze", False)
    close = info.get("close", 0)
    highest_high_n = info.get("highest_high_n", 0)
    lowest_low_n = info.get("lowest_low_n", 0)
    sma = info.get("sma", 0)
    upper_band = info.get("upper_band", 0)
    lower_band = info.get("lower_band", 0)

    if not close or not highest_high_n or not lowest_low_n:
        return

    with st.expander("🎯 Next Trade Targets", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 LONG Entry")
            long_distance = highest_high_n - close
            long_distance_pct = (long_distance / close * 100) if close else 0

            st.metric(
                "Trigger Price (Break Above)",
                f"${highest_high_n:,.2f}",
                delta=f"+${long_distance:,.2f} ({long_distance_pct:+.2f}%)",
            )
            st.caption(f"Close must exceed the highest high of the last "
                       f"{st.session_state.get('bb_squeeze_lookback', 20)} candles"
                       f"{' AND squeeze must be active' if not in_squeeze else ''}")
            st.write(f"**Est. Entry at Ask:** ${st.session_state.get('ask_price', 'N/A')}")
            st.write(f"**Upper Band:** ${upper_band:,.2f}  |  **SMA:** ${sma:,.2f}")

        with col2:
            st.markdown("#### 📉 SHORT Entry")
            short_distance = close - lowest_low_n
            short_distance_pct = (short_distance / close * 100) if close else 0

            st.metric(
                "Trigger Price (Break Below)",
                f"${lowest_low_n:,.2f}",
                delta=f"-${short_distance:,.2f} ({short_distance_pct:+.2f}%)",
            )
            st.caption(f"Close must drop below the lowest low of the last "
                       f"{st.session_state.get('bb_squeeze_lookback', 20)} candles"
                       f"{' AND squeeze must be active' if not in_squeeze else ''}")
            st.write(f"**Est. Entry at Bid:** ${st.session_state.get('bid_price', 'N/A')}")
            st.write(f"**Lower Band:** ${lower_band:,.2f}  |  **SMA:** ${sma:,.2f}")

        # Show squeeze precondition status
        if not in_squeeze:
            st.warning("⚠️ Squeeze is NOT active — breakout signals will be ignored until squeeze forms")
        else:
            st.success("🟠 Squeeze is ACTIVE — breakout signals are armed")

        # Show which side is closer
        if close < highest_high_n or close > lowest_low_n:
            if long_distance < short_distance:
                st.info(f"⚡ LONG trigger is closer (${long_distance:,.2f} away vs ${short_distance:,.2f} for SHORT)")
            else:
                st.info(f"⚡ SHORT trigger is closer (${short_distance:,.2f} away vs ${long_distance:,.2f} for LONG)")


def render_signal_info() -> None:
    """Render the last signal details."""
    signal = st.session_state.last_signal
    if signal is None:
        return

    if signal.has_signal:
        with st.expander(f"🔔 Last Signal: {signal.signal_type}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Type:** {signal.signal_type}")
                st.write(f"**Entry Price:** ${signal.entry_price}")
            with col2:
                st.write(f"**Squeeze:** {signal.squeeze_detected}")
                st.write(f"**Breakout:** {signal.breakout_detected}")
            with col3:
                st.write(f"**BB Width:** {signal.bb_result.width_pct:.4f}%" if signal.bb_result else "")
                st.write(f"**Reason:** {signal.breakout_reason}")


def render_trade_log() -> None:
    """Render the expandable trade log."""
    sm = st.session_state.state_manager

    with st.expander("📋 Trade Log", expanded=False):
        trades = sm.get_recent_trades(50)

        if not trades:
            st.write("No trades recorded yet.")
            return

        # Build DataFrame for display
        data = []
        for t in reversed(trades):  # Most recent first
            data.append({
                "Trade ID": t.trade_id,
                "Time": t.exit_time[:19] if t.exit_time else "",
                "Direction": t.direction,
                "Entry": f"${t.entry_price:,.2f}",
                "Exit": f"${t.exit_price:,.2f}",
                "Qty": t.quantity,
                "Leverage": f"{t.leverage}x",
                "P&L (₹)": f"₹{t.pnl_inr:,.2f}",
                "P&L ($)": f"${t.pnl_usd:,.2f}",
                "Reason": t.exit_reason,
                "Duration": f"{t.trade_duration_seconds:.0f}s",
            })

        df_trades = pd.DataFrame(data)

        # Color rows based on P&L
        def color_pnl(val):
            if "₹" in str(val):
                num = float(str(val).replace("₹", "").replace(",", ""))
                color = "green" if num > 0 else "red" if num < 0 else "gray"
                return f"color: {color}"
            return ""

        styled = df_trades.style.applymap(color_pnl, subset=["P&L (₹)", "P&L ($)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Summary stats
        total_pnl = sum(t.pnl_inr for t in trades)
        wins = sum(1 for t in trades if t.pnl_usd > 0)
        losses = sum(1 for t in trades if t.pnl_usd < 0)
        win_rate = (wins / len(trades) * 100) if trades else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            st.metric("Total P&L (₹)", f"₹{total_pnl:,.2f}")
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Wins/Losses", f"{wins}/{losses}")


def render_logs() -> None:
    """Render the bot activity logs."""
    with st.expander("📝 Bot Logs", expanded=False):
        logs = st.session_state.logs[-100:]  # Last 100 entries

        if not logs:
            st.write("No logs yet.")
            return

        # Build log text
        log_lines = []
        for log in reversed(logs):
            color = {
                "ERROR": "🔴",
                "WARNING": "🟡",
                "INFO": "🔵",
            }.get(log["level"], "⚪")
            log_lines.append(f"{color} `{log['time']}` [{log['level']}] {log['message']}")

        log_text = "\n".join(log_lines)

        st.markdown(log_text)


# =============================================================================
# Main App
# =============================================================================
def main() -> None:
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()

    # Auto-refresh
    st_autorefresh(interval=DASHBOARD_REFRESH_SECONDS * 1000, key="bot_autorefresh")

    # Title
    st.title(DASHBOARD_TITLE)
    st.caption(DASHBOARD_SUBTITLE)

    # Render sidebar
    render_sidebar()

    # ---- Main Dashboard Area ----

    # Metrics Row
    render_metrics_row()

    # Position Box
    render_position_box()

    # Chart
    render_chart()

    # Squeeze Status
    render_squeeze_status()

    # Next Trade Targets (breakout trigger prices)
    render_next_trade_targets()

    # Signal Info
    render_signal_info()

    # ---- Run Bot Cycle ----
    if st.session_state.api_configured:
        try:
            run_bot_cycle()
        except Exception as e:
            add_log(f"Bot cycle error: {e}", "ERROR")
            logger.error(traceback.format_exc())
            st.error(f"Bot cycle error: {e}")
    else:
        st.warning("⚠️ Enter API keys in the sidebar to start the bot")

    # Trade Log
    render_trade_log()

    # Activity Logs
    render_logs()

    # Footer
    st.divider()
    ist_now = get_ist_now()
    st.caption(
        f"Last updated: {ist_now.strftime('%Y-%m-%d %H:%M:%S')} IST | "
        f"Cycle: {st.session_state.cycle_count} | "
        f"Session: {get_current_session_name(ist_now)}"
    )


if __name__ == "__main__":
    main()