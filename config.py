"""
Bollinger Squeeze Breakout Trading Bot - Configuration
BTC/USDT Perpetual Futures on SharkEx Exchange
Version 1.0.0

All configuration constants centralized here for easy tuning.
"""

import os
from datetime import time as dt_time

# =============================================================================
# SharkEx API Configuration
# Derived from sharkex_docs.html
# =============================================================================
SHARKEX_BASE_URL = "https://api.sharkexchange.in"
SHARKEX_AUTH_URL = "https://api.sharkexchange.in"

# API Key & Secret - loaded from Streamlit session_state at runtime
# These are placeholder defaults; real keys come from the dashboard UI
API_KEY = os.environ.get("SHARKEX_API_KEY", "")
API_SECRET = os.environ.get("SHARKEX_API_SECRET", "")

# =============================================================================
# Trading Symbol & Market
# =============================================================================
SYMBOL = "BTCUSDT"              # Trading pair on SharkEx
QUOTE_ASSET = "USDT"            # Quote currency
MARGIN_ASSET = "USDT"           # Margin asset for futures
CONTRACT_TYPE = "PERPETUAL"     # Perpetual futures

# =============================================================================
# Bollinger Bands Strategy Parameters
# =============================================================================
BB_PERIOD = 20                  # SMA & std dev lookback period
BB_STD_DEV = 2.0                # Number of standard deviations
BB_SQUEEZE_LOOKBACK = 10        # Window for detecting squeeze (min width in N candles)

# =============================================================================
# Trailing Stop Parameters
# =============================================================================
TRAILING_STOP_WINDOW = 5        # Candles for trailing stop level calculation
STOP_LIMIT_OFFSET_PCT = 0.001   # 0.1% beyond stop price for stop-limit fallback

# =============================================================================
# Kline / Timeframe
# =============================================================================
KLINE_INTERVAL = "15m"          # 15-minute candles
KLINE_LIMIT = 100               # Fetch enough candles for indicators + lookback
                                 # Need: BB_PERIOD + BB_SQUEEZE_LOOKBACK + buffer ≈ 50+
                                 # Fetch 100 for safety

# =============================================================================
# Risk Management (INR-denominated)
# =============================================================================
TRADE_SIZE_INR = 20000.0        # ₹20,000 trade size
DEFAULT_LEVERAGE = 10           # Default leverage (1-125x range on SharkEx)
DAILY_LOSS_LIMIT_INR = 3000.0   # ₹3,000 max daily loss
MAX_TRADES_PER_DAY = 30         # Maximum 30 trades per day
MIN_POSITION_SIZE_USD = 10.0    # Minimum notional in USD (exchange minimum)

# =============================================================================
# Currency Conversion
# =============================================================================
USD_INR_RATE = 83.0             # Default USD/INR conversion rate (configurable in UI)

# =============================================================================
# IST Trading Sessions (Asia/Kolkata timezone)
# =============================================================================
IST_SESSIONS = [
    (dt_time(9, 30), dt_time(12, 0)),    # Morning session
    (dt_time(13, 0), dt_time(15, 30)),    # Afternoon session
    (dt_time(19, 0), dt_time(22, 0)),     # Evening session
]

TIMEZONE_IST = "Asia/Kolkata"

# =============================================================================
# Streamlit Dashboard Configuration
# =============================================================================
DASHBOARD_REFRESH_SECONDS = 60  # Auto-refresh interval
DASHBOARD_TITLE = "🔹 Bollinger Squeeze Breakout Trading Bot"
DASHBOARD_SUBTITLE = "BTC/USDT Perpetual Futures | SharkEx | 15m Chart"

# =============================================================================
# Retry & Rate Limiting
# =============================================================================
MAX_RETRIES = 3                 # Max retry attempts per API call
RETRY_BACKOFF_BASE = 1.0        # Base seconds for exponential backoff (1, 2, 4, ...)
RATE_LIMIT_COOLDOWN = 2.0       # Seconds to wait when rate-limited

# SharkEx rate limits (from docs):
# - place-order:    1s / 20 requests
# - delete-order:   1m / 30 requests
# - other endpoints: 1m / 60 requests
RATE_LIMITS = {
    "place_order":     {"window": 1.0,  "max_req": 20},
    "delete_order":    {"window": 60.0, "max_req": 30},
    "cancel_all":      {"window": 60.0, "max_req": 30},
    "close_all":       {"window": 60.0, "max_req": 30},
    "default":         {"window": 60.0, "max_req": 60},
}

# =============================================================================
# Order Types (as defined in SharkEx docs)
# =============================================================================
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_STOP_LIMIT = "STOP_LIMIT"

ORDER_SIDE_BUY = "BUY"
ORDER_SIDE_SELL = "SELL"

PLACE_TYPE_ORDER_FORM = "ORDER_FORM"
PLACE_TYPE_POSITION = "POSITION"

# =============================================================================
# File Paths
# =============================================================================
TRADE_LOG_FILE = "trade_log.json"       # Persistent trade log
STATE_FILE = "bot_state.json"           # Bot state persistence