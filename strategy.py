"""
Bollinger Bands Squeeze Breakout Strategy
==========================================
Core strategy logic for detecting squeeze breakouts on 15-minute candles.

Squeeze Condition:
    BB width on the last closed candle is the MINIMUM of the last 10 closed candles.

Long Entry:
    Squeeze detected AND close > highest high of last 10 closed candles.

Short Entry:
    Squeeze detected AND close < lowest low of last 10 closed candles.

Trailing Stop:
    5-candle trailing window. For longs: max(close, trailing_stop).
    For shorts: min(close, trailing_stop). Moves only in favorable direction.
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from config import (
    BB_PERIOD,
    BB_STD_DEV,
    BB_SQUEEZE_LOOKBACK,
    TRAILING_STOP_WINDOW,
    STOP_LIMIT_OFFSET_PCT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class BBResult:
    """Bollinger Bands calculation result."""
    sma: float
    upper_band: float
    lower_band: float
    width: float           # BB width = upper - lower (or normalized: (upper-lower)/sma * 100)
    width_pct: float       # Normalized width as percentage
    is_squeeze: bool       # True if current width is min of last N periods
    squeeze_rank: int      # Position of current width in sorted lookback (1 = smallest)


@dataclass
class SignalResult:
    """Strategy signal result for a single evaluation."""
    has_signal: bool = False
    signal_type: Optional[str] = None         # "LONG" or "SHORT"
    entry_price: Optional[float] = None       # Suggested entry (bid for long, ask for short)
    stop_loss_price: Optional[float] = None   # Initial stop loss
    bb_result: Optional[BBResult] = None
    squeeze_detected: bool = False
    breakout_detected: bool = False
    breakout_reason: str = ""


@dataclass
class TrailingStopResult:
    """Trailing stop calculation result."""
    level: float           # Current trailing stop price level
    updated: bool          # Whether level changed from previous
    previous_level: float  # Previous level for logging


# =============================================================================
# Indicator Calculation
# =============================================================================
def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD_DEV,
    squeeze_lookback: int = BB_SQUEEZE_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and squeeze detection on the DataFrame.

    Expects columns: 'close', 'high', 'low' (optionally 'open', 'volume')
    Adds columns: 'bb_sma', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_pct',
                  'bb_squeeze', 'bb_squeeze_rank'

    Returns the modified DataFrame.
    """
    close = df["close"].astype(float)

    # Bollinger Bands
    bb_sma = close.rolling(window=period).mean()
    bb_std = close.rolling(window=period).std(ddof=0)  # population std for BB

    df["bb_sma"] = bb_sma
    df["bb_upper"] = bb_sma + std_dev * bb_std
    df["bb_lower"] = bb_sma - std_dev * bb_std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    # Normalized width as percentage of SMA
    df["bb_width_pct"] = (df["bb_width"] / df["bb_sma"]) * 100

    # Squeeze detection: min width of last N candles
    # We use rolling min on bb_width (excluding current, then compare)
    min_width_lookback = df["bb_width"].shift(1).rolling(window=squeeze_lookback).min()
    df["bb_squeeze"] = df["bb_width"] <= min_width_lookback

    # Squeeze rank (1 = tightest in window)
    def rolling_rank(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling rank (1 = smallest value in window)."""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            window_vals = series.iloc[i - window + 1 : i + 1]
            result.iloc[i] = (window_vals < series.iloc[i]).sum() + 1
        return result

    df["bb_squeeze_rank"] = rolling_rank(df["bb_width"], squeeze_lookback)

    return df


def detect_breakout(
    df: pd.DataFrame,
    squeeze_lookback: int = BB_SQUEEZE_LOOKBACK,
) -> pd.DataFrame:
    """
    Detect breakout signals on the DataFrame (must already have BB columns).

    Adds columns: 'highest_high_n', 'lowest_low_n', 'long_signal', 'short_signal'
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Highest high and lowest low of last N candles (excluding current)
    df["highest_high_n"] = high.shift(1).rolling(window=squeeze_lookback).max()
    df["lowest_low_n"] = low.shift(1).rolling(window=squeeze_lookback).min()

    # Long: squeeze on current closed candle AND close > highest high of last N
    df["long_signal"] = df["bb_squeeze"] & (close > df["highest_high_n"])

    # Short: squeeze on current closed candle AND close < lowest low of last N
    df["short_signal"] = df["bb_squeeze"] & (close < df["lowest_low_n"])

    return df


def evaluate_signal(
    df: pd.DataFrame,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> SignalResult:
    """
    Evaluate the latest candle for trading signals.

    Args:
        df: DataFrame with OHLCV and BB columns (calculate_bollinger_bands + detect_breakout called)
        bid: Current best bid price (for long entry at ask)
        ask: Current best ask price (for short entry at bid)

    Returns:
        SignalResult with signal details
    """
    if len(df) < BB_PERIOD + BB_SQUEEZE_LOOKBACK:
        return SignalResult()

    latest = df.iloc[-1]
    result = SignalResult()

    # Build BB result
    result.bb_result = BBResult(
        sma=float(latest["bb_sma"]),
        upper_band=float(latest["bb_upper"]),
        lower_band=float(latest["bb_lower"]),
        width=float(latest["bb_width"]),
        width_pct=float(latest["bb_width_pct"]),
        is_squeeze=bool(latest.get("bb_squeeze", False)),
        squeeze_rank=int(latest.get("bb_squeeze_rank", 0)),
    )

    result.squeeze_detected = result.bb_result.is_squeeze

    # Check for signals
    if bool(latest.get("long_signal", False)):
        result.has_signal = True
        result.signal_type = "LONG"
        result.entry_price = ask  # Buy at ask price (limit order)
        result.stop_loss_price = get_initial_stop(df, "LONG")
        result.breakout_detected = True
        result.breakout_reason = (
            f"Close {float(latest['close']):.2f} > "
            f"Highest High(N={BB_SQUEEZE_LOOKBACK}) {float(latest['highest_high_n']):.2f}"
        )

    elif bool(latest.get("short_signal", False)):
        result.has_signal = True
        result.signal_type = "SHORT"
        result.entry_price = bid  # Sell at bid price (limit order)
        result.stop_loss_price = get_initial_stop(df, "SHORT")
        result.breakout_detected = True
        result.breakout_reason = (
            f"Close {float(latest['close']):.2f} < "
            f"Lowest Low(N={BB_SQUEEZE_LOOKBACK}) {float(latest['lowest_low_n']):.2f}"
        )

    return result


def get_initial_stop(df: pd.DataFrame, direction: str) -> Optional[float]:
    """
    Calculate initial stop loss based on Bollinger Bands opposite band.

    For longs: BB lower band of the signal candle
    For shorts: BB upper band of the signal candle
    """
    if len(df) < BB_PERIOD:
        return None

    latest = df.iloc[-1]
    if direction == "LONG":
        return float(latest["bb_lower"])
    elif direction == "SHORT":
        return float(latest["bb_upper"])
    return None


# =============================================================================
# Trailing Stop Logic
# =============================================================================
def calculate_trailing_stop(
    df: pd.DataFrame,
    direction: str,
    current_level: Optional[float] = None,
    window: int = TRAILING_STOP_WINDOW,
) -> TrailingStopResult:
    """
    Calculate trailing stop level based on N-candle window.

    For LONG positions:
        Trailing stop = max close of last N candles
        Level only moves UP (never down)

    For SHORT positions:
        Trailing stop = min close of last N candles
        Level only moves DOWN (never up)

    Args:
        df: DataFrame with 'close' column, latest candles
        direction: "LONG" or "SHORT"
        current_level: Current trailing stop level (if tracking)
        window: Number of candles for trailing window

    Returns:
        TrailingStopResult with new level and update status
    """
    if len(df) < window:
        return TrailingStopResult(level=current_level or 0, updated=False, previous_level=current_level or 0)

    close_window = df["close"].astype(float).iloc[-window:]

    if direction == "LONG":
        raw_level = float(close_window.max())
        # Trail only UP
        if current_level is None or raw_level > current_level:
            return TrailingStopResult(
                level=raw_level,
                updated=(current_level is not None and raw_level > current_level),
                previous_level=current_level or raw_level,
            )
        else:
            return TrailingStopResult(
                level=current_level,
                updated=False,
                previous_level=current_level,
            )
    elif direction == "SHORT":
        raw_level = float(close_window.min())
        # Trail only DOWN
        if current_level is None or raw_level < current_level:
            return TrailingStopResult(
                level=raw_level,
                updated=(current_level is not None and raw_level < current_level),
                previous_level=current_level or raw_level,
            )
        else:
            return TrailingStopResult(
                level=current_level,
                updated=False,
                previous_level=current_level,
            )
    else:
        raise ValueError(f"Invalid direction: {direction}")


def is_trailing_stop_hit(
    current_price: float,
    trailing_stop_level: float,
    direction: str,
) -> bool:
    """
    Check if trailing stop has been hit.

    For LONG: price <= trailing_stop_level
    For SHORT: price >= trailing_stop_level
    """
    if direction == "LONG":
        return current_price <= trailing_stop_level
    elif direction == "SHORT":
        return current_price >= trailing_stop_level
    return False


def get_stop_limit_price(
    stop_price: float,
    direction: str,
    offset_pct: float = STOP_LIMIT_OFFSET_PCT,
) -> float:
    """
    Calculate stop-limit order price (0.1% beyond stop price as fallback).

    For LONG exit (sell): limit price = stop_price * (1 - offset_pct)  (lower)
    For SHORT exit (buy): limit price = stop_price * (1 + offset_pct)  (higher)
    """
    if direction == "LONG":
        return stop_price * (1.0 - offset_pct)
    elif direction == "SHORT":
        return stop_price * (1.0 + offset_pct)
    return stop_price


# =============================================================================
# Utility: Check if squeeze is in progress (not just final candle)
# =============================================================================
def get_squeeze_info(df: pd.DataFrame) -> Dict:
    """
    Get detailed squeeze information for the dashboard.
    """
    if len(df) < BB_PERIOD + BB_SQUEEZE_LOOKBACK:
        return {"in_squeeze": False, "message": "Not enough data"}

    latest = df.iloc[-1]
    return {
        "in_squeeze": bool(latest.get("bb_squeeze", False)),
        "width_pct": float(latest.get("bb_width_pct", 0)),
        "sma": float(latest.get("bb_sma", 0)),
        "upper_band": float(latest.get("bb_upper", 0)),
        "lower_band": float(latest.get("bb_lower", 0)),
        "squeeze_rank": int(latest.get("bb_squeeze_rank", 0)),
        "highest_high_n": float(latest.get("highest_high_n", 0)),
        "lowest_low_n": float(latest.get("lowest_low_n", 0)),
        "close": float(latest["close"]),
    }


# =============================================================================
# Prepare DataFrame for Strategy
# =============================================================================
def klines_to_dataframe(klines: List[Dict]) -> pd.DataFrame:
    """
    Convert SharkEx kline response to a pandas DataFrame with required columns.

    Expected kline format from SharkEx (per docs):
    [
      {"startTime": "1726312200000", "open": "5270382.19", "high": "5270408.64",
       "low": "5270382.19", "close": "5270382.19", "endTime": "1726312259999", "volume": "59.102"},
      ...
    ]
    OR list of dicts with keys: openTime/startTime, open, high, low, close, volume, closeTime/endTime
    """
    if not klines:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    records = []
    for k in klines:
        if isinstance(k, list):
            records.append({
                "open_time": k[0] if len(k) > 0 else 0,
                "open": float(k[1]) if len(k) > 1 else 0.0,
                "high": float(k[2]) if len(k) > 2 else 0.0,
                "low": float(k[3]) if len(k) > 3 else 0.0,
                "close": float(k[4]) if len(k) > 4 else 0.0,
                "volume": float(k[5]) if len(k) > 5 else 0.0,
                "close_time": k[6] if len(k) > 6 else 0,
            })
        elif isinstance(k, dict):
            # SharkEx docs use "startTime"/"endTime"; also support "openTime"/"closeTime"
            records.append({
                "open_time": k.get("startTime", k.get("openTime", k.get("open_time", 0))),
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": float(k.get("volume", 0)),
                "close_time": k.get("endTime", k.get("closeTime", k.get("close_time", 0))),
            })

    df = pd.DataFrame(records)
    # Set datetime index
    if "open_time" in df.columns and len(df) > 0:
        # SharkEx returns startTime/endTime as strings; convert to int for comparison
        ot_val = float(df["open_time"].iloc[-1])
        if ot_val > 1000000000000:
            df["datetime"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
        else:
            df["datetime"] = pd.to_datetime(df["open_time"].astype(float), unit="s")
    else:
        df["datetime"] = pd.NaT

    df.set_index("datetime", inplace=True, drop=False)

    return df


def prepare_strategy_df(klines: List[Dict]) -> pd.DataFrame:
    """
    Fetch klines, convert to DataFrame, and compute all indicators.

    Returns DataFrame ready for signal evaluation.
    """
    df = klines_to_dataframe(klines)
    if df.empty or len(df) < BB_PERIOD + BB_SQUEEZE_LOOKBACK:
        logger.warning(f"Insufficient data: {len(df)} candles, need {BB_PERIOD + BB_SQUEEZE_LOOKBACK}")
        return df

    df = calculate_bollinger_bands(df)
    df = detect_breakout(df)

    return df