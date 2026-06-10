"""
SharkEx API Client
==================
HMAC-SHA256 authenticated client for SharkEx REST API.
All endpoints and authentication methods derived from sharkex_docs.html.

Auth Flow:
- GET:  signature = HMAC-SHA256(query_string, api_secret)
- POST/PATCH/PUT/DELETE: signature = HMAC-SHA256(json.dumps(body), api_secret)
- Headers: {"api-key": api_key, "signature": signature}
- Market data endpoints are public (no auth headers needed)
"""

import hashlib
import hmac
import json
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from config import (
    SHARKEX_BASE_URL,
    SYMBOL,
    KLINE_INTERVAL,
    KLINE_LIMIT,
    MARGIN_ASSET,
    CONTRACT_TYPE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RATE_LIMIT_COOLDOWN,
    RATE_LIMITS,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_STOP_MARKET,
    ORDER_TYPE_STOP_LIMIT,
    ORDER_SIDE_BUY,
    ORDER_SIDE_SELL,
    PLACE_TYPE_ORDER_FORM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiter (Token Bucket Style)
# =============================================================================
class RateLimiter:
    """Simple rate limiter tracking request timestamps per endpoint category."""

    def __init__(self):
        self._timestamps: Dict[str, List[float]] = {}

    def _get_limit(self, category: str) -> Tuple[float, int]:
        """Return (window_seconds, max_requests) for a category."""
        rule = RATE_LIMITS.get(category, RATE_LIMITS["default"])
        return rule["window"], rule["max_req"]

    def wait_if_needed(self, category: str = "default") -> None:
        """Block until it's safe to send the next request."""
        window, max_req = self._get_limit(category)
        now = time.time()
        if category not in self._timestamps:
            self._timestamps[category] = []
        # Purge timestamps outside the window
        self._timestamps[category] = [
            t for t in self._timestamps[category] if now - t < window
        ]
        if len(self._timestamps[category]) >= max_req:
            oldest = self._timestamps[category][0]
            wait = window - (now - oldest) + 0.1  # small buffer
            if wait > 0:
                logger.debug(f"Rate limit: waiting {wait:.2f}s for category '{category}'")
                time.sleep(wait)
                # Recalculate after waiting
                now = time.time()
                self._timestamps[category] = [
                    t for t in self._timestamps[category] if now - t < window
                ]

    def record(self, category: str = "default") -> None:
        """Record a request timestamp."""
        if category not in self._timestamps:
            self._timestamps[category] = []
        self._timestamps[category].append(time.time())


# Global rate limiter instance
_rate_limiter = RateLimiter()


# =============================================================================
# Signature Generation
# =============================================================================
def _generate_signature(api_secret: str, payload: str) -> str:
    """
    Generate HMAC-SHA256 signature.
    
    - For GET requests: payload = query_string (e.g., "symbol=BTCUSDT&limit=10")
    - For POST/PATCH/PUT/DELETE: payload = JSON.stringify(body)
    """
    return hmac.new(
        api_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# =============================================================================
# HTTP Request Wrapper with Retry Logic
# =============================================================================
def _make_request(
    method: str,
    endpoint: str,
    api_key: str = "",
    api_secret: str = "",
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    is_public: bool = False,
    rate_limit_category: str = "default",
) -> Dict[str, Any]:
    """
    Core HTTP request wrapper with auth, rate limiting, and retry logic.

    Args:
        method: HTTP method (GET, POST, PATCH, DELETE)
        endpoint: API endpoint path (e.g., "/v1/market/klines")
        api_key: SharkEx API key
        api_secret: SharkEx API secret
        params: URL query parameters
        json_body: JSON body for POST/PATCH requests
        is_public: If True, skip authentication headers
        rate_limit_category: Category for rate limiting

    Returns:
        Parsed JSON response as dict

    Raises:
        requests.RequestException: After exhausting all retries
    """
    url = f"{SHARKEX_BASE_URL}{endpoint}"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Build query string for both URL and signature
    query_string = ""
    if params:
        # Remove None values
        clean_params = {k: v for k, v in params.items() if v is not None}
        query_string = urlencode(clean_params)
        url = f"{url}?{query_string}"

    # Generate signature for authenticated endpoints
    if not is_public:
        if not api_key or not api_secret:
            raise ValueError("API key and secret are required for authenticated endpoints")

        if method.upper() == "GET":
            # GET: sign the query string
            signature_payload = query_string
        else:
            # POST/PATCH/DELETE: sign the JSON body
            signature_payload = json.dumps(json_body) if json_body else "{}"

        signature = _generate_signature(api_secret, signature_payload)
        headers["api-key"] = api_key
        headers["signature"] = signature

    # Retry loop with exponential backoff
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Rate limit check
            _rate_limiter.wait_if_needed(rate_limit_category)

            # Execute request
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                resp = requests.post(url, headers=headers, json=json_body, timeout=30)
            elif method.upper() == "PATCH":
                resp = requests.patch(url, headers=headers, json=json_body, timeout=30)
            elif method.upper() == "DELETE":
                resp = requests.delete(url, headers=headers, json=json_body, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            _rate_limiter.record(rate_limit_category)

            # Handle rate limiting (HTTP 429)
            if resp.status_code == 429:
                logger.warning(f"Rate limited (429). Cooling down {RATE_LIMIT_COOLDOWN}s...")
                time.sleep(RATE_LIMIT_COOLDOWN)
                continue

            # Handle server errors
            if resp.status_code >= 500:
                logger.warning(f"Server error {resp.status_code}. Attempt {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue

            # Parse response
            try:
                data = resp.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response: {resp.text[:500]}")
                raise requests.RequestException(f"Invalid JSON: {resp.text[:200]}")

            # Check for API-level errors
            if isinstance(data, dict) and data.get("code") and data.get("code") != 200:
                error_msg = data.get("message", data.get("msg", "Unknown error"))
                logger.error(f"API error [code={data['code']}]: {error_msg}")
                raise requests.RequestException(f"SharkEx API error: {error_msg}")

            return data

        except requests.Timeout:
            logger.warning(f"Request timeout. Attempt {attempt}/{MAX_RETRIES}")
            last_exception = requests.Timeout("Request timed out")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

        except requests.ConnectionError as e:
            logger.warning(f"Connection error: {e}. Attempt {attempt}/{MAX_RETRIES}")
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

        except requests.RequestException as e:
            # Don't retry on client errors (4xx except 429)
            if hasattr(e, 'response') and e.response is not None and 400 <= e.response.status_code < 500:
                raise
            logger.warning(f"Request error: {e}. Attempt {attempt}/{MAX_RETRIES}")
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

    raise (last_exception or requests.RequestException("Max retries exceeded"))


# =============================================================================
# Public Market Data Endpoints (No Auth Required)
# =============================================================================
def fetch_klines(
    symbol: str = SYMBOL,
    interval: str = KLINE_INTERVAL,
    limit: int = KLINE_LIMIT,
    price_type: str = "MARK_PRICE",
) -> List[Dict[str, Any]]:
    """
    Fetch candlestick/kline data.
    
    From docs: POST /v1/market/klines?priceType=MARK_PRICE
    (Despite docs showing POST, this is a data-fetch operation)
    
    Returns list of candle dicts with: openTime, open, high, low, close, volume, closeTime, ...
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "priceType": price_type,
    }
    data = _make_request(
        method="POST",
        endpoint="/v1/market/klines",
        params=params,
        is_public=True,
        rate_limit_category="default",
    )
    # Response may be a list or wrapped in a dict
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common response wrappers
        for key in ("data", "result", "klines"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def fetch_ticker(symbol: str = SYMBOL) -> Dict[str, Any]:
    """
    Fetch 24hr ticker data.
    GET /v1/market/ticker
    """
    params = {"symbol": symbol}
    return _make_request(
        method="GET",
        endpoint="/v1/market/ticker",
        params=params,
        is_public=True,
        rate_limit_category="default",
    )


def fetch_depth(symbol: str = SYMBOL, limit: int = 100) -> Dict[str, Any]:
    """
    Fetch order book depth.
    GET /v1/market/depth
    """
    params = {"symbol": symbol, "limit": limit}
    return _make_request(
        method="GET",
        endpoint="/v1/market/depth",
        params=params,
        is_public=True,
        rate_limit_category="default",
    )


def fetch_exchange_info() -> Dict[str, Any]:
    """
    Fetch exchange info (symbols, filters, contract specs).
    GET /v1/exchange/exchange-info
    """
    # Public endpoint, but docs show no auth - let's treat as public
    return _make_request(
        method="GET",
        endpoint="/v1/exchange/exchange-info",
        is_public=True,
        rate_limit_category="default",
    )


# =============================================================================
# Private / Authenticated Endpoints
# =============================================================================
def fetch_futures_wallet(api_key: str, api_secret: str) -> Dict[str, Any]:
    """
    Get futures wallet details.
    GET /v1/wallet/futures-wallet/details
    """
    return _make_request(
        method="GET",
        endpoint="/v1/wallet/futures-wallet/details",
        api_key=api_key,
        api_secret=api_secret,
        rate_limit_category="default",
    )


def fetch_positions(api_key: str, api_secret: str, symbol: str = SYMBOL) -> List[Dict[str, Any]]:
    """
    Fetch open positions.
    GET /v1/positions
    """
    params = {"symbol": symbol}
    data = _make_request(
        method="GET",
        endpoint="/v1/positions",
        params=params,
        api_key=api_key,
        api_secret=api_secret,
        rate_limit_category="default",
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "positions", "result"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def fetch_open_orders(api_key: str, api_secret: str, symbol: str = SYMBOL) -> List[Dict[str, Any]]:
    """
    Fetch open orders.
    GET /v1/order/open-orders
    """
    params = {"symbol": symbol}
    data = _make_request(
        method="GET",
        endpoint="/v1/order/open-orders",
        params=params,
        api_key=api_key,
        api_secret=api_secret,
        rate_limit_category="default",
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "orders", "result"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def update_leverage(
    api_key: str,
    api_secret: str,
    symbol: str = SYMBOL,
    leverage: int = 10,
) -> Dict[str, Any]:
    """
    Update leverage for a symbol.
    POST /v1/exchange/update/leverage
    
    Docs note: Leverage must be set before placing orders.
    Valid range: 1-125x
    """
    json_body = {
        "symbol": symbol,
        "leverage": leverage,
        "marginAsset": MARGIN_ASSET,
    }
    return _make_request(
        method="POST",
        endpoint="/v1/exchange/update/leverage",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="default",
    )


def place_order(
    api_key: str,
    api_secret: str,
    symbol: str = SYMBOL,
    side: str = ORDER_SIDE_BUY,
    order_type: str = ORDER_TYPE_LIMIT,
    quantity: float = 0.0,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    reduce_only: bool = False,
    client_order_id: Optional[str] = None,
    take_profit_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Place an order on SharkEx.

    POST /v1/order/place-order

    Args:
        order_type: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
        side: BUY or SELL
        quantity: Order quantity in contracts
        price: Required for LIMIT and STOP_LIMIT orders
        stop_price: Required for STOP_MARKET and STOP_LIMIT orders
        reduce_only: If True, order only reduces position
        client_order_id: Custom order ID for tracking
        take_profit_price: Take profit trigger price
        stop_loss_price: Stop loss trigger price
    """
    json_body: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "marginAsset": MARGIN_ASSET,
        "deviceType": "API",
        "userCategory": "EXTERNAL",
        "placeType": PLACE_TYPE_ORDER_FORM,
        "reduceOnly": reduce_only,
    }

    if price is not None:
        json_body["price"] = price
    if stop_price is not None:
        json_body["stopPrice"] = stop_price
    if client_order_id is not None:
        json_body["clientOrderId"] = client_order_id
    if take_profit_price is not None:
        json_body["takeProfitPrice"] = take_profit_price
    if stop_loss_price is not None:
        json_body["stopLossPrice"] = stop_loss_price

    return _make_request(
        method="POST",
        endpoint="/v1/order/place-order",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="place_order",
    )


def edit_order(
    api_key: str,
    api_secret: str,
    client_order_id: str,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Edit an existing order.
    PATCH /v1/order/edit-order
    """
    json_body: Dict[str, Any] = {"clientOrderId": client_order_id}
    if quantity is not None:
        json_body["quantity"] = quantity
    if price is not None:
        json_body["price"] = price
    if stop_price is not None:
        json_body["stopPrice"] = stop_price

    return _make_request(
        method="PATCH",
        endpoint="/v1/order/edit-order",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="default",
    )


def delete_order(
    api_key: str,
    api_secret: str,
    client_order_id: str,
) -> Dict[str, Any]:
    """
    Delete/cancel a specific order.
    DELETE /v1/order/delete-order
    """
    json_body = {"clientOrderId": client_order_id}
    return _make_request(
        method="DELETE",
        endpoint="/v1/order/delete-order",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="delete_order",
    )


def cancel_all_orders(
    api_key: str,
    api_secret: str,
    symbol: str = SYMBOL,
) -> Dict[str, Any]:
    """
    Cancel all open orders for a symbol.
    DELETE /v1/order/cancel-all-orders
    """
    json_body = {"symbol": symbol}
    return _make_request(
        method="DELETE",
        endpoint="/v1/order/cancel-all-orders",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="cancel_all",
    )


def close_all_positions(
    api_key: str,
    api_secret: str,
    symbol: str = SYMBOL,
) -> Dict[str, Any]:
    """
    Close all positions for a symbol (emergency exit).
    DELETE /v1/positions/close-all-positions
    """
    json_body = {"symbol": symbol}
    return _make_request(
        method="DELETE",
        endpoint="/v1/positions/close-all-positions",
        api_key=api_key,
        api_secret=api_secret,
        json_body=json_body,
        rate_limit_category="close_all",
    )


def fetch_order_history(
    api_key: str,
    api_secret: str,
    symbol: str = SYMBOL,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch order history.
    GET /v1/order/order-history
    """
    params = {"symbol": symbol, "limit": limit}
    data = _make_request(
        method="GET",
        endpoint="/v1/order/order-history",
        params=params,
        api_key=api_key,
        api_secret=api_secret,
        rate_limit_category="default",
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "orders", "result"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# =============================================================================
# Composite / Helper Functions
# =============================================================================
def get_bid_ask(symbol: str = SYMBOL) -> Tuple[Optional[float], Optional[float]]:
    """
    Get current best bid and ask from order book.
    Returns (bid, ask) tuple.
    """
    try:
        depth = fetch_depth(symbol, limit=5)
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        bid = float(bids[0][0]) if bids else None
        ask = float(asks[0][0]) if asks else None
        return bid, ask
    except Exception:
        # Fallback: use ticker
        try:
            ticker = fetch_ticker(symbol)
            bid = float(ticker.get("bidPrice", 0)) or None
            ask = float(ticker.get("askPrice", 0)) or None
            return bid, ask
        except Exception:
            return None, None


def get_current_price(symbol: str = SYMBOL) -> Optional[float]:
    """
    Get current mark price or last price.
    Tries ticker first, then klines.
    """
    try:
        ticker = fetch_ticker(symbol)
        # Try mark price first (more relevant for futures)
        price = ticker.get("markPrice") or ticker.get("lastPrice") or ticker.get("last")
        if price:
            return float(price)
    except Exception:
        pass

    try:
        klines = fetch_klines(symbol, KLINE_INTERVAL, 1)
        if klines:
            return float(klines[-1].get("close", 0))
    except Exception:
        pass

    return None


def get_available_balance(api_key: str, api_secret: str) -> Optional[float]:
    """
    Get available balance in USDT from futures wallet.
    """
    try:
        wallet = fetch_futures_wallet(api_key, api_secret)
        # Response structure: may contain 'availableBalance' or similar
        if isinstance(wallet, dict):
            return float(
                wallet.get("availableBalance")
                or wallet.get("availableMargin")
                or wallet.get("balance")
                or 0
            )
    except Exception:
        pass
    return None


def fetch_usd_inr_rate() -> Optional[float]:
    """
    Fetch live USD/INR forex rate from free public APIs.
    Falls back through multiple providers for reliability.
    Returns None if all providers fail.
    """
    # Try exchangerate-api first (no key needed)
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = resp.json()
        rate = data.get("rates", {}).get("INR")
        if rate:
            logger.info(f"USD/INR rate from exchangerate-api: {rate}")
            return float(rate)
    except Exception as e:
        logger.warning(f"exchangerate-api failed: {e}")

    # Fallback: frankfurter.app
    try:
        resp = requests.get("https://api.frankfurter.app/latest?from=USD&to=INR", timeout=5)
        data = resp.json()
        rate = data.get("rates", {}).get("INR")
        if rate:
            logger.info(f"USD/INR rate from frankfurter.app: {rate}")
            return float(rate)
    except Exception as e:
        logger.warning(f"frankfurter.app failed: {e}")

    return None