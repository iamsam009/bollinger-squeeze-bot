# 🔹 Bollinger Squeeze Breakout Trading Bot

**BTC/USDT Perpetual Futures | SharkEx Exchange | 15-Minute Chart**

A real-time automated trading bot that detects Bollinger Bands squeeze breakouts on BTC/USDT and executes trades via the SharkEx API, all controlled through a Streamlit dashboard.

---

## 📊 Strategy Overview

| Parameter | Value |
|-----------|-------|
| Indicator | Bollinger Bands (period=20, std=2.0) |
| Squeeze Condition | BB width = minimum of last 10 closed candles |
| Long Entry | Close > highest high of last 10 candles (squeeze active) |
| Short Entry | Close < lowest low of last 10 candles (squeeze active) |
| Trailing Stop | 5-candle window, unidirectional (moves only favorably) |
| Entry Orders | LIMIT orders at bid/ask for maker fees |
| Stop Loss | STOP-MARKET with STOP-LIMIT fallback (0.1% offset) |

## ⚖️ Risk Management

| Limit | Value |
|-------|-------|
| Trade Size | ₹20,000 |
| Default Leverage | 10x (1-125x available) |
| Daily Loss Limit | ₹3,000 |
| Max Trades/Day | 30 |
| Reset | Midnight IST |

## 🕐 Trading Sessions (IST)

- **Morning:** 09:30 - 12:00
- **Afternoon:** 13:00 - 15:30  
- **Evening:** 19:00 - 22:00

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- SharkEx API key and secret

### Installation

```bash
cd TRADING_CLI
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Configuration

1. Enter your SharkEx **API Key** and **API Secret** in the sidebar
2. Adjust strategy parameters as needed (BB period, squeeze lookback, leverage, etc.)
3. Click **▶️ Start Bot** to begin automated trading
4. Use **🛑 CLOSE ALL** for emergency position closure

---

## 📁 Project Structure

```
TRADING_CLI/
├── app.py                  # Streamlit dashboard (main entry point)
├── config.py               # All configuration constants
├── sharkex_client.py       # SharkEx REST API client (HMAC-SHA256 auth)
├── strategy.py             # Bollinger Bands squeeze breakout + trailing stop
├── risk_manager.py         # IST sessions, loss limits, position sizing
├── state_manager.py        # Position tracking, P&L, trade log persistence
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🛡️ Security

- API keys are entered via the Streamlit sidebar and stored in `st.session_state` only during the session
- Never commit API keys to version control
- The `.gitignore` excludes `bot_state.json`, `trade_log.json`, and `bot.log`

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.