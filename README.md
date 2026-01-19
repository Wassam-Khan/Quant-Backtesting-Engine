# Quantitative Backtesting Engine 🚀

A professional-grade, event-driven backtesting engine built in Python. This framework is designed to simulate algorithmic trading strategies with high-fidelity execution modeling, including realistic slippage, commissions, and automated data "warm-up" handling.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## 📊 Key Engineering Features

* **Modular Architecture:** Strict separation of concerns between Data Ingestion, Strategy Logic, Execution Simulation, and Analytics.
* **Realistic Execution Modeling:**
    * **Transaction Costs:** Models fixed and percentage-based commissions.
    * **Slippage Simulation:** Accounts for market impact by adjusting execution prices based on trade direction.
    * **Warm-up Buffers:** Automatically fetches historical data (e.g., 365 days) prior to the start date to ensure indicators like the 200-day SMA are valid from Day 1.
* **Advanced Quantitative Analytics:**
    * Calculates institutional-grade metrics: **Sharpe Ratio**, **Sortino Ratio**, and **Calmar Ratio**.
    * Risk assessment via **Value at Risk (VaR 95%)** and **Conditional VaR (CVaR)**.
    * Position sizing optimization using the **Kelly Criterion**.
* **Intelligent Caching:** Local CSV-based caching system for market data to minimize API rate-limiting and improve speed.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Wassam-Khan/Quant-Backtesting-Engine.git](https://github.com/Wassam-Khan/Quant-Backtesting-Engine.git)
    cd Quant-Backtesting-Engine
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage

Run a backtest from the CLI. The engine handles all data fetching (yfinance/CCXT) and signal generation automatically.

```bash
python scripts/run_backtest.py --symbol AAPL --start 2023-01-01 --end 2023-12-31 --capital 100000
```

## 📊  Result
<img width="1110" height="847" alt="Screenshot 2026-01-20 015718" src="https://github.com/user-attachments/assets/26c179c5-2ebc-4c56-9b43-69dc283c391b" />
