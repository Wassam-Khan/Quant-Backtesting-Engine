# Quantitative Backtesting Engine 🚀

A production-grade, event-driven backtesting engine built in Python. Designed to simulate algorithmic trading strategies with realistic execution modeling (slippage, commissions, and capital management).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## 📊 Key Features

* **Modular Architecture:** Separation of concerns between Data Ingestion, Strategy Logic, and Execution Engine.
* **Realistic Simulation:**
    * **Slippage & Commission Models:** accurately simulates transaction costs.
    * **Warm-up Buffers:** Handles indicator lookback periods (e.g., 200-day MA) to prevent data starvation at the start of simulations.
    * **Force-Close Logic:** mark-to-market valuation at the end of backtest periods.
* **Advanced Analytics:** Calculates Sharpe Ratio, Sortino Ratio, Max Drawdown, VaR (95%), and CVaR.
* **Type Safety:** Fully type-hinted codebase using Python `dataclasses` for configuration.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/quant-backtesting-engine.git](https://github.com/yourusername/quant-backtesting-engine.git)
    cd quant-backtesting-engine
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage

Run a backtest from the CLI. The engine automatically handles data fetching (yfinance) and caching.

```bash
python scripts/run_backtest.py --symbol AAPL --start 2023-01-01 --end 2023-12-31 --capital 100000