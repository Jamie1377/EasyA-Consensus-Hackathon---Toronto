"""
Backtest Statistics and Benchmarking Module

This module provides utilities for running statistical tests on backtesting strategies,
evaluating performance across multiple symbols, and analyzing which indicators contribute
most to trading success.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Any, Optional, Union

# Import necessary components
from predictor import StockPredictor
from backtester import (
    AptosBacktester,
    select_stocks,
    create_signal_generator,
    create_multi_stock_signal_generator,
)

    
import yfinance as yf
tech_sec = list(yf.Sector("technology").top_companies.index) # Example usage to ensure yfinance is imported correctly
import random

# Configure logging
log_directory = "."
log_file = f"backtest_stats_{date.today().strftime('%Y%m%d')}.log"

# Create logger
logger = logging.getLogger("backtest_stats")
logger.setLevel(logging.INFO)

# Ensure we only add handlers once
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

try:
    from private_strat import get_entry_signal # Import custom signal generator if available
except ImportError:
    logger.warning("Custom signal generator not found!")
    


async def run_multi_stock_backtest(indicators_to_drop=None):
    """
    Run a backtest on multiple stocks with specified configuration

    Args:
        indicators_to_drop: Optional pair of indicators to exclude from the strategy

    Returns:
        tuple: (backtester, history_df, metrics) - Results of the backtest
    """
    # Select stocks for backtest
    selected_stocks = select_stocks(
        num_stocks=5,
        universe="NASDAQ 100",
        lookback_days=360,
        strategy="momentum",
        interval="1d",
        end_date=None,# Default to today
    )

    # Create backtester with multiple stocks
    backtester = AptosBacktester(symbols=selected_stocks, initial_capital=100000)

    # Define backtest parameters
    start_date = (date.today() - timedelta(days=500)).strftime("%Y-%m-%d")
    end_date = date.today().strftime("%Y-%m-%d")

    # Use the predictor-based signal generator
    from predictor import StockPredictor

    # Run backtest with our multi-stock signal generator
    history_df, metrics = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        signal_generator=create_multi_stock_signal_generator(
            predictor_class=StockPredictor,
            always_reverse=False,
            autodetect_reversal=True,
            indicators_to_drop=indicators_to_drop,
        ),
    )

    # Display results
    logger.info("\n=== Multi-Stock Backtest Results ===")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Stocks: {', '.join(selected_stocks)}")
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Number of Trades: {metrics.get('num_trades', 0)}")

    if indicators_to_drop:
        logger.info(f"Indicators dropped: {indicators_to_drop}")

    # Plot results
    backtester.plot_results(history_df)

    return backtester, history_df, metrics


def counter_of_win_over_mkt(num_trials=20):
    """
    Run multiple backtest trials to count how often the strategy beats the market

    Args:
        num_trials: Number of backtest trials to run

    Returns:
        dict: Statistics about market outperformance
    """
    beat_market = 0
    beat_risk_free = 0
    total_return = 0

    logger.info(
        f"Running {num_trials} backtest trials to measure market outperformance..."
    )

    for i in range(num_trials):
        logger.info(f"Trial {i+1}/{num_trials}")

        # Select random stocks for this trial
        symbols = random.choices(
            tech_sec, k=5
        )

        # Create backtester instance
        backtester = AptosBacktester(symbols=symbols, initial_capital=100000)

        # Define time period - using a randomized lookback between 6-18 months

        lookback_days = random.randint(500, 750)
        end_date = date.today() - timedelta(days=random.randint(180, 540))
        start_date = (end_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        try:
            # Run backtest using multi-stock signal generator
            history, metrics = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                signal_generator=create_multi_stock_signal_generator(
                    predictor_class=StockPredictor,
                    always_reverse=False,
                    autodetect_reversal=False,
                ),
            )

            # Plot results for visual inspection
            backtester.plot_results(history)

            # Count wins against benchmarks
            if backtester.whether_beat_market():
                beat_market += 1

            if backtester.whether_beat_risk_free(metrics):
                beat_risk_free += 1

            total_return += metrics["total_return"]

            logger.info(
                f"Trial {i+1}, "
                f"Return={metrics['total_return']:.2%}, "
                f"Beat Market={backtester.whether_beat_market()}, "
                f"Beat Risk-Free={backtester.whether_beat_risk_free(metrics)}"
            )

        except Exception as e:
            logger.error(f"Error in trial {i+1}: {e}")

    # Calculate statistics
    market_win_rate = beat_market / num_trials if num_trials > 0 else 0
    risk_free_win_rate = beat_risk_free / num_trials if num_trials > 0 else 0
    avg_return = total_return / num_trials if num_trials > 0 else 0

    results = {
        "market_win_rate": market_win_rate,
        "risk_free_win_rate": risk_free_win_rate,
        "avg_return": avg_return,
        "num_trials": num_trials,
        "beat_market_count": beat_market,
        "beat_risk_free_count": beat_risk_free,
    }

    logger.info("\n=== Market Outperformance Results ===")
    logger.info(
        f"Win Rate vs Market: {market_win_rate:.2%} ({beat_market}/{num_trials})"
    )
    logger.info(
        f"Win Rate vs Risk-Free: {risk_free_win_rate:.2%} ({beat_risk_free}/{num_trials})"
    )
    logger.info(f"Average Return: {avg_return:.2%}")

    return results


async def analyze_indicators():
    """
    Analyze which indicators contribute most to trading success
    by running tests with different indicators dropped
    """
    # First get the full list of indicators
    predictor = StockPredictor(
        symbol="AAPL", start_date="2020-01-01", end_date="2025-05-17"
    )
    predictor.load_data()
    _, _, _, info = get_entry_signal(predictor)

    buy_indicators = info["buy_signal_indicators"]
    sell_indicators = info["sell_signal_indicators"]

    logger.info(f"Buy indicators are {buy_indicators}")
    logger.info(f"Sell indicators are {sell_indicators}")

    # Baseline test with all indicators
    logger.info("Running baseline test with all indicators...")
    _, _, baseline_metrics = await run_multi_stock_backtest()
    baseline_return = baseline_metrics["total_return"]
    logger.info(f"Baseline return: {baseline_return:.2%}")

    # Test dropping pairs of indicators
    results = []
    for i, indicator_pair in enumerate(zip(buy_indicators, sell_indicators)):
        logger.info(f"Testing with indicators dropped: {indicator_pair}")
        try:
            _, _, test_metrics = await run_multi_stock_backtest(indicator_pair)
            test_return = test_metrics["total_return"]
            impact = baseline_return - test_return

            results.append(
                {
                    "indicator_pair": indicator_pair,
                    "return": test_return,
                    "impact": impact,
                    "sharpe": test_metrics["sharpe"],
                    "win_rate": test_metrics["win_rate"],
                }
            )

            logger.info(f"Result: Return={test_return:.2%}, Impact={impact:.2%}")
        except Exception as e:
            logger.error(f"Error testing {indicator_pair}: {e}")

    # Sort results by impact
    results.sort(key=lambda x: x["impact"], reverse=True)

    # Display and visualize results
    logger.info("\n=== Indicator Impact Analysis ===")
    logger.info(f"Baseline return with all indicators: {baseline_return:.2%}")
    logger.info("\nIndicator pairs ranked by impact:")

    for i, result in enumerate(results):
        logger.info(
            f"{i+1}. {result['indicator_pair']} - "
            f"Impact: {result['impact']:.2%}, Return: {result['return']:.2%}, "
            f"Sharpe: {result['sharpe']:.2f}, Win Rate: {result['win_rate']:.2%}"
        )

    # Create visualization
    plt.figure(figsize=(10, 6))
    indicators = [
        f"{pair[0]}/{pair[1]}" for pair in [r["indicator_pair"] for r in results]
    ]
    impacts = [r["impact"] * 100 for r in results]  # Convert to percentage

    plt.barh(indicators, impacts)
    plt.xlabel("Impact on Return (%)")
    plt.ylabel("Indicator Pair")
    plt.title("Impact of Removing Indicator Pairs on Strategy Performance")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # Uncomment the tests you want to run

    # Test 1: Run the main backtest
    # asyncio.run(main())

    # Test 2: Run the multi-stock backtest
    # asyncio.run(run_multi_stock_backtest())

    # Test 3: Measure how often strategy beats the market
    # counter_of_win_over_mkt(5)

    # Test 4: Analyze indicators
    asyncio.run(analyze_indicators())

    print(
        "Select a test to run by uncommenting the relevant section in backtest_stats.py"
    )
