 # EasyA Consensus Hackathon Toronto (Aptos Track)

<div style="display: flex; justify-content: left; gap: 1em;">
  <img src="vizualizations/easya_consensus_hackathon.webp" alt="Dashboard 1" width="20%" height="20%"/>
  <img src="vizualizations/Aptos.png" alt="Dashboard 2" width="20%" height="20%"/>
</div>
<br>

 **Proof-of-concept** combining Python ML strategies with on-chain execution on Aptos.

 - **Python**: Technical indicators, ML models, backtesting (`predictor.py`, `Backtester`).
 - **Move Contracts**: Trading, risk management, signal executor (`move_contracts/sources`).
 - **On-Chain Bridge**: Python wrapper for Move calls (`onchain_backtest.py`).
 - **Deployment**: Automated compile, publish, tests (`deploy_contracts.py`).

 ## Table of Contents
 1. [Modules Overview](#modules-overview)
 2. [Project Description](#project-description)
 3. [Usage Examples](#usage-examples)
 4. [Visualization](#visualization)
 5. [How to Start](#how-to-start)

 ## Modules Overview

- **`predictor.py`**: Signal generation, feature flags (Fourier, PCA, rolling stats).
- **`backtester.py`**: Dedicated backtesting module. Contains `AptosBacktester` class, stock selection, signal generators, and all backtest-related utilities. (All backtesting logic previously in `aptos_integration_v3_1.py` is now here.)
- **`onchain_backtest.py`**: Async Python API for Move backtests (initialize, add data, run, results).
- **`deploy_contracts.py`**: Aptos CLI integration to compile, publish, fund, and test Move modules.
- **Move Contracts** in `move_contracts/sources`:
  - `minimal_trading.move`: Trade counter, `execute_trade_with_counterparty` for AptosCoin transfer.
  - `simple_risk_management.move`: On-chain risk limits, daily loss, emergency stop.
  - `simple_signal_executor.move`: Signal intake, confidence check, execution eventing.

## Project Description
This project builts based on [the repo from aptos-agent from aptos-labs](https://github.com/aptos-labs/aptos-agent). Thanks to the Aptos for the support and guidance during the hackathon.


**Aptos Agent Backtester:**
AI/ML-Powered Blockchain Trading Aptos Agent combines machine learning prediction with Aptos blockchain execution for automated trading. Features include **technical analysis signals**, on-chain transaction execution, **wallet management**, comprehensive **backtesting** (now modularized in `backtester.py`), real-time **portfolio tracking**, and risk management parameters for DeFi traders.

**Modular Backtesting Architecture:**
All backtesting logic, including the `AptosBacktester` class, signal generators, stock selection, and performance reporting, has been moved from `aptos_integration_v3_1.py` to a dedicated `backtester.py` module. This improves code organization, maintainability, and extensibility. Other modules now import backtesting functionality from `backtester.py`.


Here is the snapshot of the output. 
\
\
\
![](vizualizations/Figure_2.png)
### Figure 2: Single Stock Portfolio Analysis Dashboard 
Figure 2 shows performance for a single asset (QBTS).

<br> 
<br> 
<br> 

![](vizualizations/Figure_3.png)
### Figure 3: Multi-Stock Portfolio Analysis Dashboard

Backtest Report: Total Return: 42.11%, Sharpe: 1.11, Max Drawdown: -15.51%, Win Rate: 59.80%, Number of Trades: 1694, CAGR: 15.02%, Sortino: 1.88

<br> 
<br> 
<br> 

![](vizualizations/Figure_4.png)
### Figure 4: Multi-Stock Portfolio Analysis Dashboard

Figure **3** and **4** represents the multi-stock portfolio backtesting visualization with three comprehensive panels:



<br> 


1. **Portfolio Performance Panel (Top)**
- Blue line tracks the portfolio's normalized value
- **Green** triangles mark **buy** signals and **red** triangles indicate **sell** signals
- Orange **SPGI** benchmark line demonstrates our strategy's outperformance, especially during market corrections
- Portfolio maintains consistent growth trajectory with strategic entry/exit points

2. **Portfolio Composition Panel (Middle)**
* Stacked area chart visualizing asset allocation across multiple stocks and cash positions
* Dynamic allocation shows how the system actively rebalances based on market conditions
   

3. **Individual Symbol Performance Panel (Bottom)**
* Line chart comparing normalized performance of each stock in the portfolio
* Black line represents the average performance across all securities



[Here is the link of the presentation](https://www.canva.com/design/DAGnnzZRxhU/VGBKUeKuUp-0kf3HJntAHg/edit?utm_content=DAGnnzZRxhU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


## Technical Explanation

Our Aptos Agent project leverages several key technologies:

### Core Technologies
- **Python** for backend development and ML model & Strategy implementation
- **Scikit-Learn/TensorFlow/PyTorch** for training prediction models
- **Aptos Move** for smart contract development
- **React** for frontend dashboard interface

### Aptos Blockchain Technologies
We specifically leveraged Aptos's unique features:

1. **Move VM (potential)** - Used for secure, deterministic smart contract execution that enables our automated trading strategies with reliable transaction outcomes

2. **Parallel Execution Engine** - Aptos's BlockSTM allows high throughput transactions, critical for real-time trading decisions during market fluctuations

3. **Aptos SDK** - Integrated their Python SDK for seamless blockchain interaction from our prediction models

4. **Aptos Tokenization Framework** - For portfolio management and tracking digital assets

5. **Fast Finality** - Aptos's sub-second finality ensures our trading signals are executed promptly, reducing slippage in volatile markets

### On-Chain Integration Features
This project now includes full on-chain execution and risk management via Move smart contracts, integrated with Python:

- **Minimal Trading Module** (`minimal_trading.move`)
  - Simple counter resource tracks number of trades
  - `execute_trade_with_counterparty` entry function performs APTOS coin transfers to counterparties
- **Risk Management Module** (`simple_risk_management.move`)
  - On-chain `RiskManager` resource enforces per-trade position size, daily loss limits, and emergency stop
  - Includes `check_trade_risk`, `update_daily_loss`, and viewpoint functions (`get_risk_status`, `can_open_position`)
- **Signal Executor Module** (`simple_signal_executor.move`)
  - Receives Python-generated signals on-chain as events
  - Verifies confidence thresholds and emits execution events
  - Tracks execution results and allows batch processing

#### Python On-Chain Bridge
- **AptosOnChainBacktester** (`onchain_backtest.py`)
  - Wraps Move contract calls to initialize backtests, add price data, run steps, and retrieve results
- **Deployment & Testing Script** (`deploy_contracts.py`)
  - Automates `aptos move compile`, `publish`, and faucet funding
  - Runs local Move tests and on-chain transaction tests for all modules

With these additions, trading strategies developed in Python can be deployed, executed, and managed entirely on-chain for transparent, auditable execution.

<!-- What makes our project uniquely possible on Aptos is the combination of their high-performance blockchain (over 160,000 TPS) with sub-second finality, which is essential for algorithmic trading where execution speed matters. The Move language's resource-oriented programming also provides better security guarantees for managing user funds in a DeFi context compared to other blockchain environments. -->


### Trading Strategy Overview (Not included in the repo)

The multi-stock portfolio backtest implementation employs a sophisticated adaptive trading strategy with several key components:

#### Strategy Components

1. **Intelligent Stock Selection**
   * Uses fundamental and technical filters to identify promising candidates from major indices
   * Employs momentum, value, and quality factors to rank stocks based on the selected strategy 
   * Automatically adapts selection criteria based on broader market conditions

2. **Adaptive Position Sizing**
   * Dynamically adjusts position sizes based on volatility, trend strength, and technical indicators
   * Implements volatility-scaled position sizing (larger positions in lower volatility assets)
   * Increases exposure during identified bottoming patterns with volume confirmation
   * Reduces position sizes during high market uncertainty periods

3. **Multi-factor Entry Signals**
   * Combines technical indicators (RSI, Moving Average crossovers) with trend analysis
   * Identifies potential buy opportunities during oversold conditions (RSI < 30)
   * Incorporates rate-of-change and momentum factors for entry timing
   * Signal reversal capability that can switch between long and short bias based on market regime

4. **Risk Management Rules**
   * Implements dynamic trailing stops based on profit levels (tighter stops at higher profits)
   * Uses multi-tier take-profit levels (scaling out at 25%, 40% profit targets)
   * Hard stop-losses at predetermined maximum loss thresholds
   * Portfolio-level diversification rules to limit exposure to similar assets

## Setup Instructions

### Prerequisites
- Python 3.9+
- Aptos SDK
- Required Python libraries: pandas, numpy, matplotlib, scikit-learn, yfinance

### Installation
```bash
# Clone the repository
git clone https://github.com/Jamie1377/EasyA-Consensus-Hackathon---Toronto.git
cd aptos-agent-hackathon

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration