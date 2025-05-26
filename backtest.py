# Check entry points for trading
from predictor import StockPredictor
from aptos_integration_v3_1 import AptosBacktester, create_signal_generator
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import logging
from pytickersymbols import PyTickerSymbols
stock_data = PyTickerSymbols()
nasdaq_tickers = stock_data.get_stocks_by_index('NASDAQ 100')  # Corrected index name
sp500_tickers = stock_data.get_stocks_by_index('S&P 500')
sp500_tickers = [ticker['symbol'] for ticker in nasdaq_tickers]  # Remove duplicates
# sp500_tickers = [ticker['symbol'] for ticker in sp500_tickers ]
# sp500_tickers = list(yf.Sector('financial-services').top_companies.index)


def find_correlated_stocks(ticker, lookback='1y', limit=10):
    """Find stocks with similar price movements (high correlation)"""
    try:
        # Get price data for target stock - ensure it's a Series
        target_data = yf.download(ticker, period=lookback, interval='1d', progress=False)['Close']
        
        # Convert DataFrame with single column to Series if needed
        if isinstance(target_data, pd.DataFrame):
            target_data = target_data.squeeze()
        
        # Sample a subset for efficiency (adjust as needed)
        sample_size = min(100, len(sp500_tickers))
        sample_tickers = np.random.choice(sp500_tickers, sample_size, replace=False).tolist()
        
        if ticker in sample_tickers:
            sample_tickers.remove(ticker)
        
        # Download data for all sampled tickers - only retrieve if list is not empty
        if not sample_tickers:
            print("No tickers to compare with. Check your sp500_tickers list.")
            return pd.DataFrame(columns=['Symbol', 'Correlation'])
            
        print(f"Downloading price data for {len(sample_tickers)} stocks...")
        all_data = yf.download(sample_tickers, period=lookback, interval='1d', progress=False)['Close']
        
        # Handle case where only one ticker is returned
        if isinstance(all_data, pd.Series):
            all_data = pd.DataFrame({sample_tickers[0]: all_data})
        
        # Calculate correlation with target stock
        correlations = {}
        for col in all_data.columns:
            # Make sure we have data for both stocks
            common_data = pd.DataFrame()
            common_data['target'] = target_data
            common_data['compare'] = all_data[col]
            common_data = common_data.dropna()
            
            if len(common_data) > 30:  # Ensure enough data points
                correlation = common_data['target'].corr(common_data['compare'])
                correlations[col] = correlation
        
        # Sort by correlation (highest first)
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Return top N similar stocks
        return pd.DataFrame(sorted_correlations[:limit], columns=['Symbol', 'Correlation'])
    
    except Exception as e:
        print(f"Error in find_correlated_stocks for {ticker}: {str(e)}")
        return pd.DataFrame(columns=['Symbol', 'Correlation'])
    
def find_similar_volatility_stocks(ticker, lookback='1y', limit=10):
    """Find stocks with similar volatility and trading volume"""
    # Get data for target stock
    stock_data = yf.download(ticker, period=lookback, interval='1d')
    
    # Calculate volatility and average volume
    target_volatility = np.mean(stock_data['Close'].pct_change().std() * np.sqrt(252)) # Annualized
    target_volume = stock_data['Volume'].mean()[0]
    
    print(f"{ticker} - Volatility: {target_volatility:.2f}, Avg Volume: {target_volume:,.0f}")
    
    # Get data for comparison stocks (using S&P 500 as example)
    # sp500_tickers = sp500_tickers
    sample_size = min(100, len(sp500_tickers))
    sample_tickers = np.random.choice(sp500_tickers, sample_size, replace=False).tolist()
    
    if ticker in sample_tickers:
        sample_tickers.remove(ticker)
    
    # Collect volatility and volume data
    stock_metrics = []
    for t in sample_tickers:
        try:
            data = yf.download(t, period=lookback, interval='1d', progress=False)
            if len(data) > 30:  # Ensure enough data
                vol = data['Close'].pct_change().std() * np.sqrt(252)
                avg_volume = data['Volume'].mean()
                stock_metrics.append({
                    'Symbol': t,
                    'Volatility': vol,
                    'AvgVolume': avg_volume,
                    'VolDiff': abs(vol - target_volatility),  # Difference from target
                    'VolumeDiff': abs(1 - (avg_volume / target_volume))  # Relative difference
                })
        except Exception as e:
            print(f"Error processing {t}: {e}")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(stock_metrics)
    
    # Calculate combined similarity score (lower is better)
    metrics_df['SimilarityScore'] = metrics_df['VolDiff'] + metrics_df['VolumeDiff']
    
    # Sort by similarity score
    # return metrics_df.sort_values('SimilarityScore').head(limit)
    return metrics_df

def find_similar_fundamentals(ticker, limit=10):
    """Find stocks with similar fundamental characteristics"""
    # Get stock info
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Extract key fundamentals
    target_pe = info.get('forwardPE', None)
    target_pb = info.get('priceToBook', None)
    target_mc = info.get('marketCap', None)
    target_beta = info.get('beta', None)
    
    print(f"{ticker} - P/E: {target_pe}, P/B: {target_pb}, Market Cap: ${target_mc:,.0f}, Beta: {target_beta}")
    
    # Get comparison stocks
    # sp500_tickers = sp500_tickers
    sample_size = min(150, len(sp500_tickers))
    sample_tickers = np.random.choice(sp500_tickers, sample_size, replace=False).tolist()
    
    if ticker in sample_tickers:
        sample_tickers.remove(ticker)
    
    # Collect fundamental data
    fund_data = []
    for t in sample_tickers:
        try:
            comp = yf.Ticker(t)
            comp_info = comp.info
            
            pe = comp_info.get('forwardPE', None)
            pb = comp_info.get('priceToBook', None)
            mc = comp_info.get('marketCap', None)
            beta = comp_info.get('beta', None)
            
            # Skip if missing key data
            if not all([pe, mc, beta]):
                continue
                
            # Calculate similarity scores (lower is better)
            pe_diff = abs(pe - target_pe) / target_pe if target_pe and pe else 1
            pb_diff = abs(pb - target_pb) / target_pb if target_pb and pb else 1
            mc_diff = abs(mc - target_mc) / target_mc if target_mc and mc else 1
            beta_diff = abs(beta - target_beta) if target_beta and beta else 1
            
            # Combined score
            similarity_score = pe_diff * 0.3 + pb_diff * 0.2 + mc_diff * 0.25 + beta_diff * 0.25
            
            fund_data.append({
                'Symbol': t,
                'PE': pe,
                'PB': pb,
                'MarketCap': mc,
                'Beta': beta,
                'SimilarityScore': similarity_score
            })
            
        except Exception as e:
            continue
    
    # Convert to DataFrame and sort
    fund_df = pd.DataFrame(fund_data)
    return fund_df.sort_values('SimilarityScore').head(limit)

def find_similar_stocks_comprehensive(ticker, limit=20):
    """Find similar stocks using multiple criteria"""
    # Get similar stocks from different methods
    correlated = find_correlated_stocks(ticker, limit=30)
    volatility = find_similar_volatility_stocks(ticker, limit=30)
    fundamentals = find_similar_fundamentals(ticker, limit=30)
    
    # Combine results
    all_stocks = set(correlated['Symbol'].tolist() + 
                     volatility['Symbol'].tolist() + 
                     fundamentals['Symbol'].tolist())
    
    # Calculate combined score for each stock
    combined_scores = []
    for stock in all_stocks:
        score = 0
        
        # Add correlation score (higher is better)
        if stock in correlated['Symbol'].values:
            corr_value = correlated[correlated['Symbol'] == stock]['Correlation'].values[0]
            score += abs(corr_value) * 10  # Scale up for importance
        
        # Add volatility similarity (lower is better, so invert)
        if stock in volatility['Symbol'].values:
            vol_score = volatility[volatility['Symbol'] == stock]['SimilarityScore'].values[0]
            score += (1 / (vol_score + 0.1)) * 5
            
        # Add fundamentals similarity (lower is better, so invert)
        if stock in fundamentals['Symbol'].values:
            fund_score = fundamentals[fundamentals['Symbol'] == stock]['SimilarityScore'].values[0]
            score += (1 / (fund_score + 0.1)) * 7
            
        combined_scores.append({
            'Symbol': stock,
            'CombinedScore': float(score)
        })
    
    # Convert to DataFrame, sort and return top picks
    result_df = pd.DataFrame(combined_scores).sort_values('CombinedScore', ascending=False)
    return result_df.head(limit)

def create_strategy_specific_screener(ticker):
    """Create a screener based on the characteristics of your successful stock"""
    # Get stock data
    stock = yf.Ticker(ticker)
    info = stock.info
    price_data = yf.download(ticker, period='1y')
    
    # Extract key attributes that might be relevant for your strategy
    volatility = np.mean(price_data['Close'].pct_change().std() * np.sqrt(252))
    avg_volume = price_data['Volume'].mean()[0]
    beta = info.get('beta', 1.0)
    market_cap = info.get('marketCap', 0)
    pe = info.get('forwardPE', None)
    pb = info.get('priceToBook', None)
    
    # Define ranges for screening (typically ±30% of your target stock's values)
    vol_min, vol_max = volatility * 0.7, volatility * 1.3
    volume_min = avg_volume * 0.5  # At least half the volume
    beta_min, beta_max = beta * 0.7, beta * 1.3
    
    # Market cap category (small, mid, large)
    if market_cap < 2e9:
        cap_category = "Small Cap"
        cap_min, cap_max = 3e8, 2e9
    elif market_cap < 10e9:
        cap_category = "Mid Cap"
        cap_min, cap_max = 2e9, 10e9
    else:
        cap_category = "Large Cap"
        cap_min, cap_max = 10e9, 1e12
    
    print(f"Creating screener based on {ticker} ({cap_category}):")
    print(f"- Volatility range: {vol_min:.2f} to {vol_max:.2f}")
    print(f"- Minimum volume: {volume_min:,.0f}")
    print(f"- Beta range: {beta_min:.2f} to {beta_max:.2f}")
    print(f"- Market cap range: ${cap_min:,.0f} to ${cap_max:,.0f}")
    if pe:
        print(f"- P/E ratio: {pe:.2f} ±30%")
    if pb:
        print(f"- P/B ratio: {pb:.2f} ±30%")
        
    # In a full implementation, this would return a screener configuration
    # or directly screen stocks using a service like yfinance, pandas-datareader, etc.
    
    # For now, return the criteria as a dict
    return {
        'volatility_range': (vol_min, vol_max),
        'min_volume': volume_min,
        'beta_range': (beta_min, beta_max),
        'market_cap_range': (cap_min, cap_max),
        'pe_ratio': pe,
        'pb_ratio': pb
    }

def apply_strategy_screener(criteria, limit=20):
    """Apply the strategy screener criteria to find similar stocks"""
    # This is a simplified implementation
    # In practice, you'd use a proper stock screening API
    
   
    # For demonstration, we'll screen a sample
    sample_size = min(200, len(sp500_tickers))
    sample_tickers = np.random.choice(sp500_tickers, sample_size, replace=False).tolist()
    
    # Apply criteria
    matched_stocks = []
    for ticker in sample_tickers:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            info = stock.info
            price_data = yf.download(ticker, period='1y', progress=False)
            
            if len(price_data) < 100:  # Skip stocks with insufficient data
                continue
                
            # Check criteria
            volatility = price_data['Close'].pct_change().std() * np.sqrt(252)
            avg_volume = price_data['Volume'].mean()
            beta = info.get('beta', None)
            market_cap = info.get('marketCap', 0)
            pe = info.get('forwardPE', None)
            pb = info.get('priceToBook', None)
            
            # Apply filters
            if not (criteria['volatility_range'][0] <= volatility <= criteria['volatility_range'][1]):
                continue
            
            if avg_volume < criteria['min_volume']:
                continue
                
            if beta and not (criteria['beta_range'][0] <= beta <= criteria['beta_range'][1]):
                continue
                
            if not (criteria['market_cap_range'][0] <= market_cap <= criteria['market_cap_range'][1]):
                continue
                
            # Optional PE filter
            if criteria['pe_ratio'] and pe:
                if not (criteria['pe_ratio']*0.7 <= pe <= criteria['pe_ratio']*1.3):
                    continue
            
            # Stock passed all filters
            matched_stocks.append({
                'Symbol': ticker,
                'Volatility': volatility,
                'Avg_Volume': avg_volume,
                'Beta': beta,
                'Market_Cap': market_cap,
                'PE': pe,
                'PB': pb
            })
            
        except Exception as e:
            continue
            
    # Return results
    return pd.DataFrame(matched_stocks).head(limit)



symbol = "QBTS"
start = "2020-03-01"
end = "2025-05-17"
# end = date.today()
_predictor = StockPredictor(symbol=symbol, start_date=start, end_date=end)
_predictor.load_data()
backtester = AptosBacktester(symbol=symbol, initial_capital=100000)
# Run a simple backtest with default strategy
print("Running backtest...")
history, metrics = backtester.run_backtest(
    start_date=start,
    end_date=end,
    signal_generator=create_signal_generator(
        predictor=_predictor
    ),  # Use the predictor's signal generator
)

print("\nBacktest Results:")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Number of Trades: {metrics['num_trades']}")
print(f"Number of BUY orders: {metrics['num_buy_trades']}")
print(f"Number of SELL orders: {metrics['num_sell_trades']}")

# Plot results
backtester.plot_results(history)




# Start with your successful stock
successful_ticker = "QBTS"  # Your Stock A
# successful_ticker = "AAPL"  # Your Stock A

# 1. Find similar stocks using multiple methods
similar_stocks = find_similar_stocks_comprehensive(successful_ticker)
# similar_stocks = find_correlated_stocks(successful_ticker, limit=30)
print(f"Top similar stocks to {successful_ticker}:")
print(similar_stocks)

candidates = similar_stocks['Symbol'].head(5).tolist()
logging.info(f"Top candidates for backtesting: {candidates}")
# # 2. Create and apply a strategy-specific screener
# strategy_criteria = create_strategy_specific_screener(successful_ticker)
# screener_results = apply_strategy_screener(strategy_criteria)
# print("\nStocks matching your strategy criteria:")
# print(screener_results)

# # 3. Backtest your strategy on the top candidates
# if len(screener_results) == 0:
#     print("No stocks matched the screener criteria.")
#     candidates = []
# else:
#     candidates = list(set(similar_stocks['Symbol'].head(10).tolist() + 
#                       screener_results['Symbol'].head(10).tolist()))



for candidate in candidates:
    print(f"\nBacktesting strategy on {candidate}...")
    test_predictor = StockPredictor(symbol=candidate, start_date="2020-03-01", end_date="2025-05-17")
    test_predictor.load_data()
    
    backtester = AptosBacktester(symbol=candidate, initial_capital=10000)
    history, metrics = backtester.run_backtest(
        start_date="2020-03-01",
        end_date="2025-05-17",
        signal_generator=create_signal_generator(test_predictor)
    )
    backtester.plot_results(history)
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")