import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_money_flow():
    # 1. Define the Assets representing different capital pools
    tickers = {
        'Stocks (S&P500)': '^GSPC',
        'Bonds (TLT)': 'TLT',
        'Gold': 'GC=F',
        'USD Index': 'DX-Y.NYB',
        'Bitcoin': 'BTC-USD'
    }
    
    print("Fetching intermarket data...")
    # Fetch data for the last 6 months to see the trend
    df = yf.download(list(tickers.values()), period="6mo", progress=False)['Close']
    
    # Rename columns for readability
    # Invert dictionary to map Ticker -> Name
    inv_tickers = {v: k for k, v in tickers.items()}
    df.columns = [inv_tickers.get(c, c) for c in df.columns]
    
    # 2. Calculate Daily Returns (Percentage Moves)
    returns = df.pct_change().dropna()
    
    # 3. Calculate Correlation with Stocks (S&P 500)
    # We look at the last 30 days to see the "Current Regime"
    # +1.0 = Moving Together
    # -1.0 = Moving Opposite (Money Flowing FROM one TO the other)
    recent_corr = returns.tail(30).corr()['Stocks (S&P500)'].sort_values()
    
    print("\n" + "="*50)
    print(" MONEY FLOW ANALYSIS (Correlation with Stocks)")
    print(" Positive = Moving WITH Stocks | Negative = Hedge/Safety")
    print("="*50)
    
    for asset, corr in recent_corr.items():
        if asset == 'Stocks (S&P500)': continue
        
        status = ""
        if corr < -0.3: status = "[HEDGE / SAFETY FLOW]"   # Money moving opposite to stocks
        elif corr > 0.7: status = "[RISK-ON SYNC]"         # Moving exactly with stocks
        else: status = "[Neutral / Uncorrelated]"
        
        print(f"{asset:<20} | Correlation: {corr:>6.2f}  {status}")

    # 4. Visual Check: Normalized Performance (Last 30 Days)
    # Re-base everything to start at 100 so we can compare growth
    last_30 = df.tail(30).copy()
    normalized = (last_30 / last_30.iloc[0]) * 100
    
    plt.figure(figsize=(12, 6))
    for col in normalized.columns:
        # Highlight Stocks in thick black, others in colors
        width = 3 if col == 'Stocks (S&P500)' else 1.5
        style = '-'
        if col == 'Stocks (S&P500)': color = 'black'
        else: color = None # Auto color
        
        plt.plot(normalized.index, normalized[col], label=col, linewidth=width, linestyle=style, color=color)
        
    plt.title("Money Flow: 30-Day Relative Performance")
    plt.ylabel("Rebased Price (Starts at 100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze_money_flow()