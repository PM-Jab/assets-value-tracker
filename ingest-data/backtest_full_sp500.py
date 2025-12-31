import pandas as pd
import pandas_ta as ta
import yfinance as yf
import pandas_datareader.data as web
import joblib
import matplotlib.pyplot as plt
import datetime
import concurrent.futures
from sqlalchemy import create_engine

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
MODEL_FILE = "sp500_universal_model.pkl"
USE_MACRO_FILTER = True  # Set to False to test "Pure AI"
TEST_PERIOD_YEARS = 2

def get_all_tickers(engine):
    """Fetches all tickers from your database."""
    return pd.read_sql("SELECT DISTINCT ticker FROM stock_prices", engine)['ticker'].tolist()

def get_macro_data():
    """Fetches Yield Curve for the test period."""
    if not USE_MACRO_FILTER: return pd.DataFrame()
    
    start = datetime.datetime.now() - datetime.timedelta(days=TEST_PERIOD_YEARS*365 + 100)
    try:
        yc = web.DataReader('T10Y2Y', 'fred', start, datetime.datetime.now())
        yc.columns = ['yield_curve']
        return yc
    except:
        return pd.DataFrame()

def backtest_ticker(ticker, model, macro_df):
    try:
        # 1. Fetch Data
        df = yf.download(ticker, period=f"{TEST_PERIOD_YEARS}y", progress=False, auto_adjust=True)
        
        # --- FIX STARTS HERE ---
        # If yfinance returns a MultiIndex (e.g., ('Close', 'AAPL')), flatten it to just 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Now it is safe to lowercase
        df.columns = df.columns.str.lower()
        # --- FIX ENDS HERE ---
        
        if len(df) < 200: return None
        
        # 2. Features
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['dist_sma50'] = df['close'] / df['SMA_50']
        df['dist_sma200'] = df['close'] / df['SMA_200']
        df['RSI'] = ta.rsi(df['close'], length=14)
        bbands = ta.bbands(df['close'], length=20, std=2.0)
        df['bb_width'] = bbands.iloc[:, 3]
        df['vol_change'] = df['volume'].pct_change()
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df.dropna(inplace=True)

        # 3. Merge Macro
        if not macro_df.empty:
            # Timezone Fix
            if df.index.tz is not None and macro_df.index.tz is None:
                df.index = df.index.tz_localize(None)
            elif df.index.tz is None and macro_df.index.tz is not None:
                macro_df.index = macro_df.index.tz_localize(None)
            
            df = df.join(macro_df).ffill()

        # 4. Predict
        features = ['dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 'return_1d', 'return_5d']
        
        # Ensure all features exist before predicting
        if not all(col in df.columns for col in features):
            return None

        df['signal'] = model.predict(df[features])
        df['conf'] = model.predict_proba(df[features])[:, 1]

        # 5. Simulate Trade
        cash = 2000
        shares = 0
        equity = []

        for i in range(len(df)):
            row = df.iloc[i]
            
            macro_risk = (row['yield_curve'] < -0.1) if 'yield_curve' in row else False
            buy_signal = (row['signal'] == 1) and (row['conf'] > 0.55)

            if buy_signal and not macro_risk:
                if cash > 0:
                    shares = cash / row['close']
                    cash = 0
            elif (not buy_signal) or macro_risk:
                if shares > 0:
                    cash = shares * row['close']
                    shares = 0
            
            equity.append(cash + (shares * row['close']))
            
        return pd.Series(equity, index=df.index, name=ticker)
    
    except Exception as e:
        # Un-comment this if you want to see errors, but for 500 stocks it might be spammy
        # print(f"Failed on {ticker}: {e}") 
        return None

def run_fund_simulation():
    print("Loading Engine & Model...")
    engine = create_engine(DB_URI)
    model = joblib.load(MODEL_FILE)
    tickers = get_all_tickers(engine)
    macro_df = get_macro_data()
    
    print(f"Starting Simulation on {len(tickers)} stocks...")
    print(f"Strategy: {'Hybrid (AI + Macro)' if USE_MACRO_FILTER else 'Pure AI'}")
    
    # Run in Parallel (Fast!)
    equity_curves = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(backtest_ticker, t, model, macro_df): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0: print(f"Processed {i}/{len(tickers)}...", end="\r")
            res = future.result()
            if res is not None:
                equity_curves.append(res)

    print("\nAggregating Portfolio Performance...")
    # Combine all individual stock curves into one DataFrame
    portfolio_df = pd.concat(equity_curves, axis=1)
    
    # Sum daily values to get Total Fund Equity
    # (ffill handles days where some stocks have data and others don't)
    portfolio_df.ffill(inplace=True)
    portfolio_df.fillna(2000, inplace=True) # Fill pre-IPO dates with initial cash
    
    total_equity = portfolio_df.sum(axis=1)
    
    # --- BENCHMARK COMPARISON (SPY ETF) ---
    print("Fetching Benchmark (SPY)...")
    spy = yf.download("SPY", start=total_equity.index[0], end=total_equity.index[-1], progress=False)
    
    # Fix: Handle MultiIndex columns for SPY just like we did for other stocks
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    spy = spy['Close'] # Now this is safe
    
    # Normalize SPY to start at the same value as our Fund
    spy_normalized = (spy / spy.iloc[0]) * total_equity.iloc[0]

    # --- REPORT ---
    initial_val = total_equity.iloc[0]
    final_val = total_equity.iloc[-1]
    ret = (final_val - initial_val) / initial_val
    
    # FIX: Use .iloc[-1] and .iloc[0] to get scalars, and convert to float with .item()
    spy_final = spy_normalized.iloc[-1].item() if hasattr(spy_normalized.iloc[-1], 'item') else spy_normalized.iloc[-1]
    spy_initial = spy_normalized.iloc[0].item() if hasattr(spy_normalized.iloc[0], 'item') else spy_normalized.iloc[0]
    
    spy_ret = (spy_final - spy_initial) / spy_initial

    print("\n" + "="*50)
    print(f"FUND PERFORMANCE REPORT ({len(equity_curves)} Stocks)")
    print("="*50)
    print(f"Initial AUM:     ${initial_val:,.2f}")
    print(f"Final AUM:       ${final_val:,.2f}")
    print(f"Total Return:    {ret:.2%}")
    print(f"S&P 500 Return:  {spy_ret:.2%}")
    print("-" * 50)
    
    if ret > spy_ret:
        print("✅ RESULT: AI BEAT THE MARKET")
    else:
        print("❌ RESULT: AI Underperformed")

    # --- PLOT ---
    plt.figure(figsize=(12, 6))
    plt.plot(total_equity.index, total_equity, label='AI Fund', color='green', linewidth=2)
    plt.plot(spy_normalized.index, spy_normalized, label='S&P 500 (Benchmark)', color='gray', alpha=0.5)
    plt.title(f"AI Hedge Fund Simulation vs S&P 500")
    plt.ylabel("Assets Under Management ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_fund_simulation()