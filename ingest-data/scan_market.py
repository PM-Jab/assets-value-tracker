import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine
import joblib
import concurrent.futures

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
MODEL_FILE = "sp500_universal_model.pkl"

def get_all_tickers(engine):
    """Get every single ticker in the database."""
    return pd.read_sql("SELECT DISTINCT ticker FROM stock_prices;", engine)['ticker'].tolist()

def analyze_ticker(ticker, model, engine):
    """Analyzes a SINGLE stock and returns the result."""
    try:
        # Fetch last 300 days (enough for SMA_200)
        query = f"SELECT time, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY time DESC LIMIT 300"
        df = pd.read_sql(query, engine)
        
        if len(df) < 200: return None # Not enough data
        
        df = df.sort_values('time') # Sort ASC for calculation
        df.set_index('time', inplace=True)
        
        # --- Calculate Features (Same as training) ---
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
        if df.empty: return None

        # Take the LATEST row
        latest = df.iloc[[-1]]
        
        # Predict
        features = ['dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 'return_1d', 'return_5d']
        prediction = model.predict(latest[features])[0]
        probability = model.predict_proba(latest[features])[0][1] # Probability of "1" (UP)
        
        return {
            "Ticker": ticker,
            "Price": latest['close'].values[0],
            "RSI": latest['RSI'].values[0],
            "Signal": "BUY" if prediction == 1 else "SELL",
            "Confidence": probability
        }
    except:
        return None

def run_scanner():
    print("Loading Universal Model...")
    try:
        model = joblib.load(MODEL_FILE)
    except:
        print("Model not found! Run train_universal.py first.")
        return

    engine = create_engine(DB_URI)
    tickers = get_all_tickers(engine)
    print(f"Scanning {len(tickers)} stocks...")
    
    results = []
    
    # --- PARALLEL PROCESSING (Speed up scanning) ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {executor.submit(analyze_ticker, t, model, engine): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0: print(f"Scanned {i}/{len(tickers)}...", end="\r")
            res = future.result()
            if res:
                results.append(res)

    # --- REPORTING ---
    scan_df = pd.DataFrame(results)
    
    # Filter for High Confidence BUY signals (> 60%)
    opportunities = scan_df[ (scan_df['Signal'] == 'BUY') & (scan_df['Confidence'] > 0.60) ]
    opportunities = opportunities.sort_values('Confidence', ascending=False)
    
    print("\n" + "="*50)
    print(f"TOP OPPORTUNITIES (Total Found: {len(opportunities)})")
    print("="*50)
    print(opportunities[['Ticker', 'Price', 'RSI', 'Confidence']].head(15))
    
    # Save report
    opportunities.to_csv("daily_scanner_report.csv", index=False)
    print("\nFull report saved to 'daily_scanner_report.csv'")

if __name__ == "__main__":
    run_scanner()