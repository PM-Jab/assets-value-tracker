import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine, text
import joblib
import concurrent.futures

# --- Configuration ---
# Update this if your model filename is different
MODEL_FILE = "sp500_macro_model.pkl" 
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'

def get_all_tickers(engine):
    """Get every single ticker in the database."""
    return pd.read_sql("SELECT DISTINCT ticker FROM stock_prices;", engine)['ticker'].tolist()

def get_latest_macro(engine):
    """
    Fetches the single most recent row of economic data.
    We apply this 'current state of the economy' to every stock we scan.
    """
    query = """
    SELECT fed_rate, yield_curve, inflation_cpi 
    FROM macro_data 
    ORDER BY time DESC 
    LIMIT 1;
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise ValueError("Macro data table is empty! Run ingest_macro.py first.")
        
        # Return as a dictionary (e.g., {'fed_rate': 5.33, 'yield_curve': -0.4, ...})
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return None

def analyze_ticker(ticker, model, macro_dict, engine):
    """Analyzes a SINGLE stock using Price + Macro Data."""
    try:
        # 1. Fetch Stock Data (Last 300 days)
        query = f"SELECT time, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY time DESC LIMIT 300"
        df = pd.read_sql(query, engine)
        
        if len(df) < 200: return None
        
        df = df.sort_values('time')
        df.set_index('time', inplace=True)
        
        # 2. Technical Features (MUST match training exactly)
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

        # 3. Take the LATEST row (Today's Close)
        latest = df.iloc[[-1]].copy()
        
        # 4. INJECT MACRO DATA
        # We add the economic columns to this specific row
        latest['fed_rate'] = macro_dict['fed_rate']
        latest['yield_curve'] = macro_dict['yield_curve']
        latest['inflation_cpi'] = macro_dict['inflation_cpi']

        # 5. Predict
        # The list must match train_with_macro.py EXACTLY
        features = [
            'dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 
            'return_1d', 'return_5d',
            'fed_rate', 'yield_curve', 'inflation_cpi'
        ]
        
        prediction = model.predict(latest[features])[0]
        probability = model.predict_proba(latest[features])[0][1] # Probability of "UP"
        
        return {
            "Ticker": ticker,
            "Price": latest['close'].values[0],
            "RSI": latest['RSI'].values[0],
            "Signal": "BUY" if prediction == 1 else "SELL",
            "Confidence": probability,
            "Fed_Rate": macro_dict['fed_rate'] # Just for reference in the report
        }
    except Exception as e:
        # print(f"Error on {ticker}: {e}")
        return None

def run_scanner():
    print(f"Loading Model: {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
    except:
        print("Model not found! Run train_with_macro.py first.")
        return

    engine = create_engine(DB_URI)
    
    # 1. Get Global Data
    tickers = get_all_tickers(engine)
    macro_dict = get_latest_macro(engine)
    
    if not macro_dict:
        return

    print(f"Current Market Regime: Fed Rate {macro_dict['fed_rate']}%, Yield Curve {macro_dict['yield_curve']}")
    print(f"Scanning {len(tickers)} stocks...")
    
    results = []
    
    # 2. Parallel Scanning
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, t, model, macro_dict, engine): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0: print(f"Scanned {i}/{len(tickers)}...", end="\r")
            res = future.result()
            if res:
                results.append(res)

    # 3. Report
    scan_df = pd.DataFrame(results)
    
    # Filter: Buy Signals with > 60% Confidence
    opportunities = scan_df[ (scan_df['Signal'] == 'BUY') & (scan_df['Confidence'] > 0.60) ]
    opportunities = opportunities.sort_values('Confidence', ascending=False)
    
    print("\n" + "="*60)
    print(f"MACRO-AWARE TOP OPPORTUNITIES (Total: {len(opportunities)})")
    print("="*60)
    print(opportunities[['Ticker', 'Price', 'RSI', 'Confidence', 'Fed_Rate']].head(15))
    
    opportunities.to_csv("macro_scanner_report.csv", index=False)
    print("\nFull report saved to 'macro_scanner_report.csv'")

if __name__ == "__main__":
    run_scanner()