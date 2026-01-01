import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
MODEL_FILE = "sp500_macro_model.pkl"

def get_top_tickers(engine, limit=50):
    """Finds top 50 most liquid stocks to train on."""
    query = f"""
    SELECT ticker FROM stock_prices 
    GROUP BY ticker 
    ORDER BY AVG(volume) DESC LIMIT {limit};
    """
    return pd.read_sql(query, engine)['ticker'].tolist()

def fetch_and_process(ticker, engine):
    """
    Fetches Stock Data AND joins it with Macro Data.
    """
    # --- 1. THE JOIN QUERY (Crucial Step) ---
    # We join stock_prices (s) with macro_data (m) on the 'time' column.
    query = f"""
    SELECT 
        s.time, s.close, s.volume, 
        m.fed_rate, m.yield_curve, m.inflation_cpi, m.unemployment
    FROM stock_prices s
    LEFT JOIN macro_data m ON s.time = m.time
    WHERE s.ticker = '{ticker}'
    ORDER BY s.time ASC;
    """
    
    try:
        df = pd.read_sql(query, engine)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Fill missing macro data (forward fill last known rate)
        df.ffill(inplace=True)
        df.dropna(inplace=True) # Drop if any early days have absolutely no macro data

        # --- 2. Technical Features (Same as before) ---
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

        # --- 3. Create Target (5-Day Horizon) ---
        df['future_close'] = df['close'].shift(-5)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        return pd.DataFrame()

def train_macro_model():
    engine = create_engine(DB_URI)
    tickers = get_top_tickers(engine, limit=50)
    
    all_data = []
    print(f"Gathering data for {len(tickers)} stocks...")
    
    for i, t in enumerate(tickers):
        print(f"Processing {t} ({i+1}/{len(tickers)})...", end="\r")
        df = fetch_and_process(t, engine)
        if not df.empty:
            all_data.append(df)
            
    full_df = pd.concat(all_data)
    print(f"\nTraining on {len(full_df)} rows.")

    # --- 4. NEW FEATURE LIST (Include Macro) ---
    feature_cols = [
        'dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 
        'return_1d', 'return_5d',
        'fed_rate', 'yield_curve', 'inflation_cpi'  # <--- NEW FEATURES
    ]
    
    X = full_df[feature_cols]
    y = full_df['target']

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nMACRO-AWARE MODEL ACCURACY: {acc:.2%}")
    
    # Show Importance (Did the Fed Rate matter?)
    print("\nFeature Importance:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances)
    
    joblib.dump(model, MODEL_FILE)
    print(f"Saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_macro_model()