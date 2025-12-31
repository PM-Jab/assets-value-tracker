import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
MODEL_FILE = "sp500_universal_model.pkl"

def get_top_tickers(engine, limit=50):
    """Finds the top 50 most liquid stocks to train on."""
    print(f"Finding top {limit} stocks by volume...")
    query = f"""
    SELECT ticker 
    FROM stock_prices 
    GROUP BY ticker 
    ORDER BY AVG(volume) DESC 
    LIMIT {limit};
    """
    df = pd.read_sql(query, engine)
    return df['ticker'].tolist()

def fetch_and_process(ticker, engine):
    """Fetches data and calculates features (Same logic as before)."""
    try:
        query = f"SELECT time, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY time ASC"
        df = pd.read_sql(query, engine)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # --- Feature Engineering ---
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['dist_sma50'] = df['close'] / df['SMA_50']
        df['dist_sma200'] = df['close'] / df['SMA_200']
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        bbands = ta.bbands(df['close'], length=20, std=2.0)
        df['bb_width'] = bbands.iloc[:, 3]
        df['vol_change'] = df['volume'].pct_change()
        
        # Context Lags
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)

        # Target (5-Day Horizon)
        df['future_close'] = df['close'].shift(-5)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty if error

def train_universal_model():
    engine = create_engine(DB_URI)
    tickers = get_top_tickers(engine, limit=50) # Train on top 50 stocks
    
    all_data = []
    print("Gathering training data...")
    
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {t}...", end="\r")
        df = fetch_and_process(t, engine)
        if not df.empty:
            all_data.append(df)
            
    # Combine 50 stocks into one giant dataset
    full_df = pd.concat(all_data)
    print(f"\nTraining on {len(full_df)} total rows of data.")

    # Features
    feature_cols = ['dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 'return_1d', 'return_5d']
    X = full_df[feature_cols]
    y = full_df['target']

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nUNIVERSAL MODEL ACCURACY: {acc:.2%}")
    
    joblib.dump(model, MODEL_FILE)
    print(f"Saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_universal_model()