import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine
from macro_data import get_fed_data
from news_sentiment import get_sentiment

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
TICKER = "AAPL"

def fetch_data(ticker, engine):
    query = f"SELECT time, open, high, low, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY time ASC;"
    df = pd.read_sql(query, engine)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def add_features(df):
    # 1. Standard Indicators
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    
    # 2. Relative Features (Ratios)
    df['dist_sma50'] = df['close'] / df['SMA_50']
    df['dist_sma200'] = df['close'] / df['SMA_200']
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    bbands = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_width'] = bbands.iloc[:, 3] # Bandwidth
    df['vol_change'] = df['volume'].pct_change()

    # 3. NEW: Context Features (Lags)
    # This tells the model: "Did we drop 5% yesterday?"
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    
    # 1. MERGE MACRO DATA (The "Fed" Context)
    macro_df = get_fed_data(start_date=df.index[0].strftime('%Y-%m-%d'))
    
    # Join macro data onto your stock data by Date
    df = df.join(macro_df, how='left')
    df.ffill(inplace=True) # Fill gaps (e.g., weekends)
    
    # 2. ADD NEWS SENTIMENT (The "Vibe" Context)
    # WARNING: Historical news is hard to get for free. 
    # For training (Backtesting), you might use a static score or 0.
    # For LIVE PREDICTION (`predict_tomorrow.py`), you fetch the real score:
    
    # Example logic for live prediction:
    # df['news_score'] = get_sentiment(ticker)
    
    return df

def create_target(df, horizon=5): 
    # --- CRITICAL CHANGE: HORIZON is now 5 DAYS ---
    # We are predicting if price will be higher next week, not tomorrow.
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    engine = create_engine(DB_URI)
    print(f"Processing {TICKER}...")
    
    df = fetch_data(TICKER, engine)
    df = add_features(df)
    df = create_target(df, horizon=5) # 5-Day Prediction
    
    df.to_csv(f"{TICKER}_processed.csv")
    print(f"Saved {TICKER}_processed.csv with {len(df)} rows.")