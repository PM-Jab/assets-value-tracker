import pandas as pd
import pandas_ta as ta
import json
import joblib
from sqlalchemy import create_engine
from datetime import datetime

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'
MODEL_FILE = "sp500_universal_model.pkl"
PORTFOLIO_FILE = "portfolio.json"
STOP_LOSS_PCT = 0.05  # 5% Hard Stop Loss

def get_latest_data(ticker, engine):
    """Fetches latest data for a single owned stock."""
    try:
        query = f"SELECT time, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY time DESC LIMIT 300"
        df = pd.read_sql(query, engine)
        if len(df) < 200: return None
        
        df = df.sort_values('time')
        df.set_index('time', inplace=True)
        return df
    except:
        return None

def analyze_position(position, model, engine):
    ticker = position['ticker']
    buy_price = position['buy_price']
    
    df = get_latest_data(ticker, engine)
    if df is None:
        return {"Ticker": ticker, "Action": "ERROR", "Reason": "No Data"}

    # 1. Calculate Features (Must match training!)
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
    latest = df.iloc[[-1]]
    current_price = latest['close'].values[0]

    # --- DECISION LOGIC ---

    # Rule 1: STOP LOSS CHECK
    # If we lost more than 5%, SELL immediately.
    loss_pct = (current_price - buy_price) / buy_price
    if loss_pct < -STOP_LOSS_PCT:
        return {
            "Ticker": ticker,
            "Current": current_price,
            "P&L": f"{loss_pct:.2%}",
            "Action": "SELL NOW",
            "Reason": "STOP LOSS HIT"
        }

    # Rule 2: AI CHECK
    features = ['dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change', 'return_1d', 'return_5d']
    prediction = model.predict(latest[features])[0]
    confidence = model.predict_proba(latest[features])[0][1] # Confidence of UP

    if prediction == 0:  # AI predicts DOWN
        return {
            "Ticker": ticker,
            "Current": current_price,
            "P&L": f"{loss_pct:.2%}",
            "Action": "SELL NOW",
            "Reason": "AI SIGNAL REVERSAL (Bearish)"
        }
    
    # Rule 3: WEAK HOLD CHECK
    # If AI says UP but confidence is low (e.g., < 55%), maybe take profit?
    if prediction == 1 and confidence < 0.55:
        return {
            "Ticker": ticker,
            "Current": current_price,
            "P&L": f"{loss_pct:.2%}",
            "Action": "WATCH / SELL",
            "Reason": f"Weak Confidence ({confidence:.2%})"
        }

    return {
        "Ticker": ticker,
        "Current": current_price,
        "P&L": f"{loss_pct:.2%}",
        "Action": "HOLD",
        "Reason": f"AI Bullish ({confidence:.2%})"
    }

def run_portfolio_check():
    print("Loading Portfolio & Model...")
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
    
    model = joblib.load(MODEL_FILE)
    engine = create_engine(DB_URI)
    
    print(f"{'TICKER':<8} | {'PRICE':<8} | {'P&L':<8} | {'ACTION':<12} | {'REASON'}")
    print("-" * 60)
    
    for position in portfolio:
        res = analyze_position(position, model, engine)
        
        # Color code output
        action = res['Action']
        if "SELL" in action:
            print(f"{res['Ticker']:<8} | {res['Current']:<8.2f} | {res['P&L']:<8} | \033[91m{action:<12}\033[0m | {res['Reason']}")
        elif "HOLD" in action:
            print(f"{res['Ticker']:<8} | {res['Current']:<8.2f} | {res['P&L']:<8} | \033[92m{action:<12}\033[0m | {res['Reason']}")
        else:
            print(f"{res['Ticker']:<8} | {res['Current']:<8.2f} | {res['P&L']:<8} | {action:<12} | {res['Reason']}")

if __name__ == "__main__":
    run_portfolio_check()