import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- Configuration ---
TICKER = "AAPL"
DATA_FILE = f"{TICKER}_processed.csv"
MODEL_FILE = f"{TICKER}_model.pkl"
INITIAL_CAPITAL = 10000  # Start with $10,000 USD


def run_backtest():
    # 1. Load Data & Model
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Load the trained model
    model = joblib.load(MODEL_FILE)
    
    # 2. Re-create Features (Must match training exactly!)
    feature_cols = [
        'dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change',
        'return_1d', 'return_5d'
    ]
    
    # 3. Split the same way we did in training (Last 20% is unseen data)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()
    
    # 4. Generate Predictions
    X_test = test_df[feature_cols]
    test_df['prediction'] = model.predict(X_test)
    
    # 5. Simulate Trading
    # Calculate daily returns of the stock
    test_df['stock_return'] = test_df['close'].pct_change()
    
    # Strategy Logic:
    # If prediction was 1 (UP) yesterday, we hold the stock today.
    # If prediction was 0 (DOWN), we stay in Cash (0% return).
    # We shift prediction by 1 because we trade based on *yesterday's* signal.
    test_df['strategy_return'] = test_df['stock_return'] * test_df['prediction'].shift(1)
    
    # 6. Calculate Cumulative Growth
    # (1 + return).cumprod() calculates compound growth
    test_df['buy_and_hold_equity'] = INITIAL_CAPITAL * (1 + test_df['stock_return']).cumprod()
    test_df['ai_strategy_equity'] = INITIAL_CAPITAL * (1 + test_df['strategy_return']).cumprod()
    
    # 7. Print Results
    final_bnh = test_df['buy_and_hold_equity'].iloc[-1]
    final_ai = test_df['ai_strategy_equity'].iloc[-1]
    
    print("-" * 40)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Buy & Hold Final: ${final_bnh:,.2f} ({(final_bnh - INITIAL_CAPITAL)/INITIAL_CAPITAL:.2%})")
    print(f"AI Strategy Final: ${final_ai:,.2f} ({(final_ai - INITIAL_CAPITAL)/INITIAL_CAPITAL:.2%})")
    print("-" * 40)

    # 8. Plot Performance
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['buy_and_hold_equity'], label='Buy & Hold', color='gray', alpha=0.6)
    plt.plot(test_df.index, test_df['ai_strategy_equity'], label='AI Strategy', color='green', linewidth=2)
    plt.title(f"Backtest: AI vs Buy & Hold ({TICKER})")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()