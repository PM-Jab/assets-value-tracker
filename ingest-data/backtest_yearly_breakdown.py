import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASKET_TICKERS = ['EVRG', 'FAST', 'DTE', 'MO', 'LNT', 'NI', 'KO', 'HSY']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['^IRX', '^TNX']

# Extend back slightly to capture full 2019 if needed
START_DATE = '2018-12-25' 
END_DATE = '2025-12-30'

# ==========================================
# 2. DATA INGESTION
# ==========================================
print("Fetching data...")
all_tickers = BASKET_TICKERS + [BENCHMARK] + MACRO_TICKERS
raw_data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

# Handle MultiIndex column structure
if 'Close' in raw_data.columns.levels[0]:
    data = raw_data['Close']
else:
    data = raw_data

# Use new pandas recommended ffill
data = data.ffill().dropna()

# ==========================================
# 3. STRATEGY LOGIC (Identical to previous)
# ==========================================
data['Yield_Curve'] = data['^TNX'] - data['^IRX']
data['Rate_Vol'] = data['^IRX'].rolling(window=30).std()

# Signal: 1 = Basket, 0 = SPY
data['Signal'] = np.where(
    (data['Yield_Curve'] > 0) & (data['Rate_Vol'] < 0.5), 
    1, 
    0 
)
data['Signal'] = data['Signal'].shift(1)

# ==========================================
# 4. CALCULATE RETURNS
# ==========================================
returns = data.pct_change()
basket_returns = returns[BASKET_TICKERS].mean(axis=1)
spy_returns = returns[BENCHMARK]

# Strategy Returns
strat_returns = (data['Signal'] * basket_returns) + ((1 - data['Signal']) * spy_returns)

# ==========================================
# 5. YEARLY BREAKDOWN
# ==========================================
# Combine into a DataFrame for resampling
analysis_df = pd.DataFrame({
    'Strategy': strat_returns,
    'SP500': spy_returns
})

# Resample by Year ('YE' is the new alias for Year End in pandas 2.2+, 'Y' for older)
# We calculate annual return by compounding daily returns
yearly_perf = analysis_df.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100

print("\n" + "="*50)
print(f" YEARLY PERFORMANCE COMPARISON")
print("="*50)
print(f"{'Year':<10} | {'Strategy':<15} | {'S&P 500':<15} | {'Diff'}")
print("-" * 55)

for index, row in yearly_perf.iterrows():
    year = index.year
    strat = row['Strategy']
    sp500 = row['SP500']
    diff = strat - sp500
    
    # Color coding for terminal output (optional, keeping it simple text here)
    marker = "WIN " if diff > 0 else "    "
    print(f"{year:<10} | {strat:>14.2f}% | {sp500:>14.2f}% | {diff:>+6.2f}% {marker}")

print("-" * 55)