import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASKET_TICKERS = ['EVRG', 'FAST', 'DTE', 'MO', 'LNT', 'NI', 'KO', 'HSY']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['^IRX', '^TNX']

START_DATE = '2018-01-01'
END_DATE = '2025-12-30'

# ==========================================
# 2. DATA INGESTION
# ==========================================
print("Fetching data (Trend + Macro)...")
all_tickers = BASKET_TICKERS + [BENCHMARK] + MACRO_TICKERS
raw_data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

# Handle MultiIndex
if 'Close' in raw_data.columns.levels[0]:
    data = raw_data['Close']
else:
    data = raw_data

# Fill missing data
data = data.ffill().dropna()

# ==========================================
# 3. CALCULATE RETURNS (FIXED)
# ==========================================
# We calculate returns ONLY for the tradeable assets (Stocks + SPY)
# We do this BEFORE adding indicator columns to avoid ZeroDivisionError
tradeable_assets = BASKET_TICKERS + [BENCHMARK]
returns = data[tradeable_assets].pct_change()

# ==========================================
# 4. CALCULATE INDICATORS
# ==========================================

# A. MACRO INDICATORS
data['Yield_Curve'] = data['^TNX'] - data['^IRX']
data['Rate_Vol'] = data['^IRX'].rolling(window=30).std()

# B. TREND INDICATOR (The "Bull Market" Filter)
# Calculate 200-day Simple Moving Average of SPY
data['SPY_SMA200'] = data[BENCHMARK].rolling(window=200).mean()

# Check if SPY is in a Bull Trend (Price > SMA)
data['Bull_Trend'] = data[BENCHMARK] > data['SPY_SMA200']

# ==========================================
# 5. SIGNAL GENERATION
# ==========================================

# Step 1: Define the "Defensive" Macro Condition
macro_is_defensive = (data['Yield_Curve'] > 0) & (data['Rate_Vol'] < 0.5)

# Step 2: Combine with Trend Override
# Logic: 
# IF Bull_Trend is TRUE -> Hold SPY (Signal 0)
# ELSE IF Macro is Defensive -> Hold Basket (Signal 1)
# ELSE -> Hold SPY (Signal 0)
# (Basically: Only use Basket if Market is Weak AND Macro is favorable)

data['Signal'] = np.where(
    (data['Bull_Trend'] == False) & (macro_is_defensive == True),
    1, 
    0 
)

# Shift signal by 1 day (trade tomorrow)
data['Signal'] = data['Signal'].shift(1)

# Align returns with the Signal (drop the rows where Signal is NaN due to SMA calculation)
aligned_data = data.dropna()
aligned_returns = returns.loc[aligned_data.index]

# ==========================================
# 6. PERFORMANCE CALCULATION
# ==========================================
basket_returns_mean = aligned_returns[BASKET_TICKERS].mean(axis=1)
spy_returns = aligned_returns[BENCHMARK]
signal = aligned_data['Signal']

# Strategy Returns
strat_returns = (signal * basket_returns_mean) + ((1 - signal) * spy_returns)

# Cumulative Returns
cum_strat = (1 + strat_returns).cumprod()
cum_spy = (1 + spy_returns).cumprod()

# ==========================================
# 7. YEARLY BREAKDOWN REPORT
# ==========================================
analysis_df = pd.DataFrame({'Strategy': strat_returns, 'SP500': spy_returns})
yearly_perf = analysis_df.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100

print("\n" + "="*50)
print(f" YEARLY PERFORMANCE (WITH TREND OVERRIDE)")
print("="*50)
print(f"{'Year':<10} | {'Strategy':<15} | {'S&P 500':<15} | {'Diff'}")
print("-" * 55)

for index, row in yearly_perf.iterrows():
    year = index.year
    strat = row['Strategy']
    sp500 = row['SP500']
    diff = strat - sp500
    
    # Highlight specific years
    marker = ""
    if year == 2020: marker = "<-- CHECK THIS"
    if year == 2022: marker = "<-- CHECK THIS"
    
    print(f"{year:<10} | {strat:>14.2f}% | {sp500:>14.2f}% | {diff:>+6.2f}% {marker}")

print("-" * 55)

# Final Stats
total_ret_strat = (cum_strat.iloc[-1] - 1) * 100
total_ret_spy = (cum_spy.iloc[-1] - 1) * 100
print(f"\nTotal Return: Strategy {total_ret_strat:.2f}% vs SPY {total_ret_spy:.2f}%")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(cum_strat, label='Trend + Macro Strategy', color='purple', linewidth=2)
plt.plot(cum_spy, label='S&P 500', color='gray', linestyle='--')
plt.title('Trend Override: Fixing the 2020 Lag')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()