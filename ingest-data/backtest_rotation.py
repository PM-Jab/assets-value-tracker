import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASKET_TICKERS = ['EVRG', 'FAST', 'DTE', 'MO', 'LNT', 'NI', 'KO', 'HSY']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['^IRX', '^TNX']

START_DATE = '2019-01-01'
END_DATE = '2025-12-30'

def calculate_max_drawdown(series):
    """Calculates the Maximum Drawdown (Risk) of a price series."""
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown.min(), drawdown

# ==========================================
# 2. DATA INGESTION (FIXED)
# ==========================================
print(f"Downloading data for Basket + {BENCHMARK} + Macro indicators...")
all_tickers = BASKET_TICKERS + [BENCHMARK] + MACRO_TICKERS

# FIX: We use auto_adjust=True. 
# This makes the 'Close' column contain the Dividend/Split adjusted price.
# This avoids the 'Adj Close' KeyError issues.
raw_data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

# Check if we have a MultiIndex (Price, Ticker) or just (Ticker)
# If 'Close' is a level, we select it.
if 'Close' in raw_data.columns.levels[0]:
    data = raw_data['Close']
else:
    # Fallback for some versions where it might be flat or different
    data = raw_data

# Ensure all columns are numeric and handle missing data
data = data.apply(pd.to_numeric, errors='coerce')
data = data.ffill().dropna()

# ==========================================
# 3. MACRO LOGIC & SIGNAL GENERATION
# ==========================================
# Calculate Yield Curve (10Y - 13Week)
data['Yield_Curve'] = data['^TNX'] - data['^IRX']

# Calculate Rate Volatility (Standard Deviation of Short Term Rates over 30 days)
data['Rate_Vol'] = data['^IRX'].rolling(window=30).std()

# Create Signal
# 1 = Defensive Basket (Yield Curve Positive AND Rates Stable)
# 0 = S&P 500 (Growth/Risk Mode)
data['Signal'] = np.where(
    (data['Yield_Curve'] > 0) & (data['Rate_Vol'] < 0.5), 
    1, 
    0 
)

# Shift signal by 1 day (we trade tomorrow based on today's close)
data['Signal'] = data['Signal'].shift(1)

# ==========================================
# 4. BACKTEST EXECUTION
# ==========================================
returns = data.pct_change()

# Basket Performance (Equal Weight)
basket_returns = returns[BASKET_TICKERS].mean(axis=1)

# Benchmark Performance
spy_returns = returns[BENCHMARK]

# Strategy Performance:
# If Signal is 1, use Basket Returns. If 0, use SPY Returns.
strat_returns = (data['Signal'] * basket_returns) + ((1 - data['Signal']) * spy_returns)

# Cumulative Returns (Growth of $1)
cum_strat = (1 + strat_returns).cumprod()
cum_spy = (1 + spy_returns).cumprod()
cum_basket = (1 + basket_returns).cumprod()

# ==========================================
# 5. RISK ANALYSIS (DRAWDOWNS)
# ==========================================
max_dd_strat, dd_curve_strat = calculate_max_drawdown(cum_strat)
max_dd_spy, dd_curve_spy = calculate_max_drawdown(cum_spy)

# ==========================================
# 6. RESULTS & REPORTING
# ==========================================
total_ret_strat = (cum_strat.iloc[-1] - 1) * 100
total_ret_spy = (cum_spy.iloc[-1] - 1) * 100

print("\n" + "="*40)
print(" MACRO ROTATION BACKTEST RESULTS")
print("="*40)
print(f"Strategy Total Return:  {total_ret_strat:.2f}%")
print(f"S&P 500 Total Return:   {total_ret_spy:.2f}%")
print("-" * 40)
print(f"Strategy Max Drawdown:  {max_dd_strat:.2%}")
print(f"S&P 500 Max Drawdown:   {max_dd_spy:.2%}")
print("-" * 40)

# Win Rate calculation
wins = strat_returns[strat_returns > 0].count()
total_days = strat_returns.count()
print(f"Daily Win Rate:         {(wins/total_days):.2%}")

current_mode = "DEFENSIVE BASKET" if data['Signal'].iloc[-1] == 1 else "S&P 500"
print(f"Current Signal (Today): {current_mode}")

# ==========================================
# 7. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

# Top Plot: Cumulative Returns
ax1.plot(cum_strat, label='Macro Rotation Strategy', color='green', linewidth=2)
ax1.plot(cum_spy, label='S&P 500 (Benchmark)', color='gray', linestyle='--', alpha=0.7)
ax1.set_title('Performance: Defensive Rotation vs S&P 500')
ax1.set_ylabel('Growth of $1')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Plot: Drawdowns (Risk)
ax2.plot(dd_curve_strat, label='Strategy Drawdown', color='green', linewidth=1)
ax2.plot(dd_curve_spy, label='S&P 500 Drawdown', color='red', linewidth=1, alpha=0.5)
ax2.fill_between(dd_curve_spy.index, dd_curve_spy, color='red', alpha=0.1)
ax2.set_title('Risk Profile: Drawdowns')
ax2.set_ylabel('Drawdown %')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()