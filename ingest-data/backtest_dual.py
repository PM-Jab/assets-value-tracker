import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. SETUP BASKETS & MODEL
# ==========================================
# 1. THE SWORD (Offense / Growth / Tech / Crypto)
OFFENSIVE_BASKET = [
    'NVDA', 'MSFT', 'META', 'AMZN', 'AMD', 'PLTR', 'COIN', 'IBIT'
]

# 2. THE SHIELD (Defense / Utilities / Staples)
DEFENSIVE_BASKET = [
    'EVRG', 'DTE', 'LNT', 'NI', 'SO', 'KO', 'PG', 'O'
]

BENCHMARK = 'SPY'
MACRO_TICKERS = ['^IRX', '^TNX', '^GSPC'] # Needed for model features

# Load the trained model
print("Loading Model...")
model = joblib.load('dual_regime_model_v2.pkl')

START_DATE = '2020-01-01' # Backtest period
END_DATE = '2025-12-30'

# ==========================================
# 2. FETCH DATA
# ==========================================
print("Fetching Market Data...")
all_tickers = OFFENSIVE_BASKET + DEFENSIVE_BASKET + [BENCHMARK] + MACRO_TICKERS
raw_data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

if 'Close' in raw_data.columns.levels[0]:
    data = raw_data['Close']
else:
    data = raw_data

data = data.ffill().dropna()

# ==========================================
# 3. RECREATE MODEL FEATURES
# ==========================================
# We must recreate the EXACT same features the model was trained on
features = pd.DataFrame(index=data.index)

features['Fed_Rate'] = data['^IRX']
features['Yield_Curve'] = data['^TNX'] - data['^IRX']
features['Rate_Vol_30'] = data['^IRX'].rolling(window=30).std()
features['Rate_Change_30'] = data['^IRX'].diff(30)
features['SP500_Ret_1M'] = data['^GSPC'].pct_change(21)
features['SP500_Vol_1M'] = data['^GSPC'].rolling(21).std()
features['Above_SMA200'] = np.where(data['^GSPC'] > data['^GSPC'].rolling(200).mean(), 1, 0)

# Drop NaN values (model cannot handle NaNs)
valid_data = features.dropna()
data = data.loc[valid_data.index] # Align price data with valid features

# ==========================================
# 4. PREDICT REGIMES
# ==========================================
print("Predicting Regimes...")
# 1 = Offense, 0 = Defense
valid_data['Signal'] = model.predict(valid_data)

# Shift signal by 1 day (Trade tomorrow based on today's data)
valid_data['Signal'] = valid_data['Signal'].shift(1)

# ==========================================
# 5. CALCULATE PERFORMANCE
# ==========================================
returns = data.pct_change()

# Calculate Basket Returns (Equal Weight)
offense_ret = returns[OFFENSIVE_BASKET].mean(axis=1)
defense_ret = returns[DEFENSIVE_BASKET].mean(axis=1)
spy_ret = returns[BENCHMARK]

# Strategy Logic:
# If Signal == 1: Use Offense Returns
# If Signal == 0: Use Defense Returns
strat_ret = (valid_data['Signal'] * offense_ret) + ((1 - valid_data['Signal']) * defense_ret)

# Cumulative Returns
cum_strat = (1 + strat_ret).cumprod()
cum_spy = (1 + spy_ret).cumprod()
cum_offense = (1 + offense_ret).cumprod()
cum_defense = (1 + defense_ret).cumprod()

# ==========================================
# 6. REPORTING
# ==========================================
total_ret_strat = (cum_strat.iloc[-1] - 1) * 100
total_ret_spy = (cum_spy.iloc[-1] - 1) * 100

print("\n" + "="*40)
print(" DUAL BASKET AI STRATEGY RESULTS")
print("="*40)
print(f"Strategy Total Return:  {total_ret_strat:.2f}%")
print(f"S&P 500 Total Return:   {total_ret_spy:.2f}%")
print("-" * 40)

# Current Signal
last_signal = valid_data['Signal'].iloc[-1]
mode = "OFFENSE (SWORD)" if last_signal == 1 else "DEFENSE (SHIELD)"
print(f"CURRENT MODEL SIGNAL:   {mode}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(cum_strat, label='AI Dual Strategy', color='blue', linewidth=2)
plt.plot(cum_spy, label='S&P 500', color='gray', linestyle='--', alpha=0.5)
plt.plot(cum_offense, label='Offense Only (Hold)', color='green', alpha=0.3)
plt.plot(cum_defense, label='Defense Only (Hold)', color='red', alpha=0.3)

plt.title('AI Strategy: Switching between Offense & Defense')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()