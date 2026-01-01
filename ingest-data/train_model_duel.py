import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================
OFFENSE_PROXY = 'QQQ'  # Nasdaq 100
DEFENSE_PROXY = 'XLU'  # Utilities
BENCHMARK = '^GSPC'    # S&P 500
MACRO_TICKERS = ['^IRX', '^TNX'] 

START_DATE = '1999-01-01' # Maximum history
END_DATE = '2025-12-30'
PREDICTION_WINDOW = 21 # Predict the winner over the next MONTH (21 trading days)

print("Step 1: Downloading Training Data...")
tickers = [OFFENSE_PROXY, DEFENSE_PROXY, BENCHMARK] + MACRO_TICKERS
data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

if 'Close' in data.columns.levels[0]:
    data = data['Close']
data = data.ffill().dropna()

# ==========================================
# 2. FEATURE ENGINEERING (Improved)
# ==========================================
print("Step 2: Creating Features...")
df = pd.DataFrame(index=data.index)

# A. Macro Features
df['Yield_Curve'] = data['^TNX'] - data['^IRX']
df['Fed_Rate'] = data['^IRX']
df['Rate_Vol'] = data['^IRX'].rolling(60).std() # Longer window for rate stability

# B. Market Technicals
# Distance from 200 SMA (normalized)
sma200 = data[BENCHMARK].rolling(200).mean()
df['Dist_SMA200'] = (data[BENCHMARK] - sma200) / sma200

# Recent Momentum (3 Month)
df['Mom_3M'] = data[BENCHMARK].pct_change(63)

# Volatility Ratio (Is volatility rising?)
df['Vol_Ratio'] = data[BENCHMARK].rolling(21).std() / data[BENCHMARK].rolling(63).std()

# ==========================================
# 3. LABEL CREATION (The Target)
# ==========================================
# Calculate returns over the NEXT 21 days
fwd_ret_offense = data[OFFENSE_PROXY].shift(-PREDICTION_WINDOW) / data[OFFENSE_PROXY] - 1
fwd_ret_defense = data[DEFENSE_PROXY].shift(-PREDICTION_WINDOW) / data[DEFENSE_PROXY] - 1

# Target: 1 if Offense wins significantly (> 1% diff), else 0 (Defense)
# We add a small buffer (0.01) so we don't switch for tiny random noise
df['Target'] = np.where(fwd_ret_offense > (fwd_ret_defense + 0.01), 1, 0)

df = df.dropna()

# ==========================================
# 4. TRAINING (TimeSeries Split)
# ==========================================
print("Step 3: Training with Time-Series Validation...")

feature_cols = ['Yield_Curve', 'Fed_Rate', 'Rate_Vol', 'Dist_SMA200', 'Mom_3M', 'Vol_Ratio']
X = df[feature_cols]
y = df['Target']

# Use the last 3 years as the Test Set
test_size = 252 * 3 
X_train = X.iloc[:-test_size]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:-test_size]
y_test = y.iloc[-test_size:]

# Train
model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATION
# ==========================================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\nModel Training Complete.")
print(f"Prediction Window: {PREDICTION_WINDOW} Days")
print(f"Test Set Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nFeature Importance:")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)

joblib.dump(model, 'dual_regime_model_v2.pkl')
print("\nModel saved as 'dual_regime_model_v2.pkl'")