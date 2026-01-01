import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
import requests
import io

warnings.simplefilter('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILE = 'sp500_universal_model.pkl'
MACRO_TICKERS = ['^IRX', '^TNX', '^GSPC']

# Sector definitions
OFFENSE_SECTORS = ['Information Technology', 'Communication Services', 'Consumer Discretionary']
DEFENSE_SECTORS = ['Utilities', 'Consumer Staples', 'Health Care', 'Real Estate']

print("--------------------------------------------------")
print(" HYBRID INTELLIGENT SCANNER (SELF-HEALING)")
print("--------------------------------------------------")

# ==========================================
# 2. LOAD MODEL & DETECT FEATURES
# ==========================================
print("1. Loading AI Brain...")
try:
    model = joblib.load(MODEL_FILE)
    # AUTO-DETECT: Get the exact feature names the model wants
    if hasattr(model, 'feature_names_in_'):
        REQUIRED_FEATURES = list(model.feature_names_in_)
        print(f"   Model expects {len(REQUIRED_FEATURES)} features: {REQUIRED_FEATURES}")
    else:
        # Fallback if attribute missing (older sklearn versions)
        print("   Warning: Could not auto-detect features. Using default.")
        REQUIRED_FEATURES = ['RSI', 'bb_width', 'dist_sma200', 'dist_sma50', 'return_1d', 'return_5d', 'vol_change']
        
except FileNotFoundError:
    print(f"CRITICAL ERROR: '{MODEL_FILE}' not found.")
    exit()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def calculate_technicals(df):
    """Calculates all possible features so the model can pick what it needs."""
    df = df.copy()
    
    # 1. Price Momentum
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    
    # 2. Volume Change (Handle missing Volume)
    if 'Volume' in df.columns:
        df['vol_change'] = df['Volume'].pct_change()
    else:
        df['vol_change'] = 0
    
    # 3. Distance from SMAs
    df['sma50'] = df['Close'].rolling(window=50).mean()
    df['dist_sma50'] = (df['Close'] - df['sma50']) / df['sma50']
    
    df['sma200'] = df['Close'].rolling(window=200).mean()
    df['dist_sma200'] = (df['Close'] - df['sma200']) / df['sma200']
    
    # 4. Bollinger Band Width
    window = 20
    df['sma20'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['upper_bb'] = df['sma20'] + (2 * df['std'])
    df['lower_bb'] = df['sma20'] - (2 * df['std'])
    df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['sma20']
    
    # 5. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, 0.0001) 
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

# ==========================================
# 4. MACRO REGIME CHECK
# ==========================================
print("2. Checking Macro Regime...")
macro = yf.download(MACRO_TICKERS, period="6mo", progress=False, auto_adjust=True)
if 'Close' in macro.columns.levels[0]: macro = macro['Close']

yield_curve = macro['^TNX'].iloc[-1] - macro['^IRX'].iloc[-1]

if yield_curve > -0.5: 
    regime = "OFFENSE"
    target_sectors = OFFENSE_SECTORS
else:
    regime = "DEFENSE"
    target_sectors = DEFENSE_SECTORS

print(f"   Yield Curve: {yield_curve:.2f} -> {regime} MODE")

# ==========================================
# 5. FETCH CANDIDATES
# ==========================================
print("3. Fetching Candidate List...")
candidates = ['NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'PLTR', 'COIN', 'IBIT']

try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        df_sp500 = pd.read_html(io.BytesIO(response.content))[0]
        sector_map = dict(zip(df_sp500['Symbol'], df_sp500['GICS Sector']))
        all_tickers = [t.replace('.', '-') for t in df_sp500['Symbol'].tolist()]
        candidates = [t for t in all_tickers if sector_map.get(t.replace('-', '.'), '') in target_sectors]
        if regime == "OFFENSE": candidates += ['COIN', 'IBIT', 'MSTR', 'PLTR']
        print(f"   Scanning {len(candidates)} stocks...")
    else:
        print("   Wiki Fetch Failed. Using fallback.")
except Exception as e:
    print(f"   Wiki Error: {e}. Using fallback.")

# ==========================================
# 6. SCAN & PREDICT
# ==========================================
print("4. Calculating & Predicting...")
results = []
chunk_size = 20
scan_list = candidates[:150] # Remove limit for full scan

for i in range(0, len(scan_list), chunk_size):
    batch = scan_list[i:i+chunk_size]
    print(f"   Processing batch {i}...")
    
    try:
        data = yf.download(batch, period="2y", progress=False, auto_adjust=True)
        if len(batch) == 1: data = pd.concat({batch[0]: data}, axis=1)
        
        # Handle MultiIndex
        if 'Close' in data.columns.levels[0]: 
            close_data = data['Close']
            vol_data = data['Volume'] if 'Volume' in data.columns.levels[0] else None
        else:
            close_data = data
            vol_data = None

        for ticker in batch:
            try:
                if ticker not in close_data.columns: continue
                
                # Build single stock DF
                stock_df = pd.DataFrame({'Close': close_data[ticker]})
                if vol_data is not None and ticker in vol_data.columns:
                    stock_df['Volume'] = vol_data[ticker]
                
                if len(stock_df) < 200: continue

                # Calculate
                tech_df = calculate_technicals(stock_df)
                
                if tech_df is not None and len(tech_df) > 0:
                    # DYNAMICALLY SELECT FEATURES
                    # This line fixes the order error automatically
                    if all(feat in tech_df.columns for feat in REQUIRED_FEATURES):
                        feats = tech_df.iloc[[-1]][REQUIRED_FEATURES]
                        
                        score = model.predict_proba(feats)[0][1]
                        
                        results.append({
                            'Ticker': ticker,
                            'AI_Score': score,
                            'RSI': tech_df['RSI'].iloc[-1],
                            'Trend': tech_df['dist_sma200'].iloc[-1]
                        })
                    else:
                        missing = [f for f in REQUIRED_FEATURES if f not in tech_df.columns]
                        print(f"     {ticker} missing features: {missing}")

            except Exception as inner_e:
                # print(f"Error {ticker}: {inner_e}")
                continue
                
    except Exception as e:
        print(f"   Batch Error: {e}")
        continue

# ==========================================
# 7. RESULTS
# ==========================================
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='AI_Score', ascending=False).head(20)

    print("\n" + "="*65)
    print(f" TOP AI PICKS FOR {regime} REGIME")
    print("="*65)
    print(f"{'Ticker':<8} | {'AI Conf':<10} | {'RSI':<6} | {'Trend (vs 200SMA)'}")
    print("-" * 65)
    for index, row in results_df.iterrows():
        print(f"{row['Ticker']:<8} | {row['AI_Score']:>7.2%}   | {row['RSI']:>4.0f}   | {row['Trend']:>+8.2%}")
    print("-" * 65)
else:
    print("\nNo results found.")