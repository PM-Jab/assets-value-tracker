import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

TICKER = "AAPL"
DATA_FILE = f"{TICKER}_processed.csv"

def train_and_evaluate():
    print(f"Loading {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
        df.set_index('time', inplace=True)
    except FileNotFoundError:
        print("Run process_data.py first!")
        return

    # --- UPDATE: Added 'return_1d' and 'return_5d' ---
    feature_cols = [
        'dist_sma50', 'dist_sma200', 'RSI', 'bb_width', 'vol_change',
        'return_1d', 'return_5d'
    ]
    
    X = df[feature_cols]
    y = df['target']

    # Split Data (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training on {len(X_train)} days. Testing on {len(X_test)} days.")

    # Train Random Forest (Adjusted Parameters)
    # min_samples_leaf=5 prevents the model from obsessing over single outliers
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict & Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print("\n" + "="*30)
    print(f"MODEL ACCURACY (5-Day Horizon): {acc:.2%}")
    print("="*30)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nFeature Importance:")
    for name, imp in zip(feature_cols, model.feature_importances_):
        print(f"{name}: {imp:.4f}")

    joblib.dump(model, f"{TICKER}_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()