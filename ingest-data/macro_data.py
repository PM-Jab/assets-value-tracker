import pandas_datareader.data as web
import pandas as pd
import datetime

def get_fed_data(start_date="2020-01-01"):
    """
    Fetches official Macroeconomic data from FRED (Federal Reserve).
    1. FEDFUNDS: The Interest Rate (The cost of money)
    2. T10Y2Y: The Yield Curve (Predicts recessions)
    """
    print("Fetching Fed Rates & Yield Curve...")
    
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.now()
    
    # 'fred' is the data source for the Federal Reserve
    try:
        # DFF = Daily Federal Funds Rate
        fed_rate = web.DataReader('DFF', 'fred', start, end)
        
        # T10Y2Y = 10-Year Treasury Minus 2-Year Treasury
        yield_curve = web.DataReader('T10Y2Y', 'fred', start, end)
        
        # Combine
        macro_df = pd.concat([fed_rate, yield_curve], axis=1)
        macro_df.columns = ['fed_rate', 'yield_curve']
        
        # Forward fill because macro data doesn't change every minute
        macro_df.ffill(inplace=True)
        
        return macro_df
        
    except Exception as e:
        print(f"Error fetching Macro Data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_fed_data()
    print(df.tail())