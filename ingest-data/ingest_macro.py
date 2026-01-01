import pandas as pd
import pandas_datareader.data as web
import datetime
import time
from sqlalchemy import create_engine, text

# --- Configuration ---
DB_URI = 'postgresql://time:mysecretpassword@localhost:6002/time'

def init_db(engine):
    print("Initializing Database...")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS macro_data (
        time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        fed_rate DOUBLE PRECISION,
        yield_curve DOUBLE PRECISION,
        inflation_cpi DOUBLE PRECISION,
        unemployment DOUBLE PRECISION,
        PRIMARY KEY (time)
    );
    """
    timescale_sql = "SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);"

    with engine.connect() as conn:
        # Step 1: Create Table & COMMIT IMMEDIATELY
        # This saves the table regardless of what happens next.
        conn.execute(text(create_table_sql))
        conn.commit()
        
        # Step 2: Try TimescaleDB (Separate Transaction)
        try:
            conn.execute(text(timescale_sql))
            conn.commit()
            print("Converted macro_data to Hypertable.")
        except Exception:
            # If this fails, we don't care. The table already exists safely.
            print("Note: TimescaleDB extension not active (Standard Postgres table used).")

def fetch_macro_data():
    """Fetches data with RETRY logic to handle network timeouts."""
    print("Fetching data from Federal Reserve (FRED)...")
    
    # 1. Define Date Range
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    
    # 2. Define Indicators
    indicators = {
        'DFF': 'fed_rate',           
        'T10Y2Y': 'yield_curve',     
        'CPIAUCSL': 'inflation_cpi', 
        'UNRATE': 'unemployment'     
    }
    
    # 3. Retry Loop (The Fix)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}...")
            
            # Fetch data
            df = web.DataReader(list(indicators.keys()), 'fred', start, end)
            
            # Process data
            df.rename(columns=indicators, inplace=True)
            df.ffill(inplace=True)
            df.index.name = 'time'
            df.reset_index(inplace=True)
            
            print(f"Success! Fetched {len(df)} rows.")
            return df
            
        except Exception as e:
            print(f"Connection failed ({e}). Retrying in 5 seconds...")
            time.sleep(5)
            
    print("❌ Failed to fetch data after multiple attempts.")
    return pd.DataFrame()

def save_to_db(df, engine):
    if df.empty: return

    print("Saving to database...")
    with engine.connect() as conn:
        # 1. Temp Table
        df.to_sql('temp_macro', conn, if_exists='replace', index=False)
        
        # 2. Upsert (Update if exists, Insert if new)
        upsert_sql = """
        INSERT INTO macro_data (time, fed_rate, yield_curve, inflation_cpi, unemployment)
        SELECT time, fed_rate, yield_curve, inflation_cpi, unemployment
        FROM temp_macro
        ON CONFLICT (time) DO UPDATE SET
            fed_rate = EXCLUDED.fed_rate,
            yield_curve = EXCLUDED.yield_curve,
            inflation_cpi = EXCLUDED.inflation_cpi,
            unemployment = EXCLUDED.unemployment;
        """
        conn.execute(text(upsert_sql))
        conn.execute(text("DROP TABLE temp_macro;"))
        conn.commit()
        
    print("Success! Macro data stored.")

if __name__ == "__main__":
    engine = create_engine(DB_URI)
    init_db(engine)
    df = fetch_macro_data()
    save_to_db(df, engine)