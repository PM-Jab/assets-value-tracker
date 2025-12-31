import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
import time
import requests
import io  # Add this import at the top of your file

# --- Configuration ---
DB_USER = 'time'
DB_PASS = 'mysecretpassword'
DB_HOST = 'localhost'
DB_PORT = '6002'
DB_NAME = 'time'

# Connection String
DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

def get_sp500_tickers():
    """Scrapes the list of S&P 500 tickers from Wikipedia using a browser header."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Fake a browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    try:
        # 1. Get the HTML content manually
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check if the request was actually successful
        
        # 2. Parse it with Pandas (wrapping raw text in StringIO/BytesIO avoids warnings)
        table = pd.read_html(io.StringIO(response.text))
        
        df = table[0]
        tickers = df['Symbol'].tolist()
        
        # Clean tickers (e.g., BRK.B -> BRK-B for yfinance)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers

    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return []

def init_db(engine):
    """Creates the table and converts it to a TimescaleDB hypertable."""
    with engine.connect() as conn:
        # 1. Create standard table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                time TIMESTAMPTZ NOT NULL,
                ticker TEXT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                PRIMARY KEY (time, ticker)
            );
        """))
        
        # 2. Convert to Hypertable (TimescaleDB magic)
        # We wrap this in a try/except because it fails if it's already a hypertable
        try:
            conn.execute(text("SELECT create_hypertable('stock_prices', 'time', if_not_exists => TRUE);"))
            conn.commit()
            print("Database initialized and Hypertable created.")
        except Exception as e:
            print(f"Hypertable creation note: {e}")
            conn.rollback()

def fetch_and_store_data(engine, tickers):
    """Fetches 5y data for each ticker and stores it."""
    
    total = len(tickers)
    print(f"Starting ingestion for {total} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            print(f"[{i+1}/{total}] Processing {ticker}...", end=" ")
            
            # 1. Fetch Data
            # auto_adjust=True handles stock splits automatically
            hist = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            
            if hist.empty:
                print("No data found.")
                continue

            # 2. Format Data for DB
            # Reset index to make 'Date' a column
            hist = hist.reset_index()
            
            # Ensure columns match our DB schema
            # yfinance returns: Date, Open, High, Low, Close, Volume
            df_insert = pd.DataFrame()
            df_insert['time'] = hist['Date']
            df_insert['ticker'] = ticker
            df_insert['open'] = hist['Open']
            df_insert['high'] = hist['High']
            df_insert['low'] = hist['Low']
            df_insert['close'] = hist['Close']
            df_insert['volume'] = hist['Volume']

            # 3. Insert into DB
            # 'append' adds to existing data
            df_insert.to_sql('stock_prices', engine, if_exists='append', index=False, method='multi', chunksize=1000)
            
            print("Done.")
            
            # Sleep briefly to be nice to the API
            time.sleep(0.5)

        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    # 1. Connect to Database
    engine = create_engine(DATABASE_URI)
    
    # 2. Setup Schema
    init_db(engine)
    
    # 3. Get Tickers
    sp500_tickers = get_sp500_tickers()
    print(f"Found {len(sp500_tickers)} tickers.")
    
    # 4. Run Ingestion
    if sp500_tickers:
        fetch_and_store_data(engine, sp500_tickers)