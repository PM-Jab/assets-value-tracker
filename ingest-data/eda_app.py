import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Market Trend EDA")

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

# --- Main App ---
st.title("📈 Asset Trading Analysis: Visual EDA")

# 1. Load Data
try:
    # We use the file we generated in the previous step
    df = load_data("AAPL_processed.csv")
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("File 'AAPL_processed.csv' not found. Please run process_data.py first.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("Chart Settings")
# Allow user to slice data (e.g., look at just the last year)
days_to_show = st.sidebar.slider("Days to Visualize", min_value=30, max_value=len(df), value=365)
sliced_df = df.tail(days_to_show)

# 3. Candlestick & Indicator Chart (The Main View)
st.subheader(f"Price Action & Indicators (Last {days_to_show} days)")

# Create a chart with 2 rows (Price on top, RSI on bottom)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

# Row 1: Candlestick
fig.add_trace(go.Candlestick(x=sliced_df.index,
                             open=sliced_df['open'], high=sliced_df['high'],
                             low=sliced_df['low'], close=sliced_df['close'],
                             name='OHLC'), row=1, col=1)

# Overlay SMAs
fig.add_trace(go.Scatter(x=sliced_df.index, y=sliced_df['SMA_50'], 
                         line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
fig.add_trace(go.Scatter(x=sliced_df.index, y=sliced_df['SMA_200'], 
                         line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)

# Row 2: RSI
fig.add_trace(go.Scatter(x=sliced_df.index, y=sliced_df['RSI'], 
                         line=dict(color='purple', width=1), name='RSI'), row=2, col=1)

# Add RSI 70/30 lines
fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")

# Update layout to remove range slider and customize dark theme
fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# 4. Correlation Heatmap (The "Scientific" Check)
st.subheader("Feature Correlation Matrix")
st.write("Do our indicators actually correlate with the Target (Next Day Up/Down)?")

# Select columns to correlate
corr_cols = ['close', 'volume', 'SMA_50', 'SMA_200', 'RSI', 'target']
corr = sliced_df[corr_cols].corr()

# Plot using Seaborn/Matplotlib
fig_corr, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# 5. Raw Data Preview
st.subheader("Raw Data View")
st.dataframe(sliced_df.tail())