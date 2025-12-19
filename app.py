import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from backend import run_trading

st.set_page_config(layout="wide")
st.title("📈 Simulated Trading with Liquid Neural Network (TCS.NS)")

# Run backend
results = run_trading()
df = pd.read_csv("TCS_2020_present.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Merge with results
merged = df.copy()
merged = merged.iloc[-len(results):].reset_index(drop=True)
merged = pd.concat([merged, results.reset_index(drop=True)], axis=1)

# Candlestick Chart
fig = go.Figure(data=[go.Candlestick(
    x=merged['Date'],
    open=merged['Open'],
    high=merged['High'],
    low=merged['Low'],
    close=merged['Close'],
    name="Candlesticks"
)])

# Plot Buy/Sell
buy_signals = merged[merged['Action'] == 'BUY']
sell_signals = merged[merged['Action'] == 'SELL']

fig.add_trace(go.Scatter(
    x=buy_signals['Date'], y=buy_signals['Close'],
    mode='markers', marker=dict(symbol="triangle-up", color="green", size=12),
    name="Buy Signal"
))

fig.add_trace(go.Scatter(
    x=sell_signals['Date'], y=sell_signals['Close'],
    mode='markers', marker=dict(symbol="triangle-down", color="red", size=12),
    name="Sell Signal"
))

fig.update_layout(title="Trading Simulation on TCS", xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# Metrics
st.subheader("📊 Trading Results")
st.dataframe(results.tail(20))

final_value = results["Portfolio"].iloc[-1]
initial_value = 100000
profit = final_value - initial_value
profit_pct = (profit / initial_value) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Final Portfolio Value", f"₹{final_value:,.2f}")
col2.metric("Total Profit/Loss", f"₹{profit:,.2f}", f"{profit_pct:.2f}%")
col3.metric("Total Trades", len(results[results['Action'] != 'HOLD']))
