import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# Model Definition
class StackedLTC(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1, num_layers=2):
        super(StackedLTC, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load model with error handling
def load_model():
    model = StackedLTC()
    try:
        if os.path.exists("lnn_final_model.pth"):
            model.load_state_dict(torch.load("lnn_final_model.pth", map_location="cpu"))
            model.eval()
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model file not found. Using untrained model.")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
    return model

model = load_model()

# Load Data
df = pd.read_csv("TCS_2020_present.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna().reset_index(drop=True)

# Normalize OHLCV
feature_cols = ["Open", "High", "Low", "Close", "Volume"]
data = df[feature_cols].values
mean, std = data.mean(axis=0), data.std(axis=0)
std[std == 0] = 1  # Prevent division by zero
scaled = (data - mean) / std

# Trading Simulator
class TradingSimulator:
    def __init__(self, initial_cash=100000, window_size=30):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares = 0
        self.history = []
        self.window_size = window_size

    def step(self, step_idx):
        if step_idx < self.window_size:
            return None

        # Prepare input
        window = scaled[step_idx - self.window_size: step_idx]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            pred = model(x).item()

        price = data[step_idx][3]  # Close price
        action = "HOLD"

        # Trading Logic
        if pred > 0.05 and self.cash >= price:
            self.shares += 1
            self.cash -= price
            action = "BUY"
        elif pred < -0.05 and self.shares > 0:
            self.shares -= 1
            self.cash += price
            action = "SELL"

        portfolio = self.cash + self.shares * price

        result = {
            "Step": step_idx,
            "Date": str(df['Date'][step_idx].date()),
            "Price": float(price),
            "Pred": float(pred),
            "Action": action,
            "Cash": round(self.cash, 2),
            "Portfolio": round(portfolio, 2)
        }
        self.history.append(result)
        return result

def run_trading():
    sim = TradingSimulator()
    results = []
    for i in range(len(df)):
        step = sim.step(i)
        if step:
            results.append(step)
    return pd.DataFrame(results)
