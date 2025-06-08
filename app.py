import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

CSV_FILE = "data.csv"
MODEL_FILE = "prophet_model.pkl"

st.set_page_config(page_title="BTC Price Predictor", layout="wide")
st.title("📈 Bitcoin Price Forecast")

# Load data
if not os.path.exists(CSV_FILE):
    st.error(f"{CSV_FILE} not found.")
    st.stop()

df = pd.read_csv(CSV_FILE)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp", "Close"]).sort_values("Timestamp")

st.success(f"Loaded {len(df)} rows.")

# Load trained model
if not os.path.exists(MODEL_FILE):
    st.error(f"Trained model not found. Please run train.py first.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

st.success("Loaded trained Prophet model.")

# Input for forecast horizon in minutes
forecast_minutes = st.number_input(
    "Enter number of minutes to forecast into the future:", 
    min_value=1, max_value=1440, value=60, step=1
)

# Show latest data sample
st.subheader("Latest Price Data Sample")
st.dataframe(df.tail(5)[["Timestamp", "Close"]])

# Forecast
st.subheader(f"Forecast for next {forecast_minutes} minutes")

try:
    future = model.make_future_dataframe(periods=forecast_minutes, freq='min')
    forecast = model.predict(future)

    # Show only future forecast (beyond latest known data)
    last_date = df["Timestamp"].max()
    future_forecast = forecast[forecast['ds'] > last_date]

    st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Plot forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot trend and seasonality
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Forecasting error: {e}")

# Helper function to plot price fluctuations
def plot_fluctuations(df_sub, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_sub["Timestamp"], df_sub["Close"])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("📊 Price Fluctuations")

last_month = df[df["Timestamp"] >= (df["Timestamp"].max() - pd.Timedelta(days=30))]
if not last_month.empty:
    plot_fluctuations(last_month, "Last 30 Days Price Fluctuations")
else:
    st.write("No data for last month.")

last_day = df[df["Timestamp"] >= (df["Timestamp"].max() - pd.Timedelta(days=1))]
if not last_day.empty:
    plot_fluctuations(last_day, "Last 24 Hours Price Fluctuations")
else:
    st.write("No data for last day.")

last_year = df[df["Timestamp"] >= (df["Timestamp"].max() - pd.Timedelta(days=365))]
if not last_year.empty:
    plot_fluctuations(last_year, "Last Year Price Fluctuations")
else:
    st.write("No data for last year.")
