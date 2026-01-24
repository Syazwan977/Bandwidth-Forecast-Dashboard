# ===========================
# FIXED VERSION (WHAT YOU WANTED)
# 1) Train = 5 days, Test = 2 days (hourly)
# 2) Add "Next Weekend Forecast" chart using current week pattern
# ===========================

import io
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib.dates import AutoDateLocator, DateFormatter
from prophet import Prophet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


# 1. HELPER FUNCTIONS -----------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from a file path and set Timestamp as index."""
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
    else:
        raise ValueError("No 'Timestamp' column found in the CSV.")
    return df


def prepare_time_series(df: pd.DataFrame, target_col: str = "Required_Bandwidth") -> pd.Series:
    """Return hourly time series of target column (from 1-minute data)."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in data.")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    ts_min = df[target_col].asfreq("min").interpolate()
    ts_hourly = ts_min.resample("h").mean().interpolate()
    return ts_hourly


def train_test_split_fixed(ts: pd.Series, train_days: int = 5, test_days: int = 2):
    """
    Fixed split:
    - Train = first 5 days
    - Test  = last 2 days
    """
    points_per_day = 24
    train_points = train_days * points_per_day
    test_points = test_days * points_per_day

    if len(ts) < (train_points + test_points):
        raise ValueError(
            f"Not enough data. Need at least {train_days + test_days} days "
            f"({train_points + test_points} hourly points). Got {len(ts)} points."
        )

    ts_7days = ts.iloc[-(train_points + test_points):]  # ensure we use last 7 days only
    train = ts_7days.iloc[:train_points]
    test = ts_7days.iloc[train_points:]
    return train, test


def eval_metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100
    return rmse, mae, mape, r2


def compute_next_weekend_index(last_timestamp: pd.Timestamp):
    """
    Build an hourly DatetimeIndex for the NEXT Saturday & Sunday (48 hours).
    """
    anchor = last_timestamp.floor("h")
    days_ahead = (5 - anchor.dayofweek) % 7  # 5 = Saturday
    if days_ahead == 0:
        days_ahead = 7  # ensure "next" Saturday, not same day
    next_sat = (anchor + pd.Timedelta(days=days_ahead)).normalize()
    next_weekend = pd.date_range(start=next_sat, periods=48, freq="h")
    return next_weekend


# 2. MODELS ---------------------------------------------------------------


def run_arima(train, steps):
    model = sm.tsa.ARIMA(train, order=(2, 1, 2))
    res = model.fit()
    forecast = res.forecast(steps=steps)
    return forecast


def run_sarima(train, steps, max_train_points=4000):
    train_sub = train if len(train) <= max_train_points else train.iloc[-max_train_points:]

    # default seasonal period for hourly = 24
    seasonal_period = 24

    model = SARIMAX(
        train_sub,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    forecast_vals = res.forecast(steps=steps)
    return forecast_vals


def run_prophet(train, steps):
    df_train = train.reset_index()
    df_train.columns = ["ds", "y"]

    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df_train)

    future = m.make_future_dataframe(periods=steps, freq="h")
    forecast_full = m.predict(future)
    return forecast_full.iloc[-steps:]["yhat"].values


def run_lstm(train, steps, lookback=24):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    def create_sequences(data, lookback_):
        X, y = [], []
        for i in range(len(data) - lookback_):
            X.append(data[i:i + lookback_])
            y.append(data[i + lookback_])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, lookback)

    model = Sequential()
    model.add(LSTM(64, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )

    # recursive forecasting for "steps" hours
    history = train_scaled.copy()
    preds = []

    for _ in range(steps):
        x_in = history[-lookback:].reshape(1, lookback, 1)
        yhat = model.predict(x_in, verbose=0)[0][0]
        preds.append(yhat)
        history = np.vstack([history, [[yhat]]])

    preds = np.array(preds).reshape(-1, 1)
    forecast = scaler.inverse_transform(preds).flatten()
    return forecast


# 4. STREAMLIT APP --------------------------------------------------------

st.set_page_config(page_title="Bandwidth Forecast Dashboard", layout="wide")


def main():
    st.title("Bandwidth Forecast Dashboard (Single Location)")

    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload QoS CSV file", type=["csv"])
    data_path = st.sidebar.text_input(
        "OR CSV file path (if not uploading)",
        "expandedBootstrapping_quality_of_service_5g.csv",
    )

    # fixed: train 5 days, test 2 days
    train_days = 5
    test_days = 2
    st.sidebar.info("Split fixed:\n- Train = 5 days\n- Test = 2 days")

    # Load data ------------------------------------------------
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "Timestamp" not in df.columns:
                st.error("Uploaded file must contain a 'Timestamp' column.")
                return
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
        else:
            df = load_data(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Location filter ------------------------------------------
    selected_location = None
    if "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
        if len(locations) > 0:
            default_idx = locations.index("Location X") if "Location X" in locations else 0
            selected_location = st.sidebar.selectbox("Select Location", locations, index=default_idx)
            df = df[df["Location"] == selected_location]
            if df.empty:
                st.error(f"No data available for location: {selected_location}")
                return
            st.sidebar.success(f"Filtering data for location: {selected_location}")
        else:
            st.sidebar.warning("Location column is present but empty.")
    else:
        st.sidebar.warning("No 'Location' column found. Using all data.")

    st.sidebar.write("Columns found:", list(df.columns))

    # Prepare time series --------------------------------------
    try:
        ts_hourly = prepare_time_series(df, target_col="Required_Bandwidth")
        train, test = train_test_split_fixed(ts_hourly, train_days=train_days, test_days=test_days)
    except Exception as e:
        st.error(f"Error preparing time series: {e}")
        return

    st.subheader(f"Time Range — {selected_location if selected_location else 'All'}")
    st.write(f"Train (5 days): {train.index[0]} -> {train.index[-1]}")
    st.write(f"Test  (2 days): {test.index[0]} -> {test.index[-1]}")

    # Train models ---------------------------------------------
    st.markdown("## Model Training and Selection")

    # --- TEST FORECASTS (2 days / 48 hours)
    steps_test = len(test)

    arima_pred = pd.Series(run_arima(train, steps_test).values, index=test.index)
    sarima_pred = pd.Series(run_sarima(train, steps_test).values, index=test.index)
    prophet_pred = pd.Series(run_prophet(train, steps_test), index=test.index)
    lstm_pred = pd.Series(run_lstm(train, steps_test), index=test.index)

    arima_metrics = dict(zip(["RMSE", "MAE", "MAPE", "R2"], eval_metrics(test.values, arima_pred.values)))
    sarima_metrics = dict(zip(["RMSE", "MAE", "MAPE", "R2"], eval_metrics(test.values, sarima_pred.values)))
    prophet_metrics = dict(zip(["RMSE", "MAE", "MAPE", "R2"], eval_metrics(test.values, prophet_pred.values)))
    lstm_metrics = dict(zip(["RMSE", "MAE", "MAPE", "R2"], eval_metrics(test.values, lstm_pred.values)))

    results = {"ARIMA": arima_metrics, "SARIMA": sarima_metrics, "Prophet": prophet_metrics, "LSTM": lstm_metrics}
    st.write("Model performance (lower RMSE is better):")
    st.table(pd.DataFrame(results).T)

    best_model_name = min(results.keys(), key=lambda m: results[m]["RMSE"])
    best_metrics = results[best_model_name]
    st.success(f"Best model selected based on RMSE: {best_model_name}")

    best_forecast = {"ARIMA": arima_pred, "SARIMA": sarima_pred, "Prophet": prophet_pred, "LSTM": lstm_pred}[best_model_name]

    # Forecast vs Actual chart (Train 5 days + Test 2 days) ----
    st.markdown("## Forecast vs Actual (Test Window = 2 days)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train.values, label="Train (5 days)", linewidth=1)
    ax.plot(test.index, test.values, label="Test (Actual 2 days)", linewidth=2)
    ax.plot(best_forecast.index, best_forecast.values, label=f"Forecast ({best_model_name})", linewidth=2)

    # overlay allocated bandwidth if available
    if "Allocated_Bandwidth" in df.columns:
        alloc_ts_min = df["Allocated_Bandwidth"].asfreq("min").interpolate()
        alloc_hourly = alloc_ts_min.resample("h").mean()
        alloc_test = alloc_hourly.reindex(test.index)
        ax.plot(alloc_test.index, alloc_test.values, label="Allocated BW (Hourly Avg)", linewidth=1.5, linestyle=":")

    ax.axvline(x=test.index[0], color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Required Bandwidth (Mbps)")

    locator = AutoDateLocator()
    formatter = DateFormatter("%Y-%m-%d\n%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        f"""
**Selected model accuracy (2-day test):**
- RMSE: {best_metrics['RMSE']:.2f}
- MAE: {best_metrics['MAE']:.2f}
- MAPE: {best_metrics['MAPE']:.2f}%
- R²: {best_metrics['R2']:.2f}
"""
    )

    # 2) NEXT WEEKEND FORECAST PLOT ----------------------------
    st.markdown("## Next Weekend Forecast (Pattern from Current Week)")

    next_weekend_index = compute_next_weekend_index(ts_hourly.index.max())
    steps_weekend = len(next_weekend_index)  # 48 hours

    # forecast next weekend using the SAME best model, trained on 5-day train window
    if best_model_name == "ARIMA":
        weekend_vals = run_arima(train, steps_weekend).values
    elif best_model_name == "SARIMA":
        weekend_vals = run_sarima(train, steps_weekend).values
    elif best_model_name == "Prophet":
        weekend_vals = run_prophet(train, steps_weekend)
    else:
        weekend_vals = run_lstm(train, steps_weekend)

    weekend_forecast = pd.Series(weekend_vals, index=next_weekend_index)

    # compare: last weekend actual (if exists) vs next weekend forecast
    last_weekend = ts_hourly[ts_hourly.index.dayofweek.isin([5, 6])]
    last_weekend = last_weekend.iloc[-48:] if len(last_weekend) >= 48 else last_weekend

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    if len(last_weekend) > 0:
        ax2.plot(last_weekend.index, last_weekend.values, label="Current Week Weekend (Actual)", linewidth=2)

    ax2.plot(weekend_forecast.index, weekend_forecast.values, label=f"Next Weekend Forecast ({best_model_name})", linewidth=2)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Required Bandwidth (Mbps)")
    ax2.set_title("Weekend Pattern Comparison: Current Weekend vs Next Weekend Forecast")

    locator2 = AutoDateLocator()
    formatter2 = DateFormatter("%Y-%m-%d\n%H:%M")
    ax2.xaxis.set_major_locator(locator2)
    ax2.xaxis.set_major_formatter(formatter2)
    fig2.autofmt_xdate()
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend()

    st.pyplot(fig2)

    st.info(
        "This chart uses the trained model (5-day training window) to forecast the next Saturday–Sunday (48 hours), "
        "and compares it with the actual weekend pattern from the current week (if available)."
    )


if __name__ == "__main__":
    main()
