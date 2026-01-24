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

# =========================
# FIXED TRAIN/TEST SETTINGS
# =========================
TRAIN_DAYS = 5
TEST_DAYS = 2
POINTS_PER_DAY = 24  # hourly
TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS
TOTAL_POINTS = TOTAL_DAYS * POINTS_PER_DAY


# 1. HELPER FUNCTIONS -----------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from a file path and set Timestamp as index."""
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values("Timestamp")
        df = df.set_index("Timestamp")
    else:
        raise ValueError("No 'Timestamp' column found in the CSV.")
    return df


def _to_hourly_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Robust conversion to hourly series (supports minute-level or already-hourly)."""
    if col not in df.columns:
        return None

    s = df[col].copy()
    s = s.loc[~s.index.duplicated(keep="last")].sort_index()

    # If we don't have enough points to infer spacing, just resample to hourly.
    if len(s.index) < 3:
        return s.resample("h").mean().interpolate()

    # infer sampling step
    deltas = (s.index.to_series().diff().dropna().dt.total_seconds()).values
    median_delta = np.median(deltas) if len(deltas) else 3600

    # minute-ish data (<=2 minutes)
    if median_delta <= 120:
        s_min = s.asfreq("min").interpolate()
        s_hour = s_min.resample("h").mean().interpolate()
        return s_hour

    # hourly-ish or slower
    s_hour = s.asfreq("h") if s.index.inferred_freq != "H" else s
    s_hour = s_hour.interpolate()
    return s_hour


def prepare_time_series(df: pd.DataFrame, target_col: str = "Required_Bandwidth") -> pd.Series:
    """Return hourly time series of target column."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in data.")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    ts_hourly = _to_hourly_series(df, target_col)
    if ts_hourly is None or ts_hourly.dropna().empty:
        raise ValueError("Could not build a valid hourly time series.")
    return ts_hourly


def prepare_alloc_series(df: pd.DataFrame, alloc_col: str = "Allocated_Bandwidth") -> pd.Series | None:
    """Return hourly allocation series if column exists."""
    if alloc_col not in df.columns:
        return None
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return _to_hourly_series(df, alloc_col)


def train_test_split_series_fixed(ts: pd.Series):
    """
    Fixed split: first 5 days train, next 2 days test.
    If series is longer than 7 days, only the first 7 days are used (current week).
    """
    if len(ts) < TOTAL_POINTS:
        raise ValueError(
            f"Not enough hourly data. Need at least {TOTAL_POINTS} points "
            f"({TOTAL_DAYS} days), but got {len(ts)}."
        )

    ts7 = ts.iloc[:TOTAL_POINTS]
    train = ts7.iloc[: TRAIN_DAYS * POINTS_PER_DAY]
    test = ts7.iloc[TRAIN_DAYS * POINTS_PER_DAY : TOTAL_POINTS]
    return train, test, ts7


def eval_metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100
    return rmse, mae, mape, r2


# 2. MODELS ---------------------------------------------------------------


def run_arima(train, test):
    model = sm.tsa.ARIMA(train, order=(2, 1, 2))
    res = model.fit()
    forecast = res.forecast(steps=len(test))
    rmse, mae, mape, r2 = eval_metrics(test.values, forecast.values)
    return forecast, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def run_sarima(train, test, max_train_points=4000):
    """SARIMA on recent window; fallback to ARIMA if it fails."""
    train_sub = train if len(train) <= max_train_points else train.iloc[-max_train_points:]

    seasonal_period = 24  # hourly -> daily seasonality

    try:
        model = SARIMAX(
            train_sub,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        forecast_vals = res.forecast(steps=len(test))
        forecast_series = pd.Series(forecast_vals, index=test.index)
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast_series.values)
        return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except MemoryError:
        st.warning("SARIMA ran out of memory. Falling back to ARIMA.")
        return run_arima(train, test)
    except Exception as e:
        st.warning(f"SARIMA fitting failed ({e}). Falling back to ARIMA.")
        return run_arima(train, test)


def run_prophet(train, test):
    df_train = train.reset_index()
    df_train.columns = ["ds", "y"]

    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df_train)

    future = m.make_future_dataframe(periods=len(test), freq="h")
    forecast_full = m.predict(future)
    forecast = forecast_full.iloc[-len(test):]["yhat"].values

    rmse, mae, mape, r2 = eval_metrics(test.values, forecast)
    forecast_series = pd.Series(forecast, index=test.index)
    return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def run_lstm(train, test, lookback=24):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    def create_sequences(data, lookback_):
        X, y = [], []
        for i in range(len(data) - lookback_):
            X.append(data[i : i + lookback_])
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

    combined = np.concatenate([train_scaled[-lookback:], test_scaled], axis=0)
    X_test, y_test = create_sequences(combined, lookback)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test).flatten()

    forecast_series = pd.Series(y_pred[-len(test) :], index=test.index)
    rmse, mae, mape, r2 = eval_metrics(y_true[-len(test) :], y_pred[-len(test) :])
    return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# 3. REPORT GENERATION HELPERS -------------------------------------------


def build_summary_excel(
    best_model_name: str,
    best_metrics: dict,
    best_forecast: pd.Series,
    test: pd.Series,
    risk_df: pd.DataFrame | None,
) -> bytes:
    """Create Excel report as bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Model metrics
        metrics_df = pd.DataFrame(best_metrics, index=[best_model_name])
        metrics_df.to_excel(writer, sheet_name="Model_Metrics")

        # Sheet 2: Forecast vs Actual (hourly)
        fa = pd.DataFrame(
            {
                "Actual_Required_Mbps": test.values,
                "Forecast_Required_Mbps": best_forecast.reindex(test.index).values,
            },
            index=test.index,
        )
        fa.index.name = "Time"
        fa.to_excel(writer, sheet_name="Forecast_vs_Actual")

        # Sheet 3: Congestion risk (if available)
        if risk_df is not None and not risk_df.empty:
            risk_df.to_excel(writer, sheet_name="Congestion_Risk", index=False)

    return output.getvalue()


def build_summary_pdf(
    location_name: str | None,
    best_model_name: str,
    best_metrics: dict,
    peak_bw: float,
    peak_time,
    recommended_capacity: float,
    risk_df: pd.DataFrame | None,
) -> bytes:
    """Create PDF summary as bytes using reportlab."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    title = "Bandwidth Forecast Summary Report"
    c.drawString(40, y, title)
    y -= 25

    c.setFont("Helvetica", 10)
    if location_name:
        c.drawString(40, y, f"Location: {location_name}")
        y -= 15

    c.drawString(40, y, f"Best Model: {best_model_name}")
    y -= 15

    c.drawString(40, y, "Model Accuracy:")
    y -= 15
    c.drawString(60, y, f"RMSE: {best_metrics['RMSE']:.2f} Mbps")
    y -= 15
    c.drawString(60, y, f"MAE:  {best_metrics['MAE']:.2f} Mbps")
    y -= 15
    c.drawString(60, y, f"MAPE: {best_metrics['MAPE']:.2f} %")
    y -= 15
    c.drawString(60, y, f"R²:   {best_metrics['R2']:.2f}")
    y -= 25

    c.drawString(40, y, "Capacity Recommendation:")
    y -= 15
    c.drawString(60, y, f"Peak forecast bandwidth: {peak_bw:.2f} Mbps")
    y -= 15
    c.drawString(60, y, f"Time of peak: {peak_time.strftime('%Y-%m-%d %H:%M')}")
    y -= 15
    c.drawString(60, y, f"Recommended capacity (peak + 20%): {recommended_capacity:.2f} Mbps")
    y -= 25

    if risk_df is not None and not risk_df.empty:
        high_risk = risk_df[risk_df["Risk_Level"] == "High"]
        c.drawString(40, y, "High-Risk Congestion Hours:")
        y -= 15

        if high_risk.empty:
            c.drawString(60, y, "No high-risk congestion hours detected.")
            y -= 15
        else:
            for _, row in high_risk.head(10).iterrows():
                txt = (
                    f"{row['Time']} | "
                    f"Forecast: {row['Forecast_Required_Mbps']:.2f} Mbps | "
                    f"Allocated: {row['Allocated_Mbps']:.2f} Mbps | "
                    f"Gap: {row['Gap_Mbps']:.2f} Mbps"
                )
                c.drawString(60, y, txt[:100])
                y -= 12
                if y < 60:
                    c.showPage()
                    y = height - 40
                    c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


# 4. STREAMLIT APP --------------------------------------------------------

st.set_page_config(page_title="Bandwidth Forecast Dashboard", layout="wide")

# ---------- DASHBOARD CSS ----------
st.markdown(
    """
<style>
[data-testid="stSidebar"] { background-color: #1e1f26 !important; color: #ffffff !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #111218 !important; color: #ffffff !important; border: 1px solid #3a3b44 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #111218 !important; color: #ffffff !important; border: 1px solid #3a3b44 !important;
}
[data-testid="stSidebar"] .stSlider > div > div { color: #ffffff !important; }
[data-testid="stFileUploader"] {
    background-color: #111218 !important; border: 1px solid #3a3b44 !important; border-radius: 8px;
}
[data-testid="stFileUploader"] * { color: #ffffff !important; }

.decision-box {
    border: 1px solid #3a3b44;
    border-radius: 10px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.03);
}

.high-risk { color: #ff6b6b; font-weight: 700; }
.medium-risk { color: #ffd27f; font-weight: 700; }
.low-risk { color: #fff59d; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    st.title("Bandwidth Forecast Dashboard (Single Location)")

    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload QoS CSV file", type=["csv"])

    data_path = st.sidebar.text_input(
        "OR CSV file path (if not uploading)",
        "expandedBootstrapping_quality_of_service_5g.csv",
    )

    # Fixed requirement: 5 days train, 2 days test
    st.sidebar.info(f"Fixed split:\n• Train = {TRAIN_DAYS} days\n• Test = {TEST_DAYS} days")

    # Load data ------------------------------------------------
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "Timestamp" not in df.columns:
                st.error("Uploaded file must contain a 'Timestamp' column.")
                return
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df = df.sort_values("Timestamp").set_index("Timestamp")
        else:
            df = load_data(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Location filter ------------------------------------------
    selected_location = None
    if "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
        if len(locations) == 0:
            st.sidebar.warning("Location column is present but empty.")
        else:
            default_idx = 0
            if "Location X" in locations:
                default_idx = locations.index("Location X")
            selected_location = st.sidebar.selectbox("Select Location", locations, index=default_idx)
            df = df[df["Location"] == selected_location]
            if df.empty:
                st.error(f"No data available for location: {selected_location}")
                return
            st.sidebar.success(f"Filtering data for location: {selected_location}")
    else:
        st.sidebar.warning("No 'Location' column found. Using all data.")

    st.sidebar.write("Columns found:", list(df.columns))

    # Prepare time series --------------------------------------
    try:
        ts_hourly = prepare_time_series(df, target_col="Required_Bandwidth")
        alloc_ts_hourly_full = prepare_alloc_series(df, alloc_col="Allocated_Bandwidth")
        train, test, ts7 = train_test_split_series_fixed(ts_hourly)

        # Align allocation to the same 7-day window if available
        if alloc_ts_hourly_full is not None:
            alloc_ts_hourly_full = alloc_ts_hourly_full.reindex(ts7.index).interpolate()
    except Exception as e:
        st.error(f"Error preparing time series: {e}")
        return

    if selected_location:
        st.subheader(f"Time Range — Location: {selected_location}")
    else:
        st.subheader("Time Range")

    st.write(f"Train (5 days): {train.index[0]} -> {train.index[-1]}")
    st.write(f"Test  (2 days): {test.index[0]} -> {test.index[-1]}")

    # Train models ---------------------------------------------
    st.markdown("## Model Training and Selection")

    arima_forecast, arima_metrics = run_arima(train, test)
    sarima_forecast, sarima_metrics = run_sarima(train, test)
    prophet_forecast, prophet_metrics = run_prophet(train, test)
    lstm_forecast, lstm_metrics = run_lstm(train, test)

    results = {
        "ARIMA": arima_metrics,
        "SARIMA": sarima_metrics,
        "Prophet": prophet_metrics,
        "LSTM": lstm_metrics,
    }

    best_model_name = min(results.keys(), key=lambda m: results[m]["RMSE"])
    best_metrics = results[best_model_name]

    st.write("Model performance (lower RMSE is better):")
    st.table(pd.DataFrame(results).T)
    st.success(f"Best model selected based on RMSE: {best_model_name}")

    if best_model_name == "ARIMA":
        best_forecast = arima_forecast
    elif best_model_name == "SARIMA":
        best_forecast = sarima_forecast
    elif best_model_name == "Prophet":
        best_forecast = prophet_forecast
    else:
        best_forecast = lstm_forecast

    # A. REAL-TIME NETWORK STATUS PANEL ------------------------
    st.markdown("## A. Real-Time Network Status Panel")

    latest_row = df.iloc[-1]
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    required_now = latest_row.get("Required_Bandwidth", np.nan)
    allocated_now = latest_row.get("Allocated_Bandwidth", np.nan)
    gap_now = allocated_now - required_now if pd.notna(required_now) and pd.notna(allocated_now) else np.nan

    signal_now = latest_row.get("Signal_Strength", np.nan)
    latency_now = latest_row.get("Latency", np.nan)
    users_now = latest_row.get("Active_Users", np.nan)

    col1.metric("Required BW (Mbps)", f"{required_now:.2f}" if pd.notna(required_now) else "N/A")
    col2.metric("Allocated BW (Mbps)", f"{allocated_now:.2f}" if pd.notna(allocated_now) else "N/A")
    col3.metric("Gap (Alloc - Req)", f"{gap_now:.2f}" if pd.notna(gap_now) else "N/A")
    col4.metric("Signal Strength (dBm)", f"{signal_now:.1f}" if pd.notna(signal_now) else "N/A")
    col5.metric("Latency (ms)", f"{latency_now:.1f}" if pd.notna(latency_now) else "N/A")
    col6.metric("Active Users", f"{int(users_now)}" if pd.notna(users_now) else "N/A")

    # B. FORECAST VS ACTUAL (BEST MODEL) -----------------------
    st.markdown("## B. Forecast vs Actual (Best Model)")

    history_days = TRAIN_DAYS
    history_points = history_days * POINTS_PER_DAY
    train_tail = train.iloc[-history_points:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_tail.index, train_tail.values, label=f"Train (last {history_days} days)", linewidth=1)
    ax.plot(test.index, test.values, label="Test (Actual)", linewidth=2)
    ax.plot(best_forecast.index, best_forecast.values, label=f"Forecast ({best_model_name})", linewidth=2)

    # overlay allocated bandwidth if available (aligned to test)
    alloc_test = None
    if alloc_ts_hourly_full is not None:
        alloc_test = alloc_ts_hourly_full.reindex(test.index)
        ax.plot(
            alloc_test.index,
            alloc_test.values,
            label="Allocated BW (Hourly)",
            linewidth=1.5,
            linestyle=":",
        )

    ax.axvline(x=test.index[0], color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(test.index[0], ax.get_ylim()[1], "  Train -> Test", va="top", ha="left", fontsize=9)

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
**Model accuracy for the selected location (Test = {TEST_DAYS} days)**

- RMSE: {best_metrics['RMSE']:.2f} Mbps  
- MAE: {best_metrics['MAE']:.2f} Mbps  
- MAPE: {best_metrics['MAPE']:.2f}%  
- R²: {best_metrics['R2']:.2f}
"""
    )

    # (2) Weekend forecast pattern visualization -----------------------
    st.markdown("## B2. Weekend Forecast Pattern (Next Weekend) Based on Current Week")

    weekend_mask = test.index.dayofweek.isin([5, 6])  # Sat=5, Sun=6
    weekend_actual = test[weekend_mask]
    weekend_forecast = best_forecast.reindex(test.index)[weekend_mask]

    if weekend_actual.empty:
        st.info(
            "No weekend hours detected inside the 2-day test window. "
            "If your 7-day data does not end on Sat/Sun, shift the dataset start/end so last 2 days are weekend."
        )
    else:
        # Weekend line chart
        fig_w, ax_w = plt.subplots(figsize=(12, 4))
        ax_w.plot(weekend_actual.index, weekend_actual.values, label="Weekend Actual", linewidth=2)
        ax_w.plot(
            weekend_forecast.index,
            weekend_forecast.values,
            label=f"Weekend Forecast ({best_model_name})",
            linewidth=2,
        )

        if alloc_ts_hourly_full is not None:
            alloc_weekend = alloc_ts_hourly_full.reindex(weekend_actual.index)
            ax_w.plot(
                alloc_weekend.index,
                alloc_weekend.values,
                label="Weekend Allocated BW",
                linewidth=1.5,
                linestyle=":",
            )

        ax_w.set_xlabel("Time")
        ax_w.set_ylabel("Required Bandwidth (Mbps)")
        ax_w.xaxis.set_major_locator(AutoDateLocator())
        ax_w.xaxis.set_major_formatter(DateFormatter("%a\n%H:%M"))
        fig_w.autofmt_xdate()
        ax_w.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax_w.legend()
        st.pyplot(fig_w)

        # Weekend hourly pattern (profile)
        df_weekend = pd.DataFrame(
            {"hour": weekend_actual.index.hour, "actual": weekend_actual.values, "forecast": weekend_forecast.values}
        )
        profile = df_weekend.groupby("hour")[["actual", "forecast"]].mean().reset_index()

        fig_p, ax_p = plt.subplots(figsize=(12, 4))
        ax_p.plot(profile["hour"], profile["actual"], label="Avg Weekend Actual", linewidth=2)
        ax_p.plot(profile["hour"], profile["forecast"], label=f"Avg Weekend Forecast ({best_model_name})", linewidth=2)
        ax_p.set_xlabel("Hour of Day")
        ax_p.set_ylabel("Avg Required Bandwidth (Mbps)")
        ax_p.set_xticks(range(0, 24))
        ax_p.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax_p.legend()
        st.pyplot(fig_p)

    # C. CAPACITY RECOMMENDATION & RISK HOURS ------------------
    st.markdown("## C. Capacity Recommendation & Congestion Risk Hours")

    peak_bw = float(best_forecast.max())
    peak_time = best_forecast.idxmax()
    recommended_capacity = 1.20 * peak_bw  # 20% headroom

    col_cap1, col_cap2, col_cap3 = st.columns(3)
    col_cap1.metric("Peak Forecast BW (Mbps)", f"{peak_bw:.2f}")
    col_cap2.metric("Time of Peak", peak_time.strftime("%Y-%m-%d %H:%M"))
    col_cap3.metric("Recommended Capacity (Mbps)", f"{recommended_capacity:.2f}")

    st.markdown(
        f"""
<div class="decision-box">
<b>Capacity Recommendation:</b><br>
• Forecast peak demand: <b>{peak_bw:.2f} Mbps</b><br>
• Recommended minimum capacity (peak + 20% headroom): <b>{recommended_capacity:.2f} Mbps</b><br><br>
This target capacity helps reduce congestion risk during peak usage hours while still being cost-conscious.
</div>
""",
        unsafe_allow_html=True,
    )

    risk_df = None
    alloc_forecast = None

    if alloc_ts_hourly_full is not None:
        alloc_forecast = alloc_ts_hourly_full.reindex(best_forecast.index)

        risk_df = pd.DataFrame(
            {
                "Time": best_forecast.index,
                "Forecast_Required_Mbps": best_forecast.values,
                "Allocated_Mbps": alloc_forecast.values,
            }
        )
        risk_df["Gap_Mbps"] = risk_df["Forecast_Required_Mbps"] - risk_df["Allocated_Mbps"]

        def label_risk(gap):
            if pd.isna(gap) or gap <= 0:
                return "No Risk"
            if gap < 5:
                return "Low"
            if gap < 15:
                return "Medium"
            return "High"

        risk_df["Risk_Level"] = risk_df["Gap_Mbps"].apply(label_risk)
        congested_hours = risk_df[risk_df["Gap_Mbps"] > 0]

        st.markdown("### All Congested Hours Requiring More Bandwidth")

        if congested_hours.empty:
            st.success("No congestion is predicted. Allocated bandwidth is sufficient for the entire forecast window.")
        else:
            st.write("These hours show forecasted demand higher than allocated bandwidth:")

            def risk_color(val):
                if val == "High":
                    return "background-color: #ff9999"
                if val == "Medium":
                    return "background-color: #ffd27f"
                if val == "Low":
                    return "background-color: #ffffb3"
                return ""

            congested_display = congested_hours.copy()
            congested_display["Time"] = congested_display["Time"].dt.strftime("%Y-%m-%d %H:%M")

            styled = (
                congested_display.set_index("Time")
                .style.format(
                    {
                        "Forecast_Required_Mbps": "{:.2f}",
                        "Allocated_Mbps": "{:.2f}",
                        "Gap_Mbps": "{:.2f}",
                    }
                )
                .applymap(risk_color, subset=["Risk_Level"])
            )

            st.dataframe(styled)

            # Automatic high-risk alerts (text)
            high_risk = congested_hours[congested_hours["Risk_Level"] == "High"]
            if not high_risk.empty:
                st.markdown("### Automatic High-Risk Alerts")
                for _, row in high_risk.iterrows():
                    t = row["Time"].strftime("%a %Y-%m-%d %H:%M")
                    st.write(
                        f"- High risk at {t}: forecast {row['Forecast_Required_Mbps']:.2f} Mbps, "
                        f"allocated {row['Allocated_Mbps']:.2f} Mbps (gap {row['Gap_Mbps']:.2f} Mbps)"
                    )

        # C.1 WHAT-IF CAPACITY SCENARIO ------------------------
        st.markdown("### What-If Capacity Scenario")

        capacity_factor = st.slider(
            "Simulate increasing allocated bandwidth (× current capacity)",
            0.5,
            2.0,
            1.0,
            0.1,
            help="1.0 = current capacity, 1.2 = +20% more bandwidth",
        )

        sim_alloc = alloc_forecast * capacity_factor

        sim_risk = pd.DataFrame(
            {
                "Time": best_forecast.index,
                "Forecast_Required_Mbps": best_forecast.values,
                "Simulated_Allocated_Mbps": sim_alloc.values,
            }
        )
        sim_risk["Gap_Mbps"] = sim_risk["Forecast_Required_Mbps"] - sim_risk["Simulated_Allocated_Mbps"]
        sim_risk["Status"] = sim_risk["Gap_Mbps"].apply(lambda g: "Congested" if g > 0 else "OK")

        congested_sim = sim_risk[sim_risk["Gap_Mbps"] > 0]

        if congested_sim.empty:
            st.success(f"With capacity factor ×{capacity_factor:.2f}, no congestion is expected in the forecast window.")
        else:
            st.warning(f"With capacity factor ×{capacity_factor:.2f}, {len(congested_sim)} hours are still congested.")
            st.dataframe(
                congested_sim.set_index("Time").style.format(
                    {
                        "Forecast_Required_Mbps": "{:.2f}",
                        "Simulated_Allocated_Mbps": "{:.2f}",
                        "Gap_Mbps": "{:.2f}",
                    }
                )
            )

        st.markdown(
            """
**Interpretation of risk levels**

- <span class="high-risk">High</span>: immediate capacity upgrade strongly recommended.  
- <span class="medium-risk">Medium</span>: monitor and consider targeted increase.  
- <span class="low-risk">Low</span>: demand is close to capacity but usually manageable.  
- No Risk: allocated capacity is above forecast demand.
""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No 'Allocated_Bandwidth' column found. Risk analysis and what-if simulation are disabled.")

    # D. ANOMALY / SPIKE DETECTION ----------------------------
    st.markdown("## D. Anomaly Detection (Unexpected Demand Spikes)")

    ts_df = ts7.to_frame("BW")
    ts_df["RollingMean"] = ts_df["BW"].rolling(window=24, min_periods=6).mean()
    ts_df["RollingStd"] = ts_df["BW"].rolling(window=24, min_periods=6).std()
    ts_df["Zscore"] = (ts_df["BW"] - ts_df["RollingMean"]) / (ts_df["RollingStd"] + 1e-6)

    anomalies = ts_df[ts_df["Zscore"].abs() > 3].dropna()

    if anomalies.empty:
        st.success("No abnormal spikes detected in bandwidth usage.")
    else:
        st.error(f"{len(anomalies)} anomalous high-demand spikes detected:")
        st.dataframe(anomalies[["BW", "Zscore"]].style.format({"BW": "{:.2f}", "Zscore": "{:.2f}"}))

    # E. WEEKLY CONGESTION PATTERN HEATMAP --------------------
    st.markdown("## E. Weekly Congestion Pattern (Day of Week × Hour of Day)")

    pattern = ts7.to_frame(name="Required_BW_Mbps").copy()
    pattern["dayofweek"] = pattern.index.dayofweek  # 0=Mon, 6=Sun
    pattern["hour"] = pattern.index.hour  # 0–23

    pivot = pattern.pivot_table(index="dayofweek", columns="hour", values="Required_BW_Mbps", aggfunc="mean")
    pivot = pivot.reindex(index=list(range(7)), columns=list(range(24)))

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
    im = ax_hm.imshow(pivot.values, aspect="auto")

    ax_hm.set_xticks(range(24))
    ax_hm.set_xticklabels(list(range(24)))
    ax_hm.set_yticks(range(7))
    ax_hm.set_yticklabels(day_labels)

    ax_hm.set_xlabel("Hour of Day")
    ax_hm.set_ylabel("Day of Week")
    title_suffix = f" — {selected_location}" if selected_location else ""
    ax_hm.set_title("Average Required Bandwidth (Mbps) by Day and Hour" + title_suffix)

    cbar = plt.colorbar(im, ax=ax_hm)
    cbar.set_label("Required Bandwidth (Mbps)")
    st.pyplot(fig_hm)

    st.markdown(
        """
Darker cells indicate hours with higher average required bandwidth for the selected location.
This helps the ISP spot recurring congestion patterns such as:

- Peak hours in the evening
- Differences between weekdays and weekends
"""
    )

    # F. NOC SUMMARY (TEXT) -----------------------------------
    st.markdown("## F. Summary for Network Operations Center (NOC)")

    avg_bw = float(best_forecast.mean())
    exceed_hours = int((best_forecast > alloc_forecast).sum()) if alloc_forecast is not None else None

    summary_text = f"""
### Automated Summary

- Location: **{selected_location if selected_location else "All"}**
- Best forecasting model: **{best_model_name}**
- Forecast peak demand: **{peak_bw:.2f} Mbps**
- Average forecast demand: **{avg_bw:.2f} Mbps**
"""
    if exceed_hours is not None:
        summary_text += f"- Hours where required > allocated bandwidth: **{exceed_hours}**\n"

    summary_text += """
### Operational Guidance

- Ensure capacity is at least equal to the *Recommended Capacity* shown above to avoid peak-time congestion.
- Focus monitoring on **High-Risk** hours highlighted in the congestion table.
- Use the **What-If Capacity Scenario** to test upgrade plans before deployment.
- Use the **Weekly Pattern Heatmap** to plan long-term capacity (e.g. more capacity on evenings or specific days).
"""
    st.markdown(summary_text)

    # G. DOWNLOADABLE REPORTS ---------------------------------
    st.markdown("## G. Downloadable Reports")

    col_rep1, col_rep2 = st.columns(2)

    # Excel report
    with col_rep1:
        excel_bytes = build_summary_excel(best_model_name, best_metrics, best_forecast, test, risk_df)
        st.download_button(
            label="Download Excel Report",
            data=excel_bytes,
            file_name="bandwidth_forecast_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # PDF report
    with col_rep2:
        pdf_bytes = build_summary_pdf(
            selected_location,
            best_model_name,
            best_metrics,
            peak_bw,
            peak_time,
            recommended_capacity,
            risk_df,
        )
        st.download_button(
            label="Download PDF Summary",
            data=pdf_bytes,
            file_name="bandwidth_forecast_summary.pdf",
            mime="application/pdf",
        )

    st.info(
        "To view the full **hour-by-hour forecasting report** (including all timestamps), "
        "please download the **Excel report** and open the sheet named **'Forecast_vs_Actual'**."
    )


if __name__ == "__main__":
    main()
