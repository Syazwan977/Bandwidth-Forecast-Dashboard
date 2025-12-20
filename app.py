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
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values("Timestamp")
        df = df.set_index("Timestamp")
    else:
        raise ValueError("No 'Timestamp' column found in the CSV.")
    return df


def prepare_time_series(df: pd.DataFrame, target_col: str = "Required_Bandwidth") -> pd.Series:
    """Return hourly time series of target column (from 1-minute data)."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in data.")
    # Ignore unnamed junk columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    ts_min = df[target_col].asfreq("min")
    ts_min = ts_min.interpolate()
    ts_hourly = ts_min.resample("h").mean().interpolate()
    return ts_hourly


def train_test_split_series(ts: pd.Series, test_days: int = 7):
    """Split hourly series into train and test using last test_days days as test."""
    points_per_day = 24  # hourly
    horizon = points_per_day * test_days
    if len(ts) <= horizon:
        raise ValueError("Time series is too short relative to test period.")
    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]
    return train, test


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

    # try to infer seasonal period; default to 24 (daily seasonality on hourly data)
    try:
        delta_min = (train_sub.index[1] - train_sub.index[0]).total_seconds() / 60.0
        seasonal_period = max(1, int(round(24 * 60 / delta_min)))
    except Exception:
        seasonal_period = 24

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

    combined = np.concatenate([train_scaled[-lookback:], test_scaled], axis=0)
    X_test, y_test = create_sequences(combined, lookback)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test).flatten()

    forecast_index = test.index
    forecast_series = pd.Series(y_pred[-len(test):], index=forecast_index)

    rmse, mae, mape, r2 = eval_metrics(y_true[-len(test):], y_pred[-len(test):])
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
    # use openpyxl instead of openpxl to avoid warning  about engine            
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
            # show up to 10 high risk entries
            for _, row in high_risk.head(10).iterrows():
                txt = (
                    f"{row['Time']} | "
                    f"Forecast: {row['Forecast_Required_Mbps']:.2f} Mbps | "
                    f"Allocated: {row['Allocated_Mbps']:.2f} Mbps | "
                    f"Gap: {row['Gap_Mbps']:.2f} Mbps"
                )
                c.drawString(60, y, txt[:100])  # truncate if too long
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

# ---------- FORMAL DASHBOARD CSS (CARDS, TITLES, DECISION BOXES) ----------
st.markdown(
    """
<style>

/* -----------------------------
   SIDEBAR DARK THEME
--------------------------------*/
[data-testid="stSidebar"] {
    background-color: #1e1f26 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar input boxes */
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #111218 !important;
    color: #ffffff !important;
    border: 1px solid #3a3b44 !important;
}

/* Sidebar dropdown */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #111218 !important;
    color: #ffffff !important;
    border: 1px solid #3a3b44 !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider > div > div {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #ffffff !important;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #111218 !important;
    border: 1px solid #3a3b44 !important;
    border-radius: 8px;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}

/* Sidebar alert boxes */
[data-testid="stNotification"] {
    background-color: #2f3a2f !important;
    border-left: 4px solid #4caf50 !important;
    color: #ffffff !important;
}

/* Fix JSON expander box */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .streamlit-expanderContent {
    background-color: #111218 !important;
    color: #ffffff !important;
}

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

    test_days = st.sidebar.slider("Test horizon (days)", 3, 14, 7)

    # Load data ------------------------------------------------
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "Timestamp" not in df.columns:
                st.error("Uploaded file must contain a 'Timestamp' column.")
                return
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"], dayfirst=True, errors="coerce"
            )
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
            if "Sungai Besi" in locations:
                default_idx = locations.index("Sungai Besi")
            selected_location = st.sidebar.selectbox(
                "Select Location",
                locations,
                index=default_idx,
            )
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
        train, test = train_test_split_series(ts_hourly, test_days=test_days)
    except Exception as e:
        st.error(f"Error preparing time series: {e}")
        return

    if selected_location:
        st.subheader(f"Time Range — Location: {selected_location}")
    else:
        st.subheader("Time Range")

    st.write(f"Train: {train.index[0]} -> {train.index[-1]}")
    st.write(f"Test:  {test.index[0]} -> {test.index[-1]}")

    # Train models ---------------------------------------------
    st.markdown('<div class="section-title">Model Training and Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

    # A. REAL-TIME NETWORK STATUS PANEL ------------------------
    st.markdown('<div class="section-title">A. Real-Time Network Status Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    latest_row = df.iloc[-1]

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    required_now = latest_row.get("Required_Bandwidth", np.nan)
    allocated_now = latest_row.get("Allocated_Bandwidth", np.nan)
    gap_now = (
        allocated_now - required_now
        if pd.notna(required_now) and pd.notna(allocated_now)
        else np.nan
    )
    signal_now = latest_row.get("Signal_Strength", np.nan)
    latency_now = latest_row.get("Latency", np.nan)
    users_now = latest_row.get("Active_Users", np.nan)

    col1.metric(
        "Required BW (Mbps)",
        f"{required_now:.2f}" if pd.notna(required_now) else "N/A",
    )
    col2.metric(
        "Allocated BW (Mbps)",
        f"{allocated_now:.2f}" if pd.notna(allocated_now) else "N/A",
    )
    col3.metric(
        "Gap (Alloc - Req)", f"{gap_now:.2f}" if pd.notna(gap_now) else "N/A"
    )
    col4.metric(
        "Signal Strength (dBm)",
        f"{signal_now:.1f}" if pd.notna(signal_now) else "N/A",
    )
    col5.metric(
        "Latency (ms)", f"{latency_now:.1f}" if pd.notna(latency_now) else "N/A"
    )
    col6.metric(
        "Active Users", f"{int(users_now)}" if pd.notna(users_now) else "N/A"
    )

    st.markdown('</div>', unsafe_allow_html=True)


    # B. FORECAST VS ACTUAL (BEST MODEL) -----------------------
    st.markdown('<div class="section-title">B. Forecast vs Actual (Best Model)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    history_days = min(3, test_days)
    history_points = history_days * 24
    train_tail = train.iloc[-history_points:]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        train_tail.index,
        train_tail.values,
        label=f"Train (last {history_days} days)",
        linewidth=1,
    )
    ax.plot(
        test.index,
        test.values,
        label="Test (Actual)",
        linewidth=2,
    )
    ax.plot(
        best_forecast.index,
        best_forecast.values,
        label=f"Forecast ({best_model_name})",
        linewidth=2,
    )

    # overlay allocated bandwidth if available
    alloc_ts_hourly = None
    if "Allocated_Bandwidth" in df.columns:
        alloc_ts_min = df["Allocated_Bandwidth"].asfreq("min").interpolate()
        alloc_ts_hourly = alloc_ts_min.resample("h").mean()
        alloc_test = alloc_ts_hourly.reindex(test.index)
        ax.plot(
            alloc_test.index,
            alloc_test.values,
            label="Allocated BW (Hourly Avg)",
            linewidth=1.5,
            linestyle=":",
        )

    ax.axvline(
        x=test.index[0],
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
    )
    ax.text(
        test.index[0],
        ax.get_ylim()[1],
        "  Train -> Test",
        va="top",
        ha="left",
        fontsize=9,
    )

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
**How to interpret this chart**

- Train: historical demand used to fit the model.
- Test (Actual): real demand in the evaluation window.
- Forecast ({best_model_name}): model prediction over the same period.
- If the forecast line is *below* the actual line, there is risk of under-provisioning and congestion.
- If the forecast line is *above* the actual line, capacity may be over-allocated.

**Model accuracy for the selected location**

- RMSE: {best_metrics['RMSE']:.2f} Mbps  
- MAE: {best_metrics['MAE']:.2f} Mbps  
- MAPE: {best_metrics['MAPE']:.2f}%  
- R²: {best_metrics['R2']:.2f}
"""
    )

    st.markdown('</div>', unsafe_allow_html=True)



    # C. CAPACITY RECOMMENDATION & RISK HOURS ------------------
    st.markdown('<div class="section-title">C. Capacity Recommendation & Congestion Risk Hours</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    if alloc_ts_hourly is not None:
        # Align allocation to forecast horizon
        alloc_forecast = alloc_ts_hourly.reindex(best_forecast.index)

        risk_df = pd.DataFrame(
            {
                "Time": best_forecast.index,
                "Forecast_Required_Mbps": best_forecast.values,
                "Allocated_Mbps": alloc_forecast.values,
            }
        )
        risk_df["Gap_Mbps"] = (
            risk_df["Forecast_Required_Mbps"] - risk_df["Allocated_Mbps"]
        )

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
            st.success(
                "No congestion is predicted. Allocated bandwidth is sufficient for the entire forecast window."
            )
        else:
            st.write(
                "These hours show forecasted demand higher than allocated bandwidth:"
            )

            def risk_color(val):
                if val == "High":
                    return "background-color: #ff9999"
                if val == "Medium":
                    return "background-color: #ffd27f"
                if val == "Low":
                    return "background-color: #ffffb3"
                return ""

            congested_display = congested_hours.copy()
            congested_display["Time"] = congested_display["Time"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

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
        sim_risk["Gap_Mbps"] = (
            sim_risk["Forecast_Required_Mbps"] - sim_risk["Simulated_Allocated_Mbps"]
        )
        sim_risk["Status"] = sim_risk["Gap_Mbps"].apply(
            lambda g: "Congested" if g > 0 else "OK"
        )

        congested_sim = sim_risk[sim_risk["Gap_Mbps"] > 0]

        if congested_sim.empty:
            st.success(
                f"With capacity factor ×{capacity_factor:.2f}, no congestion is expected in the forecast window."
            )
        else:
            st.warning(
                f"With capacity factor ×{capacity_factor:.2f}, "
                f"{len(congested_sim)} hours are still congested."
            )
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
        st.info(
            "No 'Allocated_Bandwidth' column was found. Congestion risk hours and what-if capacity scenario cannot be calculated."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # D. ANOMALY / SPIKE DETECTION ----------------------------
    st.markdown('<div class="section-title">D. Anomaly Detection (Unexpected Demand Spikes)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    ts_df = ts_hourly.to_frame("BW")
    ts_df["RollingMean"] = ts_df["BW"].rolling(window=24, min_periods=6).mean()
    ts_df["RollingStd"] = ts_df["BW"].rolling(window=24, min_periods=6).std()
    ts_df["Zscore"] = (ts_df["BW"] - ts_df["RollingMean"]) / (ts_df["RollingStd"] + 1e-6)

    anomalies = ts_df[ts_df["Zscore"].abs() > 3].dropna()

    if anomalies.empty:
        st.success("No abnormal spikes detected in bandwidth usage.")
    else:
        st.error(f"{len(anomalies)} anomalous high-demand spikes detected:")
        st.dataframe(
            anomalies[["BW", "Zscore"]].style.format(
                {"BW": "{:.2f}", "Zscore": "{:.2f}"}
            )
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # E. WEEKLY CONGESTION PATTERN HEATMAP --------------------
    st.markdown('<div class="section-title">E. Weekly Congestion Pattern (Day of Week × Hour of Day)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    pattern = ts_hourly.to_frame(name="Required_BW_Mbps").copy()
    pattern["dayofweek"] = pattern.index.dayofweek  # 0=Mon, 6=Sun
    pattern["hour"] = pattern.index.hour  # 0–23

    pivot = pattern.pivot_table(
        index="dayofweek",
        columns="hour",
        values="Required_BW_Mbps",
        aggfunc="mean",
    )

    day_order = list(range(7))
    hour_order = list(range(24))
    pivot = pivot.reindex(index=day_order, columns=hour_order)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
    im = ax_hm.imshow(pivot.values, aspect="auto")

    ax_hm.set_xticks(range(24))
    ax_hm.set_xticklabels(hour_order)
    ax_hm.set_yticks(range(7))
    ax_hm.set_yticklabels(day_labels)

    ax_hm.set_xlabel("Hour of Day")
    ax_hm.set_ylabel("Day of Week")
    title_suffix = f" — {selected_location}" if selected_location else ""
    ax_hm.set_title(
        "Average Required Bandwidth (Mbps) by Day and Hour" + title_suffix
    )

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

    st.markdown('</div>', unsafe_allow_html=True)

    # F. NOC SUMMARY (TEXT) -----------------------------------
    st.markdown('<div class="section-title">F. Summary for Network Operations Center (NOC)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    avg_bw = float(best_forecast.mean())
    if "alloc_forecast" in locals():
        exceed_hours = int((best_forecast > alloc_forecast).sum())
    else:
        exceed_hours = None

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

    st.markdown('</div>', unsafe_allow_html=True)

    # G. DOWNLOADABLE REPORTS ---------------------------------
    st.markdown('<div class="section-title">G. Downloadable Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_rep1, col_rep2 = st.columns(2)

    # Excel report
    with col_rep1:
        excel_bytes = build_summary_excel(
            best_model_name,
            best_metrics,
            best_forecast,
            test,
            risk_df,
        )
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

    # 🔹 NOTE: Explain that full hourly forecast is inside the downloaded report
    st.info(
        "To view the full **hour-by-hour forecasting report** (including all timestamps), "
        "please download the **Excel report** and open the sheet named **'Forecast_vs_Actual'**."
    )

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()






