"""Streamlit sales forecast dashboard."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    ALL_PRODUCTS,
    EVAL_HORIZON,
    EVAL_STEP,
    MODEL_PROPHET,
    MODEL_XGBOOST,
    DataError,
    clean_sales,
    run_product_forecast,
)
from report import build_report_pdf

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_sales.csv"
HORIZON_OPTIONS = (30, 60, 90)

st.set_page_config(page_title="Sales forecasting", layout="wide")
st.title("Sales forecasting")
st.caption(
    "Required columns: date, product, quantity, revenue. Names are case-insensitive."
)


def read_sales_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DataError(f"Could not parse the CSV: {exc}") from exc


@st.cache_data(show_spinner=False)
def cached_clean(file_bytes: bytes) -> pd.DataFrame:
    return clean_sales(read_sales_csv(file_bytes))


@st.cache_data(show_spinner=False)
def cached_forecast(file_bytes: bytes, product: str) -> dict:
    return run_product_forecast(cached_clean(file_bytes), product)


def _model_key(file_bytes: bytes, product: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"active-model-{digest}-{product}"


uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
if st.button("Load bundled sample"):
    st.session_state["use_sample"] = True

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    source_name = uploaded.name
elif st.session_state.get("use_sample"):
    if not SAMPLE_PATH.is_file():
        st.error(f"Bundled sample CSV was not found at {SAMPLE_PATH}.")
        st.stop()
    file_bytes = SAMPLE_PATH.read_bytes()
    source_name = SAMPLE_PATH.name
else:
    st.info("Upload a CSV or load the bundled sample to forecast revenue.")
    st.stop()

st.caption(f"Source: {source_name}")

try:
    clean = cached_clean(file_bytes)
except DataError as exc:
    st.error(str(exc))
    st.stop()

products = sorted(clean["product"].astype(str).unique().tolist())
product = st.selectbox("Product", [ALL_PRODUCTS, *products])

try:
    with st.spinner("Comparing Prophet and XGBoost on the rolling window..."):
        result = cached_forecast(file_bytes, product)
except DataError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

metrics = result["metrics"]
history = result["history"]
model_key = _model_key(file_bytes, product)
if model_key not in st.session_state:
    st.session_state[model_key] = metrics["winner"]

if st.session_state[model_key] == MODEL_XGBOOST:
    forecast_tab, importance_tab, report_tab = st.tabs(
        ["Forecast", "Feature importance", "Report"]
    )
else:
    forecast_tab, report_tab = st.tabs(["Forecast", "Report"])
    importance_tab = None

with forecast_tab:
    horizon = st.selectbox("Forecast horizon (days)", list(HORIZON_OPTIONS), index=0)
    active_model = st.selectbox(
        "Model",
        [MODEL_PROPHET, MODEL_XGBOOST],
        key=model_key,
    )
    st.write(
        f"Winner: **{metrics['winner']}** "
        "(lower RMSE, then lower MAE; Prophet if both tie)."
    )
    if active_model != metrics["winner"]:
        st.caption(f"Chart is using {active_model} because the model control was overridden.")
    comparison = pd.DataFrame(
        {
            "Model": [MODEL_PROPHET, MODEL_XGBOOST],
            "MAE": [round(metrics["prophet_mae"], 2), round(metrics["xgboost_mae"], 2)],
            "RMSE": [round(metrics["prophet_rmse"], 2), round(metrics["xgboost_rmse"], 2)],
        }
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    start = pd.Timestamp(history.index.min())
    end = pd.Timestamp(history.index.max())
    st.caption(
        f"{metrics['n_points']} daily points, {start:%Y-%m-%d} to {end:%Y-%m-%d}. "
        f"Total revenue {result['revenue_total']:,.2f}. "
        f"Total quantity {result['quantity_total']:,.0f}."
    )
    st.caption(
        f"Rolling window: {metrics['training_window']} training days, "
        f"{EVAL_HORIZON}-day holdout, step {EVAL_STEP}, "
        f"{metrics['n_origins']} origin(s). "
        "MAE and RMSE use every holdout day equally. "
        "The 90-day forecast is fit once; 30 and 60 days are prefixes and do not refit."
    )
    view = result["forecasts"][active_model].iloc[: int(horizon)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["yhat_upper"],
            mode="lines",
            line={"width": 0},
            hovertemplate="Upper %{y:,.2f}<extra></extra>",
            showlegend=False,
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["yhat_lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.20)",
            hovertemplate="Lower %{y:,.2f}<extra></extra>",
            name="95% confidence",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.to_numpy(dtype=float),
            mode="lines",
            name="Actual revenue",
            line={"color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["yhat"],
            mode="lines",
            name="Forecast",
            line={"color": "#d62728"},
        )
    )
    fig.update_layout(
        title=f"{product} — {active_model}, {horizon} days",
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)
    if active_model == MODEL_XGBOOST:
        st.caption(
            "The shaded band is a homoscedastic residual interval "
            "(prediction ± 1.96σ of rolling-origin residuals), not a conformal interval."
        )
    else:
        st.caption("The shaded band is Prophet's 95% interval.")

if importance_tab is not None:
    with importance_tab:
        importance = result["importance"]
        importance_fig = go.Figure(
            go.Bar(
                x=importance["importance"],
                y=importance["feature"],
                orientation="h",
            )
        )
        importance_fig.update_layout(
            title="XGBoost feature importance (gain)",
            xaxis_title="Gain",
            yaxis_title="Feature",
            yaxis={"categoryorder": "total ascending"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            height=420,
        )
        st.plotly_chart(importance_fig, use_container_width=True)

with report_tab:
    st.write(
        f"PDF for {product}, {horizon}-day {active_model} forecast. "
        "It includes both models' MAE and RMSE."
    )
    safe_product = "".join(ch if ch.isalnum() else "_" for ch in product).strip("_")
    if not safe_product:
        safe_product = "forecast"
    pdf_bytes = build_report_pdf(
        product=product,
        horizon=int(horizon),
        metrics=metrics,
        history=history,
        forecast=result["forecasts"][active_model],
    )
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"sales_forecast_{safe_product}_{horizon}d.pdf",
        mime="application/pdf",
    )
