"""Matplotlib PDF report for the active sales forecast."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from forecasting import EVAL_HORIZON


def build_report_pdf(
    product: str,
    horizon: int,
    metrics: dict,
    history: pd.Series,
    forecast: pd.DataFrame,
) -> bytes:
    """PDF bytes with the comparison numbers and history plus forecast."""
    view = forecast.iloc[: int(horizon)]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.72)
    lines = [
        "Sales forecast report",
        f"Product: {product}",
        f"Horizon: {int(horizon)} days",
        f"Winner: {metrics['winner']}",
        (
            f"Prophet  MAE {metrics['prophet_mae']:.2f}"
            f"  RMSE {metrics['prophet_rmse']:.2f}"
        ),
        (
            f"XGBoost  MAE {metrics['xgboost_mae']:.2f}"
            f"  RMSE {metrics['xgboost_rmse']:.2f}"
        ),
        (
            f"Training window: {metrics['training_window']} days"
            f"   Origins: {metrics['n_origins']}"
            f"   Holdout: {EVAL_HORIZON} days"
        ),
    ]
    fig.text(
        0.08,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    ax.plot(
        history.index,
        history.to_numpy(dtype=float),
        label="Actual revenue",
        color="#1f77b4",
    )
    ax.plot(view["date"], view["yhat"], label="Forecast", color="#d62728")
    ax.fill_between(
        view["date"],
        view["yhat_lower"],
        view["yhat_upper"],
        color="#d62728",
        alpha=0.2,
        label="95% confidence band",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()
