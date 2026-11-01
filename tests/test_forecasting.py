"""Tests that do not fit Prophet."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import (
    ALL_PRODUCTS,
    FEATURE_COLUMNS,
    MODEL_PROPHET,
    MODEL_XGBOOST,
    DataError,
    clean_sales,
    evaluate_xgboost,
    feature_frame,
    feature_importance_frame,
    fit_xgboost_forecast,
    next_step_features,
    parse_numeric,
    revenue_series,
    prophet_yearly_order,
    rolling_origins,
    select_winner,
    training_window_length,
    xgboost_params,
)


def test_currency_parsing_commas_symbol_and_parentheses():
    assert parse_numeric("$1,234.50") == pytest.approx(1234.50)
    assert parse_numeric("(25.00)") == pytest.approx(-25.0)
    assert parse_numeric("£2,000.00") == pytest.approx(2000.0)
    assert parse_numeric("€ 15") == pytest.approx(15.0)
    assert parse_numeric("($50.00)") == pytest.approx(-50.0)
    assert parse_numeric("1,200") == pytest.approx(1200.0)
    assert pd.isna(parse_numeric("not-a-number"))

    dollars = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            " PRODUCT ": ["Gadget", "Gadget", "Gadget"],
            "Quantity": ["1,000", "1,000", "1,000"],
            "Revenue": ["$1,234.50", "$1,234.50", "not-a-number"],
        }
    )
    cleaned = clean_sales(dollars)
    assert cleaned["revenue"].tolist() == pytest.approx([1234.50, 1234.50])
    assert cleaned["quantity"].tolist() == pytest.approx([1000.0, 1000.0])
    assert list(cleaned["date"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]

    negatives = pd.DataFrame(
        {
            "date": ["2024-02-01", "2024-02-02"],
            "product": ["Widget", "Widget"],
            "quantity": [1, 1],
            "revenue": ["(25.00)", "(25.00)"],
        }
    )
    assert clean_sales(negatives)["revenue"].tolist() == pytest.approx([-25.0, -25.0])


def test_duplicate_days_sum_then_missing_days_are_zero():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-03"],
            "product": ["Widget", "Widget", "Widget"],
            "quantity": [1, 1.5, 2],
            "revenue": [10, 5, 7],
        }
    )
    cleaned = clean_sales(frame)
    assert len(cleaned) == 3
    by_date = cleaned.set_index("date")
    assert by_date.loc[pd.Timestamp("2024-01-01"), "quantity"] == pytest.approx(2.5)
    assert by_date.loc[pd.Timestamp("2024-01-01"), "revenue"] == pytest.approx(15)
    assert by_date.loc[pd.Timestamp("2024-01-02"), "quantity"] == 0
    assert by_date.loc[pd.Timestamp("2024-01-02"), "revenue"] == 0
    assert by_date.loc[pd.Timestamp("2024-01-03"), "revenue"] == pytest.approx(7)


def test_filled_zeros_are_not_winsorized():
    dates = [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
        "2024-01-09",
    ]
    revenue = [10, 12, 11, 13, 10, 12, 11, 40]
    frame = pd.DataFrame(
        {
            "date": dates,
            "product": "Widget",
            "quantity": 1,
            "revenue": revenue,
        }
    )
    observed = pd.Series(revenue, dtype=float)
    q1 = float(observed.quantile(0.25))
    q3 = float(observed.quantile(0.75))
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    cleaned = clean_sales(frame)
    assert len(cleaned) == 9
    january_5 = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert january_5["revenue"] == 0
    assert january_5["quantity"] == 0
    assert cleaned["revenue"].max() == pytest.approx(high)
    assert high < 40


def test_winsorize_clips_outlier_without_dropping_the_row():
    dates = pd.date_range("2024-01-01", periods=31, freq="D")
    revenue = list(range(1, 31)) + [100000]
    quantity = list(range(1, 31)) + [100000]
    observed = pd.Series(revenue, dtype=float)
    q1 = float(observed.quantile(0.25))
    q3 = float(observed.quantile(0.75))
    high = q3 + 1.5 * (q3 - q1)
    frame = pd.DataFrame(
        {
            "date": dates,
            "product": "Widget",
            "quantity": quantity,
            "revenue": revenue,
        }
    )
    cleaned = clean_sales(frame)
    assert len(cleaned) == 31
    assert pd.Timestamp("2024-01-31") in set(cleaned["date"])
    assert cleaned["revenue"].max() == pytest.approx(high)
    assert cleaned["quantity"].max() == pytest.approx(high)
    assert high < 100000
    january_1 = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert january_1["revenue"] == pytest.approx(1.0)
    assert january_1["quantity"] == pytest.approx(1.0)


def test_missing_required_columns_raise():
    frame = pd.DataFrame(
        {"date": ["2024-01-01"], "product": ["Widget"], "quantity": [1]}
    )
    with pytest.raises(DataError, match="revenue"):
        clean_sales(frame)
    with pytest.raises(DataError, match="[Rr]equired columns"):
        clean_sales(pd.DataFrame())


def test_winner_selection():
    assert select_winner(1.0, 5.0, 4.0, 3.0) == MODEL_XGBOOST
    assert select_winner(4.0, 3.0, 1.0, 5.0) == MODEL_PROPHET
    assert select_winner(3.0, 5.0, 2.0, 5.0) == MODEL_XGBOOST
    assert select_winner(2.0, 5.0, 3.0, 5.0) == MODEL_PROPHET
    assert select_winner(2.0, 5.0, 2.0, 5.0) == MODEL_PROPHET


def test_training_window_and_recent_origins():
    assert training_window_length(120) == 106
    assert training_window_length(179) == 165
    assert training_window_length(193) == 179
    assert training_window_length(194) == 180
    assert training_window_length(235) == 180
    assert training_window_length(236) == 180
    assert training_window_length(400) == 344
    assert training_window_length(547) == 491
    assert training_window_length(786) == 730
    assert training_window_length(1095) == 730
    with pytest.raises(DataError, match="120"):
        training_window_length(119)
    assert rolling_origins(400, 344) == [344, 358, 372, 386]
    assert rolling_origins(120, 106) == [106]
    assert rolling_origins(1095, 730) == [1039, 1053, 1067, 1081]


def test_training_settings_follow_window_size():
    short = xgboost_params(149)
    medium = xgboost_params(150)
    long = xgboost_params(400)
    assert short == {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.08,
        "min_child_weight": 3,
    }
    assert medium["n_estimators"] == 400
    assert medium["learning_rate"] == pytest.approx(0.05)
    assert medium["min_child_weight"] == 5
    assert long["n_estimators"] == 500
    assert long["min_child_weight"] == 8
    assert short["max_depth"] == medium["max_depth"] == long["max_depth"] == 3
    assert prophet_yearly_order(180) is False
    assert prophet_yearly_order(364) is False
    assert prophet_yearly_order(365) == 3
    assert prophet_yearly_order(729) == 3
    assert prophet_yearly_order(730) == 6


def test_next_day_features_use_only_prior_values():
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    series = pd.Series(np.arange(40, dtype=float), index=index)
    last = feature_frame(series).iloc[-1]
    assert last["lag_7"] == pytest.approx(32)
    assert last["lag_14"] == pytest.approx(25)
    assert last["lag_28"] == pytest.approx(11)
    assert last["roll_mean_7"] == pytest.approx(np.mean(np.arange(32, 39)))
    assert last["roll_mean_28"] == pytest.approx(np.mean(np.arange(11, 39)))

    row = next_step_features(series).iloc[0]
    next_date = index[-1] + pd.Timedelta(days=1)
    assert row["lag_7"] == pytest.approx(series.loc[next_date - pd.Timedelta(days=7)])
    assert row["lag_14"] == pytest.approx(series.loc[next_date - pd.Timedelta(days=14)])
    assert row["lag_28"] == pytest.approx(series.loc[next_date - pd.Timedelta(days=28)])
    prior_7 = series.loc[next_date - pd.Timedelta(days=7) : next_date - pd.Timedelta(days=1)]
    prior_28 = series.loc[next_date - pd.Timedelta(days=28) : index[-1]]
    assert len(prior_7) == 7
    assert len(prior_28) == 28
    assert row["roll_mean_7"] == pytest.approx(prior_7.mean())
    assert row["roll_mean_28"] == pytest.approx(prior_28.mean())
    assert row["dayofweek"] == pytest.approx(next_date.day_of_week)
    assert row["month"] == pytest.approx(next_date.month)
    angle = 2 * np.pi * float(next_date.day_of_year) / 365.25
    assert row["sin_doy"] == pytest.approx(np.sin(angle))
    assert row["cos_doy"] == pytest.approx(np.cos(angle))


def test_xgboost_recursive_forecast_does_not_copy_future_actuals():
    index = pd.date_range("2023-01-01", periods=160, freq="D")
    rng = np.random.default_rng(42)
    values = 80 + np.arange(160) * 0.15 + 6 * np.sin(np.arange(160) * 2 * np.pi / 7)
    values = values + rng.normal(0, 1.5, size=160)
    values[-14:] = np.arange(1, 15) * 1000
    series = pd.Series(values, index=index, name="revenue")
    history = series.iloc[:-14]
    holdout = series.iloc[-14:]

    forecast, model = fit_xgboost_forecast(
        history,
        horizon=14,
        training_window=len(history),
        sigma=5.0,
    )
    importance = feature_importance_frame(model)
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert importance["importance"].ge(0).all()
    assert float(importance["importance"].sum()) > 0
    assert len(forecast) == 14
    assert forecast["date"].iloc[0] == history.index[-1] + pd.Timedelta(days=1)
    assert not np.allclose(forecast["yhat"].to_numpy(), holdout.to_numpy())
    assert np.max(np.abs(forecast["yhat"].to_numpy() - holdout.to_numpy())) > 100
    assert not np.allclose(forecast["yhat"].to_numpy(), history.iloc[-14:].to_numpy())

    long_forecast, _long_model = fit_xgboost_forecast(
        history,
        horizon=90,
        training_window=min(180, len(history)),
        sigma=0.0,
    )
    assert len(long_forecast) == 90
    assert long_forecast["date"].iloc[-1] == history.index[-1] + pd.Timedelta(days=90)
    assert np.all(long_forecast["yhat_upper"] >= long_forecast["yhat"])
    assert np.all(long_forecast["yhat"] >= long_forecast["yhat_lower"])
    assert np.all(long_forecast["yhat_upper"] > long_forecast["yhat"])
    assert np.all(long_forecast["yhat"] > long_forecast["yhat_lower"])


def test_xgboost_rolling_origin_scores_recent_holdouts():
    index = pd.date_range("2022-01-01", periods=260, freq="D")
    values = 50 + 0.05 * np.arange(260) + 8 * np.sin(np.arange(260) * 2 * np.pi / 7)
    series = pd.Series(values, index=index)
    window = training_window_length(len(series))
    scored = evaluate_xgboost(series, window)
    assert window == 204
    assert scored["n_origins"] == 4
    assert np.isfinite(scored["mae"])
    assert np.isfinite(scored["rmse"])
    assert scored["rmse"] >= 0
    assert np.isfinite(scored["sigma"])
    assert scored["sigma"] > 0


def test_all_products_sums_daily_revenue():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
            "product": ["A", "A", "B", "B"],
            "quantity": [1, 1, 1, 1],
            "revenue": [10, 10, 5, 5],
        }
    )
    cleaned = clean_sales(frame)
    series = revenue_series(cleaned, ALL_PRODUCTS)
    assert series.loc[pd.Timestamp("2024-01-01")] == pytest.approx(15)
    assert series.loc[pd.Timestamp("2024-01-02")] == pytest.approx(10)
    assert series.loc[pd.Timestamp("2024-01-03")] == pytest.approx(5)
    assert len(series) == 3


def test_sample_csv_cleans_to_complete_daily_series():
    path = Path(__file__).resolve().parents[1] / "data" / "sample_sales.csv"
    raw = pd.read_csv(path)
    revenue_text = raw["revenue"].astype(str)
    assert revenue_text.str.contains(r"\$", regex=True).any()
    assert revenue_text.str.contains(r"\(25\.00\)", regex=True).any()
    assert revenue_text.str.contains("not-a-number").any()
    cleaned = clean_sales(raw)
    assert set(cleaned["product"]) == {"Widget", "Gadget", "Doohickey"}
    for _product, group in cleaned.groupby("product"):
        span = (group["date"].max() - group["date"].min()).days + 1
        assert len(group) == span
        assert not group["revenue"].isna().any()
        assert not group["quantity"].isna().any()
    assert cleaned["revenue"].max() < 10000
    assert cleaned.groupby("product").size().min() >= 120
