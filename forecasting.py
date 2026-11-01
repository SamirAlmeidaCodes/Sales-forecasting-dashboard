"""Clean daily sales and forecast revenue with Prophet and XGBoost."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

class DataError(ValueError):
    """The upload cannot be cleaned or is too short to forecast."""


MIN_POINTS = 120
# Two years. Longer history than this mostly repeats a stale revenue level.
MAX_TRAINING_WINDOW = 730
# Window used until the series can grow and still keep four holdout origins.
_BASE_TRAINING_WINDOW = 180
EVAL_HORIZON = 14
EVAL_STEP = 14
MAX_ORIGINS = 4
FORECAST_HORIZON = 90
ALL_PRODUCTS = "All products"
MODEL_PROPHET = "Prophet"
MODEL_XGBOOST = "XGBoost"
FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_28",
    "dayofweek",
    "month",
    "sin_doy",
    "cos_doy",
]
REQUIRED_COLUMNS = ("date", "product", "quantity", "revenue")
_SIGMA_EPSILON = 1e-6
_CURRENCY_DROP = str.maketrans("", "", "$£€,\u00a0 ")


def parse_numeric(value: object) -> float:
    """Parse a number that may be currency text. Unparseable values are NaN."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<na>", "n/a", "na"}:
        return np.nan
    negative = len(text) >= 2 and text[0] == "(" and text[-1] == ")"
    if negative:
        text = text[1:-1].strip()
    text = text.translate(_CURRENCY_DROP).strip()
    if text == "" or text in {"-", "+", "."}:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if not np.isfinite(number):
        return np.nan
    return -abs(number) if negative else number


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return one daily row per product with quantity and revenue cleaned.

    Winsorizing runs on observed rows only. Missing calendar days are then
    filled with zero so those inserted days do not move the Tukey fences.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataError(
            "Sales data must be a table with columns date, product, quantity, revenue."
        )
    frame = _normalize_columns(df)
    if frame.empty:
        raise DataError("The CSV has no data rows.")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    frame["date"] = dates.dt.normalize()

    raw_product = frame["product"]
    missing_product = raw_product.isna()
    product = raw_product.astype(str).str.strip()
    missing_product = missing_product | product.eq("") | product.str.lower().isin(
        ["nan", "none", "<na>"]
    )
    frame["product"] = product
    frame = frame.loc[~missing_product].copy()
    frame = frame.dropna(subset=["date"])

    frame["quantity"] = frame["quantity"].map(parse_numeric)
    frame["revenue"] = frame["revenue"].map(parse_numeric)
    frame = frame.dropna(subset=["quantity", "revenue"])
    if frame.empty:
        raise DataError(
            "No usable rows after cleaning. Each row needs a date, a product, "
            "and numeric quantity and revenue."
        )

    grouped = frame.groupby(["date", "product"], as_index=False)[["quantity", "revenue"]].sum()
    parts: list[pd.DataFrame] = []
    for product_name, group in grouped.groupby("product", sort=False):
        group = group.sort_values("date").copy()
        group["quantity"] = _winsorize(group["quantity"])
        group["revenue"] = _winsorize(group["revenue"])
        full_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        filled = group.set_index("date").reindex(full_index)
        filled["product"] = product_name
        filled["quantity"] = filled["quantity"].fillna(0.0)
        filled["revenue"] = filled["revenue"].fillna(0.0)
        filled.index.name = "date"
        parts.append(filled.reset_index())

    clean = pd.concat(parts, ignore_index=True)
    clean = clean.sort_values(["product", "date"]).reset_index(drop=True)
    clean["quantity"] = clean["quantity"].astype(float)
    clean["revenue"] = clean["revenue"].astype(float)
    return clean


def revenue_series(clean: pd.DataFrame, product: str) -> pd.Series:
    """Daily revenue for one product, or the sum across products."""
    if product == ALL_PRODUCTS:
        grouped = clean.groupby("date")["revenue"].sum().sort_index()
        full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
        series = grouped.reindex(full_index, fill_value=0.0)
    else:
        part = clean.loc[clean["product"] == product, ["date", "revenue"]]
        if part.empty:
            raise DataError(f"Unknown product: {product}.")
        series = part.set_index("date")["revenue"].sort_index()
    return _prepare_series(series)


def quantity_total(clean: pd.DataFrame, product: str) -> float:
    if product == ALL_PRODUCTS:
        return float(clean["quantity"].sum())
    return float(clean.loc[clean["product"] == product, "quantity"].sum())


def select_winner(
    prophet_mae: float,
    prophet_rmse: float,
    xgb_mae: float,
    xgb_rmse: float,
) -> str:
    """Lower RMSE wins. A lower MAE breaks an RMSE tie. A full tie stays Prophet."""
    if xgb_rmse < prophet_rmse:
        return MODEL_XGBOOST
    if prophet_rmse < xgb_rmse:
        return MODEL_PROPHET
    if xgb_mae < prophet_mae:
        return MODEL_XGBOOST
    if prophet_mae < xgb_mae:
        return MODEL_PROPHET
    return MODEL_PROPHET


def training_window_length(n: int) -> int:
    """Fixed training window of at most two years, not an expanding one.

    Four 14-day origins need 56 days set aside. Once the series can give those
    origins a window of at least 180 days, the window grows up to 730. Shorter
    series keep the previous ladder: 180 days from 194 points, otherwise the
    length minus 14.
    """
    if n < MIN_POINTS:
        raise DataError(
            f"Need at least {MIN_POINTS} daily points after cleaning; got {n}."
        )
    room_for_four = n - EVAL_HORIZON - (MAX_ORIGINS - 1) * EVAL_STEP
    if room_for_four >= _BASE_TRAINING_WINDOW:
        return min(MAX_TRAINING_WINDOW, room_for_four)
    if n >= _BASE_TRAINING_WINDOW + EVAL_HORIZON:
        return _BASE_TRAINING_WINDOW
    return n - EVAL_HORIZON


def xgboost_params(n_usable: int) -> dict[str, float | int]:
    """Tree settings from how many lagged training rows the window contains.

    Depth stays at 3. More rows get smaller steps and a higher minimum leaf
    weight. The learning rate and the tree count move together.
    """
    if n_usable < 150:
        return {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.08,
            "min_child_weight": 3,
        }
    if n_usable < 400:
        return {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 5,
        }
    return {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 8,
    }


def prophet_yearly_order(window: int) -> int | bool:
    """Yearly Fourier order. False leaves yearly seasonality off."""
    if window < 365:
        return False
    if window < MAX_TRAINING_WINDOW:
        return 3
    return 6


def rolling_origins(n: int, training_window: int) -> list[int]:
    """Most recent holdout starts that still leave EVAL_HORIZON actuals, at most 4."""
    last = n - EVAL_HORIZON
    origins: list[int] = []
    origin = last
    while origin >= training_window and len(origins) < MAX_ORIGINS:
        origins.append(int(origin))
        origin -= EVAL_STEP
    origins.reverse()
    return origins


def feature_frame(series: pd.Series) -> pd.DataFrame:
    """Lag and calendar features. Rolling means are shifted so today is excluded."""
    values = series.astype(float)
    frame = pd.DataFrame(index=values.index)
    frame["lag_7"] = values.shift(7)
    frame["lag_14"] = values.shift(14)
    frame["lag_28"] = values.shift(28)
    prior = values.shift(1)
    frame["roll_mean_7"] = prior.rolling(7, min_periods=7).mean()
    frame["roll_mean_28"] = prior.rolling(28, min_periods=28).mean()
    frame["dayofweek"] = values.index.day_of_week.astype(float)
    frame["month"] = values.index.month.astype(float)
    # Sine and cosine keep December next to January. A raw month number does not.
    day_of_year = values.index.day_of_year.astype(float)
    angle = 2 * np.pi * day_of_year / 365.25
    frame["sin_doy"] = np.sin(angle)
    frame["cos_doy"] = np.cos(angle)
    return frame[FEATURE_COLUMNS]


def next_step_features(history: pd.Series) -> pd.DataFrame:
    """Features for the day after `history`, using only values already in `history`."""
    next_date = history.index[-1] + pd.Timedelta(days=1)
    placeholder = pd.Series(
        [np.nan],
        index=pd.DatetimeIndex([next_date]),
        name=history.name,
    )
    extended = pd.concat([history, placeholder])
    row = feature_frame(extended).iloc[[-1]]
    if row.isna().to_numpy().any():
        raise DataError(
            "Need more than 28 daily points in the training window to build lag features."
        )
    return row


def fit_xgboost_forecast(
    series: pd.Series,
    horizon: int,
    training_window: int,
    sigma: float,
) -> tuple[pd.DataFrame, XGBRegressor]:
    """Recursive multi-step forecast. Later steps see predictions, not future actuals."""
    history = _prepare_series(series)
    window = int(training_window)
    if window < 1 or window > len(history):
        raise DataError(
            f"Training window ({training_window}) does not fit the series ({len(history)} days)."
        )
    if int(horizon) < 1:
        raise DataError("Horizon must be at least 1 day.")
    train = history.iloc[-window:]
    model = _fit_xgboost(train)
    frame = _recursive_forecast(model, train, int(horizon), sigma)
    return frame, model


def feature_importance_frame(model: XGBRegressor) -> pd.DataFrame:
    names = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    scores = np.asarray(model.feature_importances_, dtype=float)
    frame = pd.DataFrame({"feature": names, "importance": scores})
    return frame.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(
        drop=True
    )


def evaluate_xgboost(series: pd.Series, training_window: int) -> dict[str, float | int]:
    """MAE, RMSE, and residual sigma on concatenated rolling-origin holdouts."""

    def point_forecast(train: pd.Series, horizon: int) -> pd.DataFrame:
        frame, _model = fit_xgboost_forecast(train, horizon, len(train), _SIGMA_EPSILON)
        return frame

    y_true, y_pred = _rolling_origin_predictions(series, training_window, point_forecast)
    mae, rmse = _mae_rmse(y_true, y_pred)
    residuals = y_true - y_pred
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "sigma": sigma,
        "n_origins": len(y_true) // EVAL_HORIZON,
    }


def run_product_forecast(clean: pd.DataFrame, product: str) -> dict[str, object]:
    """Score both models and refit each on the latest window for a 90-day forecast."""
    series = revenue_series(clean, product)
    n_points = int(len(series))
    window = training_window_length(n_points)
    xgb_eval = evaluate_xgboost(series, window)
    prophet_mae, prophet_rmse = _evaluate_prophet(series, window)
    winner = select_winner(
        prophet_mae,
        prophet_rmse,
        float(xgb_eval["mae"]),
        float(xgb_eval["rmse"]),
    )
    metrics = {
        "prophet_mae": prophet_mae,
        "prophet_rmse": prophet_rmse,
        "xgboost_mae": float(xgb_eval["mae"]),
        "xgboost_rmse": float(xgb_eval["rmse"]),
        "winner": winner,
        "xgb_sigma": float(xgb_eval["sigma"]),
        "training_window": window,
        "n_origins": int(xgb_eval["n_origins"]),
        "n_points": n_points,
    }
    prophet_fc = forecast_prophet(series, FORECAST_HORIZON, window)
    xgb_fc, xgb_model = fit_xgboost_forecast(
        series,
        FORECAST_HORIZON,
        window,
        metrics["xgb_sigma"],
    )
    return {
        "history": series,
        "metrics": metrics,
        "forecasts": {
            MODEL_PROPHET: prophet_fc,
            MODEL_XGBOOST: xgb_fc,
        },
        "importance": feature_importance_frame(xgb_model),
        "quantity_total": quantity_total(clean, product),
        "revenue_total": float(series.sum()),
    }


def forecast_prophet(
    series: pd.Series,
    horizon: int,
    training_window: int,
) -> pd.DataFrame:
    history = _prepare_series(series)
    window = int(training_window)
    if window < 1 or window > len(history):
        raise DataError(
            f"Training window ({training_window}) does not fit the series ({len(history)} days)."
        )
    train = history.iloc[-window:]
    model = _fit_prophet(train)
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                train.index[-1] + pd.Timedelta(days=1),
                periods=int(horizon),
                freq="D",
            )
        }
    )
    predicted = model.predict(future)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(predicted["ds"]).dt.normalize(),
            "yhat": predicted["yhat"].to_numpy(dtype=float),
            "yhat_lower": predicted["yhat_lower"].to_numpy(dtype=float),
            "yhat_upper": predicted["yhat_upper"].to_numpy(dtype=float),
        }
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = [str(column).strip().lstrip("\ufeff").strip().lower() for column in df.columns]
    frame = df.copy()
    frame.columns = renamed
    if frame.columns.duplicated().any():
        dupes = sorted({name for name in frame.columns[frame.columns.duplicated()]})
        raise DataError("Duplicate columns after normalizing names: " + ", ".join(dupes) + ".")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DataError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Required columns: date, product, quantity, revenue."
        )
    return frame.loc[:, list(REQUIRED_COLUMNS)].copy()


def _winsorize(values: pd.Series) -> pd.Series:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return values
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return values.clip(lower=lower, upper=upper)


def _prepare_series(series: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    prepared = pd.Series(np.asarray(series, dtype=float), index=index.normalize(), copy=True)
    prepared = prepared.sort_index()
    prepared.index.name = "date"
    prepared.name = "revenue"
    if prepared.index.has_duplicates:
        raise DataError("Revenue series has duplicate dates.")
    if len(prepared) >= 2:
        gaps = pd.Series(prepared.index).diff().iloc[1:]
        if not (gaps == pd.Timedelta(days=1)).all():
            raise DataError("Revenue series must be a complete daily index.")
    return prepared


def _fit_xgboost(series: pd.Series) -> XGBRegressor:
    frame = feature_frame(series).copy()
    frame["y"] = series.astype(float).to_numpy()
    train = frame.dropna()
    if train.empty:
        raise DataError(
            "Need more than 28 daily points in the training window to fit XGBoost."
        )
    settings = xgboost_params(len(train))
    model = XGBRegressor(
        n_estimators=int(settings["n_estimators"]),
        max_depth=int(settings["max_depth"]),
        learning_rate=float(settings["learning_rate"]),
        min_child_weight=float(settings["min_child_weight"]),
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        importance_type="gain",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"])
    return model


def _recursive_forecast(
    model: XGBRegressor,
    history: pd.Series,
    horizon: int,
    sigma: float,
) -> pd.DataFrame:
    extended = history.astype(float).copy()
    dates: list[pd.Timestamp] = []
    preds: list[float] = []
    for _ in range(horizon):
        row = next_step_features(extended)
        yhat = float(model.predict(row)[0])
        next_date = extended.index[-1] + pd.Timedelta(days=1)
        # Append the prediction, never the next actual, so later lags stay causal.
        extended.loc[next_date] = yhat
        dates.append(pd.Timestamp(next_date))
        preds.append(yhat)
    return _with_residual_band(dates, np.asarray(preds, dtype=float), sigma)


def _with_residual_band(
    dates: list[pd.Timestamp],
    yhat: np.ndarray,
    sigma: float,
) -> pd.DataFrame:
    band_sigma = float(sigma)
    if not np.isfinite(band_sigma) or band_sigma <= 0:
        band_sigma = _SIGMA_EPSILON
    # Homoscedastic residual interval, not a conformal interval.
    half_width = 1.96 * band_sigma
    return pd.DataFrame(
        {
            "date": dates,
            "yhat": yhat,
            "yhat_lower": yhat - half_width,
            "yhat_upper": yhat + half_width,
        }
    )


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def _fit_prophet(train: pd.Series):
    # Import here so unit tests can collect without Prophet installed.
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    train_df = pd.DataFrame({"ds": pd.to_datetime(train.index), "y": train.to_numpy(dtype=float)})
    window = len(train)
    yearly = prophet_yearly_order(window)
    # A longer window has more places to bend. Keep the trend smooth once a
    # full year is in the fit so the extra history estimates the slope.
    kwargs: dict[str, object] = {
        "weekly_seasonality": True,
        "yearly_seasonality": yearly,
        "daily_seasonality": False,
        "interval_width": 0.95,
    }
    if window >= 365:
        kwargs["changepoint_prior_scale"] = 0.01
    model = Prophet(**kwargs)
    model.fit(train_df)
    return model


def _rolling_origin_predictions(series: pd.Series, training_window: int, forecast):
    """Concatenated holdout actuals and point forecasts for the fixed window."""
    history = _prepare_series(series)
    window = int(training_window)
    origins = rolling_origins(len(history), window)
    if not origins:
        raise DataError(
            f"Not enough history for a {training_window}-day training window "
            f"and a {EVAL_HORIZON}-day holdout ({len(history)} daily points)."
        )
    actual_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for origin in origins:
        # The holdout and anything after it stay out of training and out of lags.
        train = history.iloc[origin - window : origin]
        actual = history.iloc[origin : origin + EVAL_HORIZON]
        predicted = forecast(train, EVAL_HORIZON)
        forecast_dates = pd.DatetimeIndex(pd.to_datetime(predicted["date"]))
        if not forecast_dates.equals(pd.DatetimeIndex(actual.index)):
            raise DataError("Forecast dates do not match the holdout.")
        actual_parts.append(actual.to_numpy(dtype=float))
        pred_parts.append(predicted["yhat"].to_numpy(dtype=float))
    return np.concatenate(actual_parts), np.concatenate(pred_parts)


def _evaluate_prophet(series: pd.Series, training_window: int) -> tuple[float, float]:
    def point_forecast(train: pd.Series, horizon: int) -> pd.DataFrame:
        return forecast_prophet(train, horizon, len(train))

    y_true, y_pred = _rolling_origin_predictions(series, training_window, point_forecast)
    return _mae_rmse(y_true, y_pred)
