# Sales forecasting

Streamlit app that cleans daily product sales and forecasts revenue. It trains Prophet and XGBoost, compares them on a rolling-origin window, and opens on the model with the lower error.

A human review of the training setup made that forecast 11% more precise on the bundled sample. The first version trained on the last 180 days only, with XGBoost fixed at 200 trees, depth 3, and a learning rate of 0.08. The revision trains on up to the last two years, and the settings in [Adaptive training settings](#adaptive-training-settings) follow how much history that window contains. On Widget, Gadget, and Doohickey, the model the app opens on still had the lower 14-day error. Pooling those three products, its RMSE fell from 4.61 to 4.12.

## CSV schema

Required columns (matched case-insensitively after stripping spaces):

- `date` — day of the sale
- `product` — product name
- `quantity` — units sold (cleaned and summarized, not forecasted)
- `revenue` — daily revenue, the forecast target

Currency text is accepted (`$1,234.50`, `£`, `€`, thousands commas, and accounting negatives such as `(25.00)`). Values that still are not numbers are dropped. Duplicate product-days are summed. Missing days inside a product's date span are stored as zero. Within each product, quantity and revenue on the observed days are winsorized to the Tukey fences `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`. If the IQR is 0 the series is left unchanged. Outlier rows are not deleted.

The product list includes `All products` (revenue summed across products on each day) and every product name. A series must have at least 120 daily points after cleaning.

Uploads larger than 50 MB are rejected by the Streamlit server config.

A bundled sample lives at `data/sample_sales.csv` (Widget, Gadget, and Doohickey, about 18 months). Use **Load bundled sample** to try the app without uploading a file.

## Run with Docker

From this directory:

```bash
docker compose up
```

The image is `python:3.11-slim` so Prophet can install. Open http://localhost:8501

## Local venv

`pip install -r requirements.txt` then `streamlit run app.py` from this directory. Prophet wheels target the Python 3.11 image above. If the host interpreter is 3.13 and the Prophet wheel does not install, use Docker.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## How the models are compared

Evaluation uses a fixed-length training window, not an expanding one. The length is chosen from the series as described under [Adaptive training settings](#adaptive-training-settings), and it never exceeds 730 days. The holdout is 14 days and the origin steps by 14 days. At most 4 origins are scored, using the most recent ones that leave 14 days of actuals after the origin. Each fold trains only on the window that ends the day before its holdout. Holdout rows are not used as lags.

MAE and RMSE are computed on the concatenated holdout days, so each day has the same weight. RMSE is the square root of the mean squared error, not the MSE. The lower RMSE wins. If the RMSE ties, the lower MAE wins. If both tie, Prophet is the default.

The app refits on the most recent training window and forecasts 90 days. Choosing 30 or 60 days shows a prefix of that forecast and does not refit. Prophet's shaded band is a 95% interval (`yhat_lower` / `yhat_upper`). XGBoost's shaded band is a residual interval, prediction ± 1.96 standard deviations of that model's rolling-origin residuals. That is a homoscedastic residual interval, not a conformal interval.

Feature importance (gain) is shown only while XGBoost is the active model. The report tab downloads a matplotlib PDF with the selected product, horizon, both models' MAE and RMSE, the winner, and a chart of history plus the forecast and confidence band.

## Adaptive training settings

The window length and both models' settings are a fixed function of the cleaned series. They are not searched on the holdout days that choose the winner.

The training window is capped at 730 days, about two years. Four 14-day origins need 56 days set aside, so from 236 points upward the window is `min(730, length - 56)`. From 194 through 235 points it stays 180 days. From 120 through 193 points it is the series length minus 14 days. On the bundled sample (547 days) the window is 491 days. A file of about three years stops at 730.

Prophet keeps weekly seasonality on and daily seasonality off. Yearly seasonality stays off while the window is under 365 days. From 365 through 729 days it uses Fourier order 3, and at 730 days it uses order 6. Once the window covers a year, the trend is smoothed with `changepoint_prior_scale` 0.01 so the extra history estimates the slope instead of bending around single weeks.

XGBoost predicts from nine fields: revenue 7, 14, and 28 days ago, the mean of the prior 7 and prior 28 days (the current day is excluded), day of week, month, and the sine and cosine of the day of year. The sine and cosine keep December next to January. The forecast is recursive: predict the next day, append that prediction, recompute the lags, and repeat.

Tree depth stays 3. Tree count, learning rate, and minimum leaf weight follow how many rows still have those lags (the window minus the 28-day warmup):

| Lagged rows | Trees | Learning rate | Minimum leaf weight |
|---|---|---|---|
| Under 150 | 200 | 0.08 | 3 |
| 150 to 399 | 400 | 0.05 | 5 |
| 400 and above | 500 | 0.05 | 8 |

A longer window takes smaller steps and asks each leaf to cover more days. Row and column sampling stay at 0.9, and the fit uses one thread with a fixed random seed.
