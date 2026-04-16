# DC Bike Rentals — Neural Network + XGBoost Ensemble Forecasting

Regression model forecasting hourly bike rental counts for a Washington D.C. bikeshare system by ensembling a Neural Network and XGBoost, averaging their predictions for the final output.

Built as the final project for the BYU-Idaho CSE 450 Machine Learning course (team project — this repository contains my individual contribution).

---

## Problem

City planners and bikeshare operators need reliable hourly demand forecasts to allocate bikes and docking stations efficiently. The challenge is that ridership is driven by overlapping signals — commuter patterns, weather, seasonality, and unpredictable events like political rallies.

---

## Technical Approach

| Step | Detail |
|---|---|
| **Dataset** | Hourly bike rental records from D.C. · weather, seasonal, and temporal features |
| **Target** | `total` = `casual + registered` riders per hour |
| **Split** | Temporal: last 20% of records as test set — no random shuffle to respect time ordering |
| **Model 1** | Neural Network (TensorFlow/Keras) — 3-layer MLP with BatchNorm + Dropout + EarlyStopping |
| **Model 2** | XGBoost Regressor — handles feature interactions and tabular structure |
| **Ensemble** | Simple average of NN and XGBoost predictions |
| **Evaluation** | RMSE and R² on held-out test set |

The ensemble outperforms both individual models on test RMSE and R², reducing variance by combining the NN's smooth temporal pattern learning with XGBoost's sharp feature interaction handling.

---

## Feature Engineering

- **Datetime decomposition** — `month`, `day`, `day_of_week`, `year` extracted from `dteday`
- **`high_event_day` flag** — days with casual ridership above the 95th percentile are flagged (political events / major holidays drive extreme spikes)
- **Dropped leaky columns** — `casual` and `registered` are components of the target and excluded

---

## Key Insight

Investigating days with extremely high casual ridership (> 1,000 rentals) revealed that political events and national holidays — not just weather — drive the largest spikes. Encoding this as a binary flag improved both models.

---

## Skills

`Python` · `TensorFlow/Keras` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `lets-plot` · Neural Networks · Ensemble Methods · Feature Engineering · Regression · Time Series Forecasting

---

## Limitations

- Simple average ensemble assumes equal model contribution — a stacked (meta-learner) ensemble could improve results
- `high_event_day` is derived from the training data; in production it would require a known event calendar
- No lag features (`cnt_lag_1h`, `cnt_lag_24h`) — adding these would capture temporal autocorrelation
- Models trained on 2011–2012 data; distribution shift to current conditions would require retraining
