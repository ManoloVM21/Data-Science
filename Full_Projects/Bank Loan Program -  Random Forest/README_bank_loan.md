# Bank Marketing Campaign — Term Deposit Subscription Prediction

Binary classification model predicting whether a bank client will subscribe to a term deposit based on a Portuguese bank's direct marketing campaign data.

Built as part of the BYU-Idaho CSE 450 Machine Learning course.

---

## Problem

The bank ran direct phone marketing campaigns to sell term deposits. Contacting every client is expensive — the goal is to identify which clients are most likely to subscribe so the campaign can be targeted more efficiently.

---

## Technical Approach

| Step | Detail |
|---|---|
| **Dataset** | 37,069 records · 20 features — demographic, financial, and campaign contact attributes |
| **Target** | `y` (binary: subscribed / not subscribed) |
| **Class balance** | ~11% positive rate — accuracy is a misleading metric; model evaluated on F1 and AUC-ROC |
| **Model** | Random Forest Classifier with `class_weight='balanced'` |
| **Tuning** | GridSearchCV — 5-fold CV, optimizing F1 |
| **Threshold** | Precision-Recall curve used to select optimal classification threshold for business tradeoff |

---

## Feature Engineering

- **Education** — ordinal encoded (illiterate → university.degree) preserving natural order
- **`pdays`** — converted to binary `contacted_before` flag (999 = never contacted)
- **`default`, `housing`, `loan`** — binary yes/no (unknown → 0)
- **`job`, `marital`, `contact`, `poutcome`, `month`, `day_of_week`** — one-hot encoded
- **Economic indicators** — kept as numeric (`euribor3m`, `emp.var.rate`, `cons.price.idx`, etc.) — RF handles multicollinearity implicitly

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | ~0.79 |
| F1 (positive class) | ~0.48 |

ROC and Precision-Recall curves included to support threshold selection based on campaign cost/revenue estimates.

---

## Skills

`Python` · `scikit-learn` · `Random Forest` · `pandas` · `NumPy` · `Matplotlib` · `lets-plot` · Classification · Feature Engineering · Class Imbalance Handling · Threshold Optimization

---

## Limitations

- Economic indicators (`euribor3m`, `emp.var.rate`, `nr.employed`) are highly correlated — VIF analysis or PCA recommended before production deployment
- `unknown` values in binary columns treated as `no` — may introduce bias
- No temporal split: dataset covers multiple campaigns run at different points in time; a time-based split would be more realistic
