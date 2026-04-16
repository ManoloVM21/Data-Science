# Data Science Portfolio (Python • ML • SQL)

This repository contains hands-on data science projects built with a focus on **Python**, **machine learning**, and **SQL-style data work** (querying, aggregation, joins/merges, and feature engineering). Recruiters can use this README as a guide to quickly understand the skills demonstrated in each project.

## Quick navigation

- **Data Cleaning Challenge** → `CHALLENGES/Data Cleaning/`
- **Data Manipulation & Visualization (Traffic Analysis)** → `CHALLENGES/Data Manipulation and Visualization/`
- **Full Projects** (multi-file projects) → `Full Projects/`
  1. [Donors](Full%20Projects/Donnors)
  2. [Bike Rentals](Full%20Projects/Bike%20Rentals%20per%20Hours%20-%20Neural%20Networks%20%26%20XGBoost)
  3. [House Pricing](Full%20Projects/House%20Price%20Predictions%20-%20XGBoost)
  4. [Bank Loan](Full%20Projects/Bank%20Loan%20Program%20-%20%20Random%20Forest)

---

## 1) Data Cleaning Challenge (Python / Pandas)

**Location:** `CHALLENGES/Data Cleaning/`

### What this project demonstrates

- **Data ingestion** from external sources (Kaggle)
- **Data quality handling** (invalid tokens such as `ERROR`, `UNKNOWN`)
- **Type enforcement** (casting columns to numeric/string/datetime)
- **Missing data strategy** (column-wise imputation)
- **Reproducible notebook-style reporting** (Quarto)

### Key skills shown

- **Python:** `pandas`, `numpy`, file system handling (`os`)
- **Data cleaning:**
  - Standardizing missing values (`replace(["ERROR","UNKNOWN"], np.nan)`)
  - Converting data types for analysis (`astype({...})`)
  - Filling missing values using:
    - mean/rounded mean for numeric columns
    - forward fill (`ffill`) for categorical/time columns

### Files

- `DataCleaning.qmd` — source notebook/report (Quarto)
- `DataCleaning.html` — rendered output

---

## 2) Data Manipulation & Visualization — Traffic Analysis (Python / SQL-style analysis)

**Location:** `CHALLENGES/Data Manipulation and Visualization/`

### What this project demonstrates

- **Large dataset sampling** and column selection for performance
- **Feature engineering** from timestamps (year/month/hour/season)
- **SQL-like aggregation in Pandas** (groupby, filters, counts)
- **Missing value handling** with median / “Unknown” strategies
- **Visualization & storytelling** (charts and written interpretation)

### Key skills shown

- **Python:** `pandas`, `numpy`
- **SQL-style data manipulation (in Pandas):**
  - Filtering subsets (equivalent to `WHERE`)
  - Aggregations (equivalent to `GROUP BY`, `COUNT`, `ORDER BY`)
  - Deriving columns (equivalent to computed fields)
- **Visualization:**
  - `lets_plot` grammar-of-graphics charts
  - `matplotlib` for supporting plots (including pie charts)

### Example analytical questions answered

- States with the highest accidents (2016–2017)
- Weather conditions most associated with accidents
- Most frequent accident hours (binned time windows)
- Seasonality / monthly trends and temperature relationship

### Files

- `project01.qmd` — source notebook/report (Quarto)
- `project01.html` — rendered output

---

## 3) Full Projects

**Location:** `Full Projects/`

1) **Donors** — [Full Projects/Donnors](Full%20Projects/Donnors)  
2) **Bike Rentals** — [Full Projects/Bike Rentals per Hours - Neural Networks & XGBoost](Full%20Projects/Bike%20Rentals%20per%20Hours%20-%20Neural%20Networks%20%26%20XGBoost)  
3) **House Pricing** — [Full Projects/House Price Predictions - XGBoost](Full%20Projects/House%20Price%20Predictions%20-%20XGBoost)  
4) **Bank Loan** — [Full Projects/Bank Loan Program -  Random Forest](Full%20Projects/Bank%20Loan%20Program%20-%20%20Random%20Forest)
---

## Tech stack

- **Python:** Pandas, NumPy, Matplotlib, Lets-Plot
- **ML workflow readiness:** cleaning, feature creation, dataset preparation (modeling extensions planned/iterated)
- **SQL / database concepts:** SQL-style querying patterns via Pandas; projects may also use `sqlite3` for local database workflows
- **Reporting:** Quarto (`.qmd` → `.html`) for recruiter-friendly reports

---

## How to run locally

### Option A — View the reports
Open the pre-rendered `.html` files inside each project folder.

### Option B — Reproduce the analysis
1. Install Python dependencies (example):
   ```bash
   pip install pandas numpy matplotlib lets-plot kagglehub
