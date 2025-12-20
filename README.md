# 💣 Econometrics Bomb Challenge: SPX × Nikkei Time-Series Analysis

## Overview
This project is an interactive **econometrics web application** built using **Streamlit**.  
Instead of a traditional dashboard, the app is designed as a **game-based challenge**, where users must make correct econometric decisions to “defuse a bomb”.

The application translates core concepts from **time-series econometrics** into an engaging, decision-driven experience while preserving rigorous model implementation and interpretation.

---

## Objectives
The goal of this project is to:
- Apply standard econometric models to financial time-series data
- Evaluate forecasting performance and model assumptions
- Encourage correct econometric reasoning through interactive gameplay

This project focuses on **methodology and interpretation**, not trading or investment advice.

---

## Data
- **Assets**:  
  - S&P 500 Index (SPX)  
  - Nikkei 225 Index (NKY)
- **Frequency**: Weekly
- **Sample**: 5-year historical data
- **Preprocessing**:
  - Prices are converted into **log returns**
  - Data are aligned by date before analysis

Users can either:
- Upload their own `.xlsx` datasets, or  
- Use built-in demo data for quick testing

---

## Game Structure (Econometrics Bomb Challenge)

The app consists of **5 levels**, each corresponding to a key econometric concept.

### 💣 Level 1 — Stationarity
**Question**: Should we model prices or returns?  
- Tests understanding of non-stationarity and spurious regression  
- Correct choice: **Returns**

---

### 💣 Level 2 — OLS Spillover Analysis
**Model**:  
\[
r_{SPX,t} = \alpha + \beta r_{NKY,t} + \varepsilon_t
\]

- Users interpret regression output (β, p-value, R²)
- Tests hypothesis testing and interpretation
- Emphasizes that **OLS shows association, not causality**

---

### 💣 Level 3 — ARIMA Forecasting
**Model**: ARIMA(p, d, q) on SPX returns

- Out-of-sample forecasting
- Comparison against a benchmark (historical mean)
- Evaluation using RMSE and MAE
- Highlights the weak predictability of financial returns

---

### 💣 Level 4 — GARCH Volatility Modeling
**Model**: GARCH(1,1) on NKY returns

- Examines conditional volatility
- Interprets volatility persistence via \( \alpha + \beta \)
- Shows that volatility can be persistent even when it appears smooth

---

### 💣 Level 5 — Final Boss (Integrated Interpretation)
Users must select the **most econometrically valid takeaway**, reinforcing:
- Cautious interpretation
- Proper use of models
- Avoidance of overclaiming

---

## Game Mechanics
- ⏳ Time limits per level
- 💰 Gold & ❤️ lives system
- 🧾 Hints available at a cost
- 💥 Visual “bomb explosion” effects for incorrect answers
- 🧙‍♀️ Narrative elements (witch / dragon) to enhance engagement
- 🏁 Final scoreboard with accuracy, score, and time

---

## Key Econometric Takeaways
- Returns are preferred over prices due to stationarity concerns
- OLS tests statistical relationships, not causality
- ARIMA forecasting performance should be evaluated against benchmarks
- Financial return predictability is often weak
- Volatility exhibits strong persistence in financial markets

---

## Technology Stack
- **Python**
- **Streamlit**
- **Statsmodels** (OLS, ARIMA)
- **ARCH** package (GARCH)
- **Plotly** (interactive visualization)
- **Pandas / NumPy / Scikit-learn**

---

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
