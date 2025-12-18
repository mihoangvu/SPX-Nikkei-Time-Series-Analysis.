# Econometrics Lab: SPX–Nikkei Time Series Analysis

This project is a small **econometrics web application** built using Streamlit.

The app is designed as an interactive environment to apply standard econometric techniques to financial time-series data.  
In this project, the empirical analysis is conducted using **5-year S&P 500 (SPX) and Nikkei 225 (NKY) data**.

---

## Purpose of the App

The goal of this app is not to build a trading system, but to **demonstrate how econometric models are applied in practice**, similar to what is covered in an Econometrics course.

The app allows users to:
- Explore relationships between two financial time series
- Evaluate predictability of returns
- Analyze volatility dynamics

---

## Data

- Frequency: Weekly  
- Sample: 5 years  
- Indices used:  
  - S&P 500 (SPX)  
  - Nikkei 225 (NKY)

Prices are first aligned by date and then transformed into **log returns** before model estimation.

---

## Models Implemented

### OLS Regression
OLS is used to examine whether **Nikkei returns can explain S&P 500 returns**, testing for return spillover effects.

### ARIMA
ARIMA models are applied to **S&P 500 returns** to study short-term predictability and to evaluate forecast performance.

Forecasts are used mainly for **model evaluation and illustration**, rather than for making long-term market predictions.

### GARCH
GARCH(1,1) models are used to analyze **volatility dynamics and persistence** in returns.

---

## Interpretation

The app focuses on:
- Empirical testing
- Model assumptions
- Interpretation of results

rather than purely visualizing data.

Overall, the app translates standard econometric workflows into an **interactive web-based format**, using SPX–Nikkei 5-year data as a concrete example.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
