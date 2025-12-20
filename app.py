import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model


# =========================
# Config
# =========================
st.set_page_config(
    page_title="💣 Econometrics Bomb Challenge — SPX × Nikkei",
    page_icon="💣",
    layout="wide",
)

@dataclass
class DataConfig:
    sheet_name: str = "Worksheet"
    header_row: int = 6
    date_col: int = 0
    value_col: int = 1


# =========================
# Helpers
# =========================
def plot_line(df: pd.DataFrame, title: str, height: int = 380):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

def plot_series(series: pd.Series, title: str, height: int = 380):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=series.name))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

def compute_log_returns(price_df: pd.DataFrame, col: str) -> pd.Series:
    logp = np.log(price_df[col].astype(float))
    ret = logp.diff().dropna()
    ret.name = f"{col}_ret"
    return ret

@st.cache_data(show_spinner=False)
def load_weekly_xlsx(file_bytes: bytes, value_name: str, cfg: DataConfig) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    df = pd.read_excel(
        bio,
        sheet_name=cfg.sheet_name,
        header=cfg.header_row,
        usecols=[cfg.date_col, cfg.value_col],
        engine="openpyxl",
    )
    df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: value_name})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    return df

def make_demo():
    rng = pd.date_range("2019-01-04", periods=320, freq="W-FRI")
    eps = np.random.normal(0, 1, size=(len(rng), 2))
    eps[:, 1] = 0.6 * eps[:, 0] + np.sqrt(1 - 0.6**2) * eps[:, 1]
    spx = 2600 + np.cumsum(eps[:, 0] * 10 + 3)
    nky = 20000 + np.cumsum(eps[:, 1] * 80 + 10)
    return pd.DataFrame({"SPX": spx}, index=rng), pd.DataFrame({"NKY": nky}, index=rng)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def bomb_header(level: int, total: int = 4):
    st.markdown("## 💣 Econometrics Bomb Challenge")
    st.caption("Defuse the bomb by making correct econometric decisions using SPX–Nikkei data (5Y weekly).")
    st.progress(level / total)
    cols = st.columns([1, 2, 1])
    cols[0].metric("Level", f"{level}/{total}")
    cols[1].metric("Status", "🔥 ACTIVE" if level <= total else "✅ DEFUSED")
    cols[2].metric("Lives", f"{st.session_state.lives} ❤️")

def explode(msg: str):
    st.session_state.lives -= 1
    st.error(f"💥 BOOM! {msg}")
    if st.session_state.lives <= 0:
        st.session_state.game_over = True
        st.session_state.level = 1

def success(msg: str):
    st.success(f"✅ {msg}")
    st.balloons()

def next_level():
    st.session_state.level += 1

def reset_game():
    st.session_state.level = 1
    st.session_state.lives = 3
    st.session_state.game_over = False
    st.session_state.answers = {}


# =========================
# Session State
# =========================
if "level" not in st.session_state:
    reset_game()

if "answers" not in st.session_state:
    st.session_state.answers = {}

# =========================
# Sidebar — Data + Settings
# =========================
st.sidebar.header("📦 Data Setup")
use_demo = st.sidebar.toggle("Use demo data (no upload)", value=False)

cfg = DataConfig(
    sheet_name=st.sidebar.text_input("Sheet name", value="Worksheet"),
    header_row=int(st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=200, value=6, step=1)),
    date_col=int(st.sidebar.number_input("Date column index", min_value=0, max_value=50, value=0, step=1)),
    value_col=int(st.sidebar.number_input("Value column index", min_value=0, max_value=50, value=1, step=1)),
)

spx_file = None
nky_file = None
if not use_demo:
    spx_file = st.sidebar.file_uploader("Upload SPX (Index A) .xlsx", type=["xlsx"], key="spx")
    nky_file = st.sidebar.file_uploader("Upload NKY (Index B) .xlsx", type=["xlsx"], key="nky")

st.sidebar.divider()
st.sidebar.header("🧠 Model Settings")
h = int(st.sidebar.slider("Forecast horizon (weeks)", min_value=4, max_value=104, value=52, step=4))
p = int(st.sidebar.slider("AR order p", 0, 5, 1))
d = int(st.sidebar.slider("Differencing d", 0, 2, 0))
q = int(st.sidebar.slider("MA order q", 0, 5, 1))

st.sidebar.divider()
st.sidebar.header("🎮 Game Control")
if st.sidebar.button("Reset game", type="secondary"):
    reset_game()
st.sidebar.caption("Tip: for video demo, use ARIMA(1,0,1) and horizon 52 weeks.")


# =========================
# Load Data
# =========================
def load_data():
    if use_demo:
        return make_demo()
    if spx_file is None or nky_file is None:
        return None, None
    spx = load_weekly_xlsx(spx_file.getvalue(), "SPX", cfg)
    nky = load_weekly_xlsx(nky_file.getvalue(), "NKY", cfg)
    return spx, nky

spx_df, nky_df = load_data()

if spx_df is None or nky_df is None:
    st.info("⬅️ Upload both files in the sidebar (or turn on demo data) to start the challenge.")
    st.stop()

common = spx_df.join(nky_df, how="inner").dropna()
prices = common[["SPX", "NKY"]]

spx_ret = compute_log_returns(prices, "SPX")
nky_ret = compute_log_returns(prices, "NKY")
rets = pd.concat([spx_ret, nky_ret], axis=1).dropna()

if len(rets) < 80:
    st.warning("Sample is quite small after alignment. Consider uploading longer history.")
# =========================
# Precompute Models (for the game)
# =========================
# OLS: SPX_ret ~ const + NKY_ret
y = rets["SPX_ret"]
x = rets["NKY_ret"]
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
beta = float(ols.params.get("NKY_ret", np.nan))
beta_p = float(ols.pvalues.get("NKY_ret", np.nan))
r2 = float(ols.rsquared)

# ARIMA on SPX returns
series = rets["SPX_ret"]
if len(series) > h + 20:
    train = series.iloc[:-h]
    test = series.iloc[-h:]
    arima_model = SARIMAX(
        train,
        order=(p, d, q),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    arima_res = arima_model.fit(disp=False)
    fcst = arima_res.get_forecast(steps=h).predicted_mean
    fcst.index = test.index
    arima_rmse = rmse(test, fcst)
    arima_mae = mae(test, fcst)

    # Benchmark: train mean forecast
    bench = pd.Series(np.repeat(train.mean(), len(test)), index=test.index)
    bench_rmse = rmse(test, bench)
    bench_mae = mae(test, bench)
else:
    train = test = fcst = None
    arima_res = None
    arima_rmse = arima_mae = np.nan
    bench_rmse = bench_mae = np.nan

# GARCH on NKY returns
r = rets["NKY_ret"] * 100.0
gmod = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant")
gres = gmod.fit(disp="off")
omega = float(gres.params.get("omega", np.nan))
alpha1 = float(gres.params.get("alpha[1]", np.nan))
beta1 = float(gres.params.get("beta[1]", np.nan))
persist = alpha1 + beta1
cond_vol = (gres.conditional_volatility / 100.0).rename("Conditional Volatility")


# =========================
# Main UI
# =========================
if st.session_state.game_over:
    st.error("💀 Game Over. You ran out of lives.")
    if st.button("Try again"):
        reset_game()

bomb_header(st.session_state.level)

# Top overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Obs (prices)", f"{len(prices):,}")
c2.metric("Obs (returns)", f"{len(rets):,}")
c3.metric("Start", str(prices.index.min().date()))
c4.metric("End", str(prices.index.max().date()))

st.divider()

left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.subheader("🧾 Case Study Data")
    plot_line(prices, "Weekly Prices (SPX & NKY)", height=320)
    plot_line(rets, "Weekly Log Returns", height=320)

with right:
    # =========================
    # LEVEL 1
    # =========================
    if st.session_state.level == 1:
        st.subheader("💣 Level 1 — Stationarity Decision")
        st.write("You see prices and returns. What should we model for standard time-series econometrics in finance?")

        st.info("Hint: Prices are often non-stationary; returns are closer to stationary.")

        a, b = st.columns(2)
        if a.button("A) Model PRICES", use_container_width=True):
            explode("Prices are typically non-stationary → risk of spurious regression.")
            st.session_state.answers["L1"] = "Prices"
        if b.button("B) Model RETURNS ✅", use_container_width=True):
            st.session_state.answers["L1"] = "Returns"
            success("Correct. We model returns for stationarity and meaningful inference.")
            next_level()

        st.markdown("**Why this matters:** Modeling returns helps satisfy stationarity assumptions for OLS/ARIMA/GARCH.")

    # =========================
    # LEVEL 2
    # =========================
    elif st.session_state.level == 2:
        st.subheader("💣 Level 2 — OLS Spillover Test")
        st.write("We regress **SPX returns** on **NKY returns**. Decide whether there is evidence of spillover.")

        # Show key stats
        m1, m2, m3 = st.columns(3)
        m1.metric("β (NKY_ret)", f"{beta:.4f}")
        m2.metric("p-value(β)", f"{beta_p:.4g}")
        m3.metric("R²", f"{r2:.3f}")

        with st.expander("Show regression diagnostics"):
            resid = ols.resid
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rets.index, y=resid, mode="lines", name="Residuals"))
            fig.update_layout(title="OLS residuals over time", height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        st.write("Question: **Is there evidence that Nikkei returns explain SPX returns?**")

        a, b = st.columns(2)
        if a.button("A) YES — β is significant ✅", use_container_width=True):
            if np.isfinite(beta_p) and beta_p < 0.05:
                st.session_state.answers["L2"] = "Yes"
                success("Correct. β is statistically significant at 5% → evidence of a relationship.")
                next_level()
            else:
                explode("Not quite. In this sample, β is not significant at 5%.")
        if b.button("B) NO — relationship is random", use_container_width=True):
            if np.isfinite(beta_p) and beta_p < 0.05:
                explode("β is significant at 5%, so 'NO' is not supported by the regression.")
            else:
                st.session_state.answers["L2"] = "No"
                success("Correct. β is not significant at 5% → no strong evidence of spillover.")
                next_level()

        st.caption("Interpretation is based on standard hypothesis testing: H0: β = 0.")

    # =========================
    # LEVEL 3
    # =========================
    elif st.session_state.level == 3:
        st.subheader("💣 Level 3 — ARIMA Forecast Reality Check")
        st.write(f"We fit ARIMA({p},{d},{q}) on **SPX returns** and forecast the last **{h} weeks**.")

        if train is None or test is None:
            st.warning("Not enough observations for the chosen forecast horizon. Reduce horizon or upload more data.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ARIMA order", f"({p},{d},{q})")
            m2.metric("RMSE (ARIMA)", f"{arima_rmse:.6f}")
            m3.metric("RMSE (Benchmark)", f"{bench_rmse:.6f}")
            m4.metric("Winner", "ARIMA ✅" if arima_rmse < bench_rmse else "Benchmark ✅")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train, mode="lines", name="Train"))
            fig.add_trace(go.Scatter(x=test.index, y=test, mode="lines", name="Test"))
            fig.add_trace(go.Scatter(x=fcst.index, y=fcst, mode="lines", name="ARIMA forecast"))
            fig.add_trace(go.Scatter(x=test.index, y=np.repeat(train.mean(), len(test)), mode="lines", name="Benchmark (train mean)", line=dict(dash="dot")))
            fig.update_layout(title="Out-of-sample forecast on SPX returns", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

            st.write("Question: **Is return predictability strong here?**")

            a, b = st.columns(2)
            if a.button("A) YES — forecasts are reliable", use_container_width=True):
                # strong predictability only if ARIMA clearly beats benchmark
                if arima_rmse < 0.9 * bench_rmse:
                    st.session_state.answers["L3"] = "Strong"
                    success("Correct. ARIMA meaningfully improves over the benchmark.")
                    next_level()
                else:
                    explode("Not supported. ARIMA does not clearly outperform the benchmark.")
            if b.button("B) NO — predictability is weak ✅", use_container_width=True):
                if arima_rmse < 0.9 * bench_rmse:
                    explode("ARIMA strongly outperforms benchmark, so predictability may be non-trivial.")
                else:
                    st.session_state.answers["L3"] = "Weak"
                    success("Correct. Financial returns are often hard to forecast; benchmark performs similarly.")
                    next_level()

            st.caption("Forecasting here is for model evaluation, not long-term market prediction.")

    # =========================
    # LEVEL 4
    # =========================
    elif st.session_state.level == 4:
        st.subheader("💣 Level 4 — GARCH Volatility Persistence")
        st.write("We estimate a GARCH(1,1) on **NKY returns** and inspect volatility persistence.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ω", f"{omega:.6f}")
        m2.metric("α1", f"{alpha1:.4f}")
        m3.metric("β1", f"{beta1:.4f}")
        m4.metric("α+β", f"{persist:.4f}")

        plot_series(cond_vol, "Conditional Volatility (NKY)", height=320)

        st.write("Question: **What does α + β close to 1 imply?**")

        a, b = st.columns(2)
        if a.button("A) Volatility is random", use_container_width=True):
            explode("Not correct. α+β close to 1 implies persistence, not randomness.")
        if b.button("B) Volatility is highly persistent ✅", use_container_width=True):
            st.session_state.answers["L4"] = "Persistent"
            success("Correct. α+β near 1 indicates highly persistent volatility (IGARCH-like behavior).")
            next_level()

        st.info(
            "If the volatility plot looks nearly flat, that can happen when volatility evolves smoothly due to high persistence—"
            "especially with weekly index returns."
        )

    # =========================
    # DEFUSED SCREEN
    # =========================
    else:
        st.subheader("✅ Bomb Defused — Key Findings")
        st.success("You successfully completed all econometric decision steps.")

        st.markdown("### Summary (auto)")
        st.write(f"- **Level 1:** Modeled **returns** instead of prices.")
        st.write(f"- **Level 2 (OLS):** β = {beta:.4f}, p = {beta_p:.4g} → " +
                 ("evidence of relationship/spillover." if beta_p < 0.05 else "no strong evidence at 5%."))
        if train is not None:
            st.write(f"- **Level 3 (ARIMA):** RMSE ARIMA = {arima_rmse:.6f}, benchmark = {bench_rmse:.6f} → " +
                     ("ARIMA improves." if arima_rmse < bench_rmse else "predictability weak (similar to benchmark)."))
        st.write(f"- **Level 4 (GARCH):** α+β = {persist:.4f} → " +
                 ("high volatility persistence." if persist >= 0.95 else "moderate persistence."))

        st.markdown("### Export cleaned dataset")
        out = pd.concat([prices, rets], axis=1).dropna().reset_index().rename(columns={"index": "Date"})
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="spx_nky_cleaned_data.csv", mime="text/csv")

        st.markdown("### Replay?")
        if st.button("Play again"):
            reset_game()

st.divider()
st.caption("Built for Econometrics coursework: OLS for spillover testing, ARIMA for forecast evaluation, and GARCH for volatility dynamics.")
