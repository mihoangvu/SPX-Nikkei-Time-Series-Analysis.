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


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class DataConfig:
    sheet_name: str = "Worksheet"
    header_row: int = 6          # Bloomberg-style often has header at row 7 (0-indexed => 6)
    date_col: int = 0
    value_col: int = 1


@st.cache_data(show_spinner=False)
def load_weekly_xlsx(file_bytes: bytes, value_name: str, cfg: DataConfig) -> pd.DataFrame:
    """Load an XLSX (bytes) into a clean Date-indexed dataframe with one value column."""
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


def compute_log_returns(price_df: pd.DataFrame, col: str) -> pd.Series:
    logp = np.log(price_df[col].astype(float))
    ret = logp.diff().dropna()
    ret.name = f"{col}_ret"
    return ret


def plot_series(df: pd.DataFrame, title: str):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def metric_row(metrics: list[tuple[str, str]]):
    cols = st.columns(len(metrics))
    for c, (k, v) in zip(cols, metrics):
        c.metric(k, v)


def decision_badge(ok: bool, msg_ok: str, msg_no: str):
    """Small helper to display green/red callouts."""
    if ok:
        st.success(msg_ok)
    else:
        st.warning(msg_no)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Econometrics Lab — SPX–Nikkei",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Econometrics Lab: SPX–Nikkei Time Series Analysis")
st.caption(
    "Interactive dashboard for OLS (spillover), ARIMA (forecasting) and GARCH (volatility) on weekly SPX–Nikkei data."
)

with st.expander("What this app does (quick guide)", expanded=True):
    st.markdown(
        """
**Workflow**
1) Upload two weekly price series (SPX and Nikkei) **or** enable *demo data* in the sidebar.  
2) The app aligns dates and converts prices to **log returns**.  
3) Run:
- **OLS** to test whether Nikkei returns explain SPX returns (spillover).  
- **ARIMA** to forecast SPX returns and evaluate accuracy (with a baseline benchmark).  
- **GARCH** to model volatility dynamics and persistence.

**Note:** Forecasts here are mainly for **model evaluation and illustration**, not for “predicting the market”.
        """
    )


# -----------------------------
# Sidebar: data
# -----------------------------
st.sidebar.header("1) Data")
st.sidebar.write("Upload two Excel files (weekly series). If your export is Bloomberg-style, the defaults usually work.")

use_demo = st.sidebar.toggle("Use demo data (no upload)", value=False)

cfg = DataConfig(
    sheet_name=st.sidebar.text_input("Sheet name", value="Worksheet"),
    header_row=st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=200, value=6, step=1),
    date_col=st.sidebar.number_input("Date column index", min_value=0, max_value=50, value=0, step=1),
    value_col=st.sidebar.number_input("Value column index", min_value=0, max_value=50, value=1, step=1),
)

spx_file = None
nky_file = None
if not use_demo:
    spx_file = st.sidebar.file_uploader("Upload SPX (Index A) .xlsx", type=["xlsx"], key="spx")
    nky_file = st.sidebar.file_uploader("Upload Nikkei (NKY / Index B) .xlsx", type=["xlsx"], key="nky")

st.sidebar.divider()
st.sidebar.header("2) Model settings")
h = st.sidebar.slider("Forecast horizon (weeks)", min_value=4, max_value=104, value=52, step=4)
arima_p = st.sidebar.slider("AR order p", 0, 5, 1)
arima_d = st.sidebar.slider("Differencing d", 0, 2, 1)
arima_q = st.sidebar.slider("MA order q", 0, 5, 1)

st.sidebar.divider()
st.sidebar.header("3) Run")
run = st.sidebar.button("Run analysis", type="primary")


# -----------------------------
# Data creation / loading
# -----------------------------
def make_demo():
    rng = pd.date_range("2019-01-04", periods=320, freq="W-FRI")
    # Correlated random walks (demo only)
    eps = np.random.normal(0, 1, size=(len(rng), 2))
    eps[:, 1] = 0.6 * eps[:, 0] + np.sqrt(1 - 0.6**2) * eps[:, 1]
    spx = 2600 + np.cumsum(eps[:, 0] * 10 + 3)
    nky = 20000 + np.cumsum(eps[:, 1] * 80 + 10)
    return (
        pd.DataFrame({"SPX": spx}, index=rng),
        pd.DataFrame({"NKY": nky}, index=rng),
    )


def load_or_demo():
    if use_demo:
        return make_demo()
    if spx_file is None or nky_file is None:
        return None, None
    spx = load_weekly_xlsx(spx_file.getvalue(), "SPX", cfg)
    nky = load_weekly_xlsx(nky_file.getvalue(), "NKY", cfg)
    return spx, nky


spx_df, nky_df = load_or_demo()

if (spx_df is None or nky_df is None) and not use_demo:
    st.info("⬅️ Upload both files in the sidebar (or turn on **Use demo data**) then click **Run analysis**.")
    st.stop()

# Align dates
common = spx_df.join(nky_df, how="inner").dropna()
if len(common) < 60:
    st.warning("The overlapping sample is quite small. Consider uploading longer histories.")
common_prices = common[["SPX", "NKY"]]

spx_ret = compute_log_returns(common_prices, "SPX")
nky_ret = compute_log_returns(common_prices, "NKY")
rets = pd.concat([spx_ret, nky_ret], axis=1).dropna()

# -----------------------------
# Top: quick overview
# -----------------------------
metric_row([
    ("Observations (prices)", f"{len(common_prices):,}"),
    ("Observations (returns)", f"{len(rets):,}"),
    ("Start", common_prices.index.min().date().isoformat()),
    ("End", common_prices.index.max().date().isoformat()),
])

tabs = st.tabs(["📦 Data", "📐 OLS", "🧠 ARIMA", "🌪️ GARCH", "📤 Export"])


# -----------------------------
# Tab: Data
# -----------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price series")
        plot_series(common_prices, "Weekly Prices")
    with c2:
        st.subheader("Log returns")
        plot_series(rets, "Weekly Log Returns")

    st.subheader("Preview")
    st.dataframe(common_prices.tail(15), width="stretch")


# -----------------------------
# Tab: OLS
# -----------------------------
with tabs[1]:
    st.subheader("OLS: SPX returns explained by Nikkei returns")
    st.caption("Model:  r_SPX,t = α + β · r_NKY,t + ε_t")

    if not run:
        st.info("Click **Run analysis** in the sidebar to fit models and populate results.")
        st.stop()

    y = rets["SPX_ret"]
    x = rets["NKY_ret"]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("α (const)", f"{ols.params['const']:.6f}")
    c2.metric("β", f"{ols.params['NKY_ret']:.4f}")
    c3.metric("R²", f"{ols.rsquared:.3f}")
    c4.metric("Adj. R²", f"{ols.rsquared_adj:.3f}")

    # Simple significance interpretation
    p_beta = float(ols.pvalues.get("NKY_ret", np.nan))
    beta = float(ols.params.get("NKY_ret", np.nan))
    sig = (p_beta < 0.05) if np.isfinite(p_beta) else False
    if np.isfinite(p_beta):
        decision_badge(
            sig,
            f"β is statistically significant at 5% (p = {p_beta:.4f}). Evidence of spillover in this sample.",
            f"β is not statistically significant at 5% (p = {p_beta:.4f}). No strong spillover evidence in this sample.",
        )
    if np.isfinite(beta):
        st.write(f"Estimated direction: **{'positive' if beta > 0 else 'negative' if beta < 0 else 'neutral'}** relationship.")

    st.markdown("#### Diagnostics")
    resid = ols.resid
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rets.index, y=resid, mode="lines", name="Residuals"))
    fig.update_layout(title="OLS residuals over time", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=resid, name="Residuals"))
    fig2.update_layout(title="Residual distribution", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Full regression summary"):
        st.text(ols.summary().as_text())


# -----------------------------
# Tab: ARIMA
# -----------------------------
with tabs[2]:
    st.subheader("ARIMA on SPX returns")
    st.caption("Fits ARIMA(p,d,q) using SARIMAX. You can tune (p, d, q) in the sidebar.")

    if not run:
        st.info("Click **Run analysis** in the sidebar to fit models and populate results.")
        st.stop()

    series = rets["SPX_ret"]
    order = (int(arima_p), int(arima_d), int(arima_q))

    # Train / test split (last h points as test)
    if len(series) <= h + 20:
        st.warning("Not enough observations for the chosen horizon. Reduce horizon or upload more data.")
        st.stop()

    train = series.iloc[:-h]
    test = series.iloc[-h:]

    model = SARIMAX(train, order=order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    fcst = res.get_forecast(steps=h)
    pred = fcst.predicted_mean
    pred.index = test.index

    # Baseline benchmark: constant mean forecast (train mean)
    bench_level = float(train.mean())
    bench_pred = pd.Series(np.repeat(bench_level, len(test)), index=test.index, name="Benchmark")

    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mae = float(mean_absolute_error(test, pred))
    rmse_b = float(np.sqrt(mean_squared_error(test, bench_pred)))
    mae_b = float(mean_absolute_error(test, bench_pred))

    metric_row([
        ("ARIMA order", str(order)),
        ("RMSE (ARIMA)", f"{rmse:.6f}"),
        ("RMSE (Benchmark)", f"{rmse_b:.6f}"),
        ("AIC", f"{res.aic:.2f}"),
    ])

    # Interpretation: did ARIMA beat baseline?
    decision_badge(
        rmse < rmse_b,
        "ARIMA beats the simple mean-return benchmark (lower RMSE).",
        "ARIMA does not beat the simple mean-return benchmark (RMSE not improved).",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode="lines", name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode="lines", name="Test"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred, mode="lines", name="ARIMA Forecast"))
    fig.add_trace(go.Scatter(x=bench_pred.index, y=bench_pred, mode="lines", name="Benchmark (train mean)", line=dict(dash="dot")))
    fig.update_layout(title="ARIMA forecast (returns) vs benchmark", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Model summary"):
        st.text(res.summary().as_text())


# -----------------------------
# Tab: GARCH
# -----------------------------
with tabs[3]:
    st.subheader("GARCH(1,1) on Nikkei returns")
    st.caption("Fits a standard GARCH(1,1) with Student-t innovations and plots conditional volatility.")

    if not run:
        st.info("Click **Run analysis** in the sidebar to fit models and populate results.")
        st.stop()

    r = rets["NKY_ret"] * 100.0  # scale for stability
    gmod = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant")
    gres = gmod.fit(disp="off")

    cond_vol = gres.conditional_volatility / 100.0

    omega = float(gres.params.get("omega", np.nan))
    a1 = float(gres.params.get("alpha[1]", np.nan))
    b1 = float(gres.params.get("beta[1]", np.nan))
    nu = float(gres.params.get("nu", np.nan))

    metric_row([
        ("ω", f"{omega:.6f}" if np.isfinite(omega) else "NA"),
        ("α1", f"{a1:.4f}" if np.isfinite(a1) else "NA"),
        ("β1", f"{b1:.4f}" if np.isfinite(b1) else "NA"),
        ("ν (df)", f"{nu:.2f}" if np.isfinite(nu) else "NA"),
    ])

    # Volatility persistence interpretation
    if np.isfinite(a1) and np.isfinite(b1):
        persistence = a1 + b1
        st.write(f"Volatility persistence (α1 + β1): **{persistence:.3f}**")
        decision_badge(
            persistence < 0.98,
            "Volatility appears mean-reverting (persistence not extremely close to 1).",
            "Volatility is highly persistent (α1 + β1 close to 1).",
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, mode="lines", name="Conditional volatility"))
    fig.update_layout(title="Estimated conditional volatility (Nikkei)", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    # 1-step variance forecast vs realized proxy (r_t^2)
    var_fcst = gres.forecast(horizon=1, reindex=False).variance
    vf = pd.Series(var_fcst.values.flatten(), index=r.index[-len(var_fcst):], name="garch_var_fcst") / (100.0**2)
    realized = (rets["NKY_ret"] ** 2).loc[vf.index]
    bench = pd.Series(np.repeat(realized.mean(), len(realized)), index=realized.index, name="bench_var_fcst")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=realized.index, y=realized, mode="lines", name="Realized var proxy (r²)", opacity=0.4))
    fig2.add_trace(go.Scatter(x=vf.index, y=vf, mode="lines", name="GARCH var forecast"))
    fig2.add_trace(go.Scatter(x=bench.index, y=bench, mode="lines", name="Constant var benchmark", line=dict(dash="dot")))
    fig2.update_layout(title="1-step-ahead variance forecast vs realized variance proxy", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Model summary"):
        st.text(str(gres.summary()))


# -----------------------------
# Tab: Export
# -----------------------------
with tabs[4]:
    st.subheader("Export cleaned data")
    st.caption("Download the aligned price and return series used in the models.")
    out = pd.concat([common_prices, rets], axis=1).dropna()
    csv = out.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="econometrics_cleaned_data.csv", mime="text/csv")

    st.markdown("#### Run command")
    st.code("streamlit run app.py", language="bash")

    st.markdown("#### Requirements")
    st.code(
        "pip install streamlit pandas numpy statsmodels scikit-learn arch plotly openpyxl",
        language="bash",
    )
