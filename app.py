import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="💣 Econometrics Bomb Challenge — SPX × Nikkei",
    page_icon="💣",
    layout="wide",
)

# =========================
# Data loader config
# =========================
@dataclass
class DataConfig:
    sheet_name: str = "Worksheet"
    header_row: int = 6
    date_col: int = 0
    value_col: int = 1


# =========================
# Styling (flash/shake for explosion)
# =========================
st.markdown(
    """
    <style>
    .boom {
      font-size: 48px;
      font-weight: 900;
      letter-spacing: 1px;
    }
    @keyframes shake {
      0% { transform: translate(1px, 1px) rotate(0deg); }
      10% { transform: translate(-2px, -1px) rotate(-1deg); }
      20% { transform: translate(-3px, 0px) rotate(1deg); }
      30% { transform: translate(3px, 2px) rotate(0deg); }
      40% { transform: translate(1px, -1px) rotate(1deg); }
      50% { transform: translate(-1px, 2px) rotate(-1deg); }
      60% { transform: translate(-3px, 1px) rotate(0deg); }
      70% { transform: translate(3px, 1px) rotate(-1deg); }
      80% { transform: translate(-1px, -1px) rotate(1deg); }
      90% { transform: translate(1px, 2px) rotate(0deg); }
      100% { transform: translate(1px, -2px) rotate(-1deg); }
    }
    .shake {
      animation: shake 0.55s;
      animation-iteration-count: 1;
    }
    .panel {
      border-radius: 14px;
      padding: 14px 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Helpers
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred)))

def plot_line(df: pd.DataFrame, title: str, height: int = 360):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

def plot_series(series: pd.Series, title: str, height: int = 360):
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


# =========================
# Game state
# =========================
TOTAL_LEVELS = 5

LEVEL_TIME_LIMITS = {
    1: 45,  # seconds
    2: 60,
    3: 75,
    4: 60,
    5: 45,
}

START_GOLD = 300
HINT_COST_GOLD = 40
WRONG_COST_GOLD = 60

def reset_game():
    st.session_state.level = 1
    st.session_state.lives = 3
    st.session_state.gold = START_GOLD
    st.session_state.score = 0
    st.session_state.correct = 0
    st.session_state.wrong = 0
    st.session_state.hints_used = 0
    st.session_state.start_time = time.time()
    st.session_state.level_start = time.time()
    st.session_state.game_over = False
    st.session_state.defused = False
    st.session_state.last_boom = False
    st.session_state.story = "🧙‍♀️ The Witch of Volatility stole your gold and armed a bomb in the market. Answer correctly to defuse it."

if "level" not in st.session_state:
    reset_game()

def start_level_timer(level: int):
    st.session_state.level_start = time.time()

def time_remaining(level: int) -> int:
    limit = LEVEL_TIME_LIMITS.get(level, 60)
    elapsed = time.time() - st.session_state.level_start
    return int(max(0, limit - elapsed))

def header_ui():
    st.markdown("## 💣 Econometrics Bomb Challenge — SPX × Nikkei (5Y Weekly)")
    st.caption(st.session_state.story)

    cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.8])
    cols[0].metric("Level", f"{st.session_state.level}/{TOTAL_LEVELS}")
    cols[1].metric("Lives", f"{st.session_state.lives} ❤️")
    cols[2].metric("Gold", f"{st.session_state.gold} 💰")
    cols[3].metric("Score", f"{st.session_state.score}")
    rem = time_remaining(st.session_state.level) if not st.session_state.defused else 0
    cols[4].metric("Timer", f"{rem}s ⏳" if not st.session_state.defused else "—")

    st.progress(min(st.session_state.level, TOTAL_LEVELS) / TOTAL_LEVELS)

def big_boom(message: str):
    st.session_state.last_boom = True
    st.session_state.lives -= 1
    st.session_state.gold -= WRONG_COST_GOLD
    st.session_state.wrong += 1
    st.session_state.score -= 120

    st.markdown('<div class="shake panel">', unsafe_allow_html=True)
    st.markdown('<div class="boom">💥 BOOOOM 💥</div>', unsafe_allow_html=True)
    st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)

    # A bit of drama
    try:
        st.snow()
    except Exception:
        pass

    if st.session_state.gold <= 0 or st.session_state.lives <= 0:
        st.session_state.game_over = True

def correct_msg(message: str):
    st.session_state.last_boom = False
    st.session_state.correct += 1
    st.session_state.score += 200
    st.success(message)
    st.balloons()

def use_hint(hint_text: str):
    if st.session_state.gold < HINT_COST_GOLD:
        st.warning("Not enough gold to buy a hint 💸")
        return
    st.session_state.gold -= HINT_COST_GOLD
    st.session_state.hints_used += 1
    st.session_state.score -= 40
    st.info(f"🧾 Hint (cost {HINT_COST_GOLD} gold): {hint_text}")

def goto_next_level():
    st.session_state.level += 1
    if st.session_state.level <= TOTAL_LEVELS:
        start_level_timer(st.session_state.level)
    else:
        st.session_state.defused = True

def timer_check_or_fail():
    # Soft timer: checks on every rerun (interactions). If time ran out, you lose.
    if st.session_state.defused or st.session_state.game_over:
        return
    rem = time_remaining(st.session_state.level)
    if rem <= 0:
        big_boom("⏰ Time’s up! The witch triggered the bomb because you hesitated.")
        # restart level timer so user can try again if still alive
        start_level_timer(st.session_state.level)


def shuffled_options(options):
    # options = list[(label, is_correct, explainer_if_correct, explainer_if_wrong)]
    opts = options[:]
    np.random.shuffle(opts)
    return opts


# =========================
# Sidebar
# =========================
st.sidebar.header("📦 Data")
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
st.sidebar.header("🧠 ARIMA Settings (used in Level 3)")
h = int(st.sidebar.slider("Forecast horizon (weeks)", min_value=4, max_value=104, value=52, step=4))
p = int(st.sidebar.slider("AR order p", 0, 5, 1))
d = int(st.sidebar.slider("Differencing d", 0, 2, 0))
q = int(st.sidebar.slider("MA order q", 0, 5, 1))
st.sidebar.caption("Tip for demo: ARIMA(1,0,1), horizon 52 weeks.")

st.sidebar.divider()
st.sidebar.header("🎮 Control")
if st.sidebar.button("Reset game", type="secondary"):
    reset_game()

st.sidebar.markdown("---")
st.sidebar.subheader("🐉 Lore")
st.sidebar.write("A witch (and sometimes a dragon) steals your gold when you make bad econometric calls.")


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

# =========================
# Precompute models (used by levels)
# =========================
# OLS: SPX_ret ~ NKY_ret
y = rets["SPX_ret"]
x = rets["NKY_ret"]
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
beta = float(ols.params.get("NKY_ret", np.nan))
beta_p = float(ols.pvalues.get("NKY_ret", np.nan))
r2 = float(ols.rsquared)

# ARIMA on SPX returns
series = rets["SPX_ret"]
arima_ok = len(series) > h + 20
if arima_ok:
    train = series.iloc[:-h]
    test = series.iloc[-h:]
    arima_model = SARIMAX(
        train,
        order=(p, d, q),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    arima_res = arima_model.fit(disp=False)
    fcst = arima_res.get_forecast(steps=h).predicted_mean
    fcst.index = test.index
    arima_rmse = rmse(test, fcst)
    arima_mae = mae(test, fcst)

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
header_ui()
timer_check_or_fail()

# Data preview panel
with st.expander("📊 Data preview (SPX & Nikkei)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        plot_line(prices, "Weekly Prices", height=280)
    with c2:
        plot_line(rets, "Weekly Log Returns", height=280)
    st.dataframe(pd.concat([prices, rets], axis=1).dropna().tail(12), width="stretch")

st.divider()

# If game over
if st.session_state.game_over:
    st.error("💀 Game Over. You ran out of lives or gold.")
    st.markdown("### 🧾 Final Scoreboard")
    total = st.session_state.correct + st.session_state.wrong
    acc = (st.session_state.correct / total) * 100 if total > 0 else 0.0
    elapsed = int(time.time() - st.session_state.start_time)
    st.write(f"- Correct: **{st.session_state.correct}**")
    st.write(f"- Wrong: **{st.session_state.wrong}**")
    st.write(f"- Hints used: **{st.session_state.hints_used}**")
    st.write(f"- Accuracy: **{acc:.1f}%**")
    st.write(f"- Time: **{elapsed}s**")
    st.write(f"- Score: **{st.session_state.score}**")
    if st.button("Try again"):
        reset_game()
        st.experimental_rerun()
    st.stop()

# If defused
if st.session_state.defused:
    st.success("✅ Bomb Defused! You beat the Witch of Volatility.")
    st.balloons()
    st.markdown("### 🧾 Scoreboard")
    total = st.session_state.correct + st.session_state.wrong
    acc = (st.session_state.correct / total) * 100 if total > 0 else 0.0
    elapsed = int(time.time() - st.session_state.start_time)
    st.write(f"- Correct: **{st.session_state.correct}**")
    st.write(f"- Wrong: **{st.session_state.wrong}**")
    st.write(f"- Hints used: **{st.session_state.hints_used}**")
    st.write(f"- Accuracy: **{acc:.1f}%**")
    st.write(f"- Time: **{elapsed}s**")
    st.write(f"- Final Score: **{st.session_state.score}**")

    st.markdown("### 🧠 Key Findings (auto)")
    st.write("- We model **returns** rather than prices to avoid non-stationarity issues.")
    st.write(f"- OLS: β = {beta:.4f}, p = {beta_p:.4g} → " + ("evidence of relationship/spillover." if beta_p < 0.05 else "no strong evidence at 5%."))
    if arima_ok:
        st.write(f"- ARIMA({p},{d},{q}) forecast: RMSE={arima_rmse:.6f} vs benchmark RMSE={bench_rmse:.6f} → " +
                 ("ARIMA improves slightly." if arima_rmse < bench_rmse else "predictability is weak (benchmark similar)."))
    st.write(f"- GARCH: α+β = {persist:.4f} → " + ("high volatility persistence." if persist >= 0.95 else "moderate persistence."))

    st.markdown("### 📤 Export cleaned dataset")
    out = pd.concat([prices, rets], axis=1).dropna().reset_index().rename(columns={"index": "Date"})
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="spx_nky_cleaned_data.csv", mime="text/csv")

    if st.button("Play again"):
        reset_game()
        st.experimental_rerun()
    st.stop()


# =========================
# LEVELS
# =========================
level = st.session_state.level

# (Re)start timer if entering level first time in session
if "level_initialized" not in st.session_state:
    st.session_state.level_initialized = level
    start_level_timer(level)
elif st.session_state.level_initialized != level:
    st.session_state.level_initialized = level
    start_level_timer(level)

# Helper to render Q with shuffled buttons + optional hint
def render_question(title: str, question: str, options, hint_text: str | None = None, explain_box: str | None = None):
    st.subheader(title)
    st.write(question)

    # Hint button
    if hint_text:
        hcol1, hcol2 = st.columns([1, 4])
        with hcol1:
            if st.button(f"Buy Hint (-{HINT_COST_GOLD}💰)", type="secondary"):
                use_hint(hint_text)
        with hcol2:
            st.caption("Hints cost gold and reduce score — use strategically.")

    opts = shuffled_options(options)
    c1, c2 = st.columns(2)
    cols = [c1, c2]
    for i, (label, is_correct, ok_msg, bad_msg) in enumerate(opts):
        with cols[i % 2]:
            if st.button(label, use_container_width=True):
                if is_correct:
                    correct_msg(ok_msg)
                    goto_next_level()
                    st.experimental_rerun()
                else:
                    big_boom(bad_msg)
                    st.experimental_rerun()

    if explain_box:
        with st.expander("Why this matters (Econometrics)", expanded=False):
            st.write(explain_box)

    st.caption("⏳ Timer is enforced when you interact (soft timer). If you wait too long, the witch triggers the bomb.")


# LEVEL 1 — Stationarity decision
if level == 1:
    st.session_state.story = "🧙‍♀️ The Witch of Volatility: 'Choose wisely… stationarity decides your fate.'"
    render_question(
        "💣 Level 1 — Stationarity Decision",
        "You see price levels and returns. What should we model for standard financial econometrics?",
        options=[
            ("Model PRICES", False,
             "Nice (but… actually this would be wrong).",  # not used
             "Prices are often non-stationary → risk of spurious regression."),
            ("Model RETURNS", True,
             "Correct. Returns are typically closer to stationary and suitable for inference.",
             "Not quite."),
        ],
        hint_text="Financial price levels are often non-stationary. Returns are commonly used to satisfy stationarity assumptions.",
        explain_box="In many financial applications, price levels behave like random walks (non-stationary). Using returns helps stabilize the mean/variance and supports valid inference in OLS/ARIMA/GARCH."
    )

# LEVEL 2 — OLS spillover
elif level == 2:
    st.session_state.story = "🐉 A Dragon Banker appears: 'Prove spillover… or pay in gold.'"
    m1, m2, m3 = st.columns(3)
    m1.metric("β (NKY_ret)", f"{beta:.4f}")
    m2.metric("p-value(β)", f"{beta_p:.4g}")
    m3.metric("R²", f"{r2:.3f}")

    with st.expander("Show OLS diagnostics", expanded=False):
        resid = ols.resid
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rets.index, y=resid, mode="lines", name="Residuals"))
        fig.update_layout(title="OLS residuals over time", height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    # Determine correct answer based on p-value
    is_spillover = bool(np.isfinite(beta_p) and beta_p < 0.05)

    render_question(
        "💣 Level 2 — OLS Spillover Test",
        "Question: Based on the regression, is there evidence that Nikkei returns help explain SPX returns?",
        options=[
            ("YES — β is statistically significant", is_spillover,
             "Correct. We reject H0: β = 0 at 5% → evidence of relationship/spillover.",
             "Not supported. In this sample, β is not significant at 5%."),
            ("NO — no strong evidence at 5%", (not is_spillover),
             "Correct. We fail to reject H0: β = 0 → no strong evidence at 5%.",
             "Actually β is significant at 5%, so 'NO' is not supported by the regression."),
        ],
        hint_text="Check the p-value for β. If p < 0.05, reject H0: β = 0.",
        explain_box="OLS tests whether the slope coefficient is statistically different from zero. A small p-value provides evidence against the null hypothesis of no relationship."
    )

# LEVEL 3 — ARIMA forecast & benchmark
elif level == 3:
    st.session_state.story = "🧙‍♀️ Witch whispers: 'Can you truly forecast returns… or is it illusion?'"

    if not arima_ok:
        st.warning("Not enough observations for chosen horizon. Reduce horizon or upload more data.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ARIMA order", f"({p},{d},{q})")
    m2.metric("RMSE (ARIMA)", f"{arima_rmse:.6f}")
    m3.metric("RMSE (Benchmark)", f"{bench_rmse:.6f}")
    m4.metric("Winner", "ARIMA" if arima_rmse < bench_rmse else "Benchmark")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode="lines", name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode="lines", name="Test"))
    fig.add_trace(go.Scatter(x=fcst.index, y=fcst, mode="lines", name="ARIMA forecast"))
    fig.add_trace(go.Scatter(x=test.index, y=np.repeat(train.mean(), len(test)),
                             mode="lines", name="Benchmark (train mean)", line=dict(dash="dot")))
    fig.update_layout(title="Out-of-sample forecast on SPX returns", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    # Define "strong predictability" only if ARIMA beats benchmark meaningfully
    strong = bool(arima_rmse < 0.9 * bench_rmse)

    render_question(
        "💣 Level 3 — ARIMA Reality Check",
        "Question: Based on ARIMA vs benchmark performance, is return predictability strong here?",
        options=[
            ("YES — predictability looks strong", strong,
             "Correct. ARIMA meaningfully improves over benchmark → evidence of some predictability (in this setup).",
             "Not supported. ARIMA does not clearly outperform benchmark → predictability looks weak."),
            ("NO — predictability is weak", (not strong),
             "Correct. Financial returns are often hard to forecast; benchmark performs similarly.",
             "But ARIMA clearly beat the benchmark here, so predictability might be non-trivial."),
        ],
        hint_text="Compare ARIMA RMSE to benchmark RMSE. If they’re similar, predictability is weak.",
        explain_box="Forecasting is used for model evaluation. For many financial returns, mean predictability is limited, so simple benchmarks can be competitive."
    )

# LEVEL 4 — GARCH persistence
elif level == 4:
    st.session_state.story = "🐉 Dragon roars: 'Explain volatility… or the gold is mine!'"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ω", f"{omega:.6f}")
    m2.metric("α1", f"{alpha1:.4f}")
    m3.metric("β1", f"{beta1:.4f}")
    m4.metric("α+β", f"{persist:.4f}")

    plot_series(cond_vol, "Conditional Volatility (NKY)", height=320)

    high_persist = bool(persist >= 0.95)

    render_question(
        "💣 Level 4 — GARCH Volatility Persistence",
        "Question: What does α + β close to 1 imply?",
        options=[
            ("Volatility is highly persistent", high_persist,
             "Correct. α+β near 1 indicates very persistent volatility (IGARCH-like behavior).",
             "Not quite. α+β here indicates persistence, not randomness."),
            ("Volatility is basically random", (not high_persist),
             "Correct. α+β is not close to 1, so persistence is not extremely high.",
             "Incorrect. With α+β near 1, volatility persistence is high."),
        ],
        hint_text="In GARCH(1,1), α+β measures persistence. Values near 1 imply shocks decay very slowly.",
        explain_box="A near-unit persistence (α+β≈1) implies long memory in volatility: conditional variance responds to shocks and reverts very slowly. With weekly data, volatility can look smooth/flat when persistence is high."
    )

# LEVEL 5 — Final boss (integrated conclusion)
elif level == 5:
    st.session_state.story = "🧙‍♀️ FINAL BOSS: 'Choose the correct takeaway… or BOOM.'"

    # Derive integrated statements
    spill = (beta_p < 0.05) if np.isfinite(beta_p) else False
    weak_pred = (not (arima_ok and (arima_rmse < 0.9 * bench_rmse)))
    vol_persist = (persist >= 0.95)

    correct_takeaway = (
        "Returns are hard to predict, but volatility is persistent; OLS tests association (not causality)."
    )

    render_question(
        "💣 Level 5 — Final Boss: Choose the takeaway",
        "Pick the most econometrically correct summary of our SPX–Nikkei results:",
        options=[
            ("OLS proves Nikkei causes SPX, and ARIMA can predict markets reliably.", False,
             "—", "OLS does not prove causality, and return predictability is usually limited."),
            (correct_takeaway, True,
             "Correct. This is a cautious, econometrically valid interpretation.",
             "—"),
            ("Because ARIMA forecasts the future, this app is a trading system.", False,
             "—", "This project focuses on model evaluation and interpretation, not trading claims."),
        ],
        hint_text="Avoid overclaiming: OLS ≠ causality, ARIMA forecasts are for evaluation, and volatility persistence is common.",
        explain_box="A strong project takeaway is cautious and econometrically valid: OLS indicates association, ARIMA often shows weak predictability in returns, and GARCH highlights volatility dynamics."
    )

st.divider()
st.caption("🎓 Built for Econometrics: hypothesis testing (OLS), forecast evaluation (ARIMA), and volatility dynamics (GARCH) — now as a game.")
