import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
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
    page_title="💣 Econometrics Bomb Challenge — SPX × Nikkei (5Y Weekly)",
    page_icon="💣",
    layout="wide",
)

# Streamlit rerun compatibility
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================
# Styling (shake + panels)
# =========================
st.markdown(
    """
    <style>
    .boom {
      font-size: 54px;
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
    .shake { animation: shake 0.55s; animation-iteration-count: 1; }
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
# Data loader config
# =========================
@dataclass
class DataConfig:
    sheet_name: str = "Worksheet"
    header_row: int = 6
    date_col: int = 0
    value_col: int = 1


# =========================
# Helpers
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def plot_line(df: pd.DataFrame, title: str, height: int = 320):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def plot_series(series: pd.Series, title: str, height: int = 320):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=series.name))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

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
# Game settings
# =========================
START_LIVES = 3
START_GOLD = 300
HINT_COST = 40
WRONG_COST_GOLD = 60
WRONG_COST_SCORE = 120
RIGHT_GAIN_SCORE = 200

LEVEL_TIME_LIMIT = 90  # seconds per "wire session" (soft timer)


# =========================
# Game state
# =========================
def reset_game():
    st.session_state.lives = START_LIVES
    st.session_state.gold = START_GOLD
    st.session_state.score = 0
    st.session_state.correct = 0
    st.session_state.wrong = 0
    st.session_state.hints_used = 0
    st.session_state.start_time = time.time()
    st.session_state.level_start = time.time()

    # Fake animation state
    st.session_state.hero_pos = 0  # 0 left, 1 mid, 2 right
    st.session_state.wires = {"red": True, "blue": True, "green": True}
    st.session_state.defused = False
    st.session_state.game_over = False

if "lives" not in st.session_state:
    reset_game()

def time_remaining() -> int:
    elapsed = time.time() - st.session_state.level_start
    return int(max(0, LEVEL_TIME_LIMIT - elapsed))

def spend_hint(hint_text: str):
    if st.session_state.gold < HINT_COST:
        st.warning("Not enough gold to buy a hint 💸")
        return
    st.session_state.gold -= HINT_COST
    st.session_state.score -= 40
    st.session_state.hints_used += 1
    st.info(f"🧾 Hint (-{HINT_COST}💰): {hint_text}")

def boom(message: str):
    st.session_state.lives -= 1
    st.session_state.gold -= WRONG_COST_GOLD
    st.session_state.score -= WRONG_COST_SCORE
    st.session_state.wrong += 1

    st.markdown('<div class="shake panel">', unsafe_allow_html=True)
    st.markdown('<div class="boom">💥 BOOOOM 💥</div>', unsafe_allow_html=True)
    st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)

    # reset timer so they can try again if still alive
    st.session_state.level_start = time.time()

    if st.session_state.lives <= 0 or st.session_state.gold <= 0:
        st.session_state.game_over = True

def correct(message: str):
    st.session_state.score += RIGHT_GAIN_SCORE
    st.session_state.correct += 1
    st.success(message)
    st.balloons()

def soft_timer_check():
    # Only triggers on interactions/reruns
    if st.session_state.defused or st.session_state.game_over:
        return
    if time_remaining() <= 0:
        boom("⏰ Time’s up! The Witch triggered the bomb because you hesitated.")


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
st.sidebar.header("🧠 ARIMA settings (for evaluation panel)")
h = int(st.sidebar.slider("Forecast horizon (weeks)", min_value=4, max_value=104, value=52, step=4))
p = int(st.sidebar.slider("AR order p", 0, 5, 1))
d = int(st.sidebar.slider("Differencing d", 0, 2, 0))
q = int(st.sidebar.slider("MA order q", 0, 5, 1))

st.sidebar.divider()
st.sidebar.header("🎮 Control")
if st.sidebar.button("Reset game", type="secondary"):
    reset_game()
    _rerun()

st.sidebar.caption("Assets expected in /assets: hero_left.jpg, hero_mid.jpg, hero_right.jpg, bomb_idle.jpg, bomb_explode.jpg, witch.jpg")


# =========================
# Load data
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
    st.info("⬅️ Upload both files in the sidebar (or turn on demo data) to start.")
    st.stop()

common = spx_df.join(nky_df, how="inner").dropna()
prices = common[["SPX", "NKY"]]
rets = pd.concat(
    [compute_log_returns(prices, "SPX"), compute_log_returns(prices, "NKY")],
    axis=1
).dropna()


# =========================
# Fit econometric models
# =========================
# OLS: SPX_ret ~ const + NKY_ret
y = rets["SPX_ret"]
x = rets["NKY_ret"]
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
beta = float(ols.params.get("NKY_ret", np.nan))
beta_p = float(ols.pvalues.get("NKY_ret", np.nan))
r2 = float(ols.rsquared)

# ARIMA: on SPX returns (evaluation)
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
    arima_rmse = arima_mae = np.nan
    bench_rmse = bench_mae = np.nan

# GARCH: on NKY returns
r = (rets["NKY_ret"] * 100.0).dropna()
gmod = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant")
gres = gmod.fit(disp="off")
omega = float(gres.params.get("omega", np.nan))
alpha1 = float(gres.params.get("alpha[1]", np.nan))
beta1 = float(gres.params.get("beta[1]", np.nan))
persist = alpha1 + beta1
cond_vol = (gres.conditional_volatility / 100.0).rename("Conditional Volatility")


# =========================
# Header
# =========================
st.markdown("# 💣 Econometrics Bomb Challenge — SPX × Nikkei (5Y Weekly)")
st.caption("🧙‍♀️ The Witch of Volatility stole your gold and armed a bomb in the market. Defuse it with econometrics.")

soft_timer_check()

# HUD
hud = st.columns([1.2, 1.2, 1.2, 1.2, 1.8])
hud[0].metric("Lives", f"{st.session_state.lives} ❤️")
hud[1].metric("Gold", f"{st.session_state.gold} 💰")
hud[2].metric("Score", f"{st.session_state.score}")
hud[3].metric("Wires left", f"{sum(1 for v in st.session_state.wires.values() if v)} / 3")
hud[4].metric("Timer", f"{time_remaining()}s ⏳")
st.progress((3 - sum(1 for v in st.session_state.wires.values() if v)) / 3)


# =========================
# Data preview (optional)
# =========================
with st.expander("📊 Data preview (SPX & Nikkei)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        plot_line(prices, "Weekly Prices", height=260)
    with c2:
        plot_line(rets, "Weekly Log Returns", height=260)
    st.dataframe(pd.concat([prices, rets], axis=1).dropna().tail(12), use_container_width=True)


# =========================
# Handle game over / defused
# =========================
def show_scoreboard(title: str):
    st.markdown(f"## {title}")
    total = st.session_state.correct + st.session_state.wrong
    acc = (st.session_state.correct / total) * 100 if total > 0 else 0.0
    elapsed = int(time.time() - st.session_state.start_time)
    st.write(f"- Correct: **{st.session_state.correct}**")
    st.write(f"- Wrong: **{st.session_state.wrong}**")
    st.write(f"- Hints used: **{st.session_state.hints_used}**")
    st.write(f"- Accuracy: **{acc:.1f}%**")
    st.write(f"- Time: **{elapsed}s**")
    st.write(f"- Final Score: **{st.session_state.score}**")

if st.session_state.game_over:
    st.error("💀 Game Over. You ran out of lives or gold.")
    show_scoreboard("🧾 Final Scoreboard")
    if st.button("Try again"):
        reset_game()
        _rerun()
    st.stop()

if st.session_state.defused:
    st.success("✅ Bomb fully defused! You beat the Witch of Volatility.")
    st.balloons()
    show_scoreboard("🧾 Final Scoreboard")

    st.markdown("### 🧠 Key Findings (auto)")
    st.write("- We model **returns** rather than prices to reduce non-stationarity issues.")
    st.write(f"- OLS: β = {beta:.4f}, p = {beta_p:.4g} → " + ("evidence of association/spillover." if beta_p < 0.05 else "no strong evidence at 5%."))
    if arima_ok:
        st.write(f"- ARIMA({p},{d},{q}): RMSE={arima_rmse:.6f} vs benchmark RMSE={bench_rmse:.6f} → " +
                 ("ARIMA improves." if arima_rmse < bench_rmse else "predictability is weak (benchmark similar)."))
    st.write(f"- GARCH(1,1): α+β = {persist:.4f} → " + ("high volatility persistence." if persist >= 0.95 else "moderate persistence."))

    st.markdown("### 📤 Export cleaned dataset")
    out = pd.concat([prices, rets], axis=1).dropna().reset_index().rename(columns={"index": "Date"})
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="spx_nky_cleaned_data.csv", mime="text/csv")

    if st.button("Play again"):
        reset_game()
        _rerun()
    st.stop()


st.divider()

# =========================
# Bomb Defusal Scene (Character + Bomb + Witch)
# =========================
st.markdown("## 🧩 Bomb Defusal Scene")

ASSETS = Path("assets")
hero_left = ASSETS / "hero_left.jpg"
hero_mid = ASSETS / "hero_mid.jpg"
hero_right = ASSETS / "hero_right.jpg"
bomb_idle = ASSETS / "bomb_idle.jpg"
bomb_explode = ASSETS / "bomb_explode.jpg"
witch = ASSETS / "witch.jpg"

missing = [p.name for p in [hero_left, hero_mid, hero_right, bomb_idle, bomb_explode, witch] if not p.exists()]
if missing:
    st.warning(f"Missing asset files in /assets: {', '.join(missing)}. (Images won't show until uploaded.)")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.session_state.hero_pos == 0:
        if hero_left.exists(): st.image(str(hero_left), caption="Econometrician")
    elif st.session_state.hero_pos == 1:
        if hero_mid.exists(): st.image(str(hero_mid), caption="Econometrician")
    else:
        if hero_right.exists(): st.image(str(hero_right), caption="Econometrician")

with col2:
    # If you want: show explode only after wrong choice? Here we show idle unless game_over
    if bomb_idle.exists():
        st.image(str(bomb_idle), caption="Market Bomb 💣")

with col3:
    if witch.exists(): st.image(str(witch), caption="Witch of Volatility 🧙‍♀️")


st.divider()

# =========================
# Wire cards
# =========================
st.markdown("## 💣 Cut the Wires")
st.caption("Each wire corresponds to an econometric concept. Cut the wrong wire → BOOM (lose ❤️, 💰, and score).")

wire_cols = st.columns(3)

# -------------------------
# RED wire: Stationarity / returns
# -------------------------
with wire_cols[0]:
    st.markdown("### 🔴 Red Wire")
    st.write("**Concept:** Stationarity / returns vs prices")

    if st.button("Buy hint (-40💰)", key="hint_red"):
        spend_hint("Financial prices are often non-stationary; returns are commonly used to satisfy stationarity assumptions.")

    with st.expander("Why this matters (Econometrics)"):
        st.write(
            "Price levels often behave like random walks (non-stationary). Modeling returns helps avoid spurious regression "
            "and supports valid inference in OLS/ARIMA/GARCH."
        )

    if st.session_state.wires["red"]:
        if st.button("Cut 🔴", key="cut_red", use_container_width=True):
            st.session_state.wires["red"] = False
            st.session_state.hero_pos = 1
            correct("✅ Correct. We model returns to reduce non-stationarity issues.")
            _rerun()
    else:
        st.success("Red wire cut ✅")

# -------------------------
# BLUE wire: OLS spillover inference
# -------------------------
with wire_cols[1]:
    st.markdown("### 🔵 Blue Wire")
    st.write("**Concept:** OLS spillover interpretation")

    st.metric("β (NKY_ret)", f"{beta:.4f}")
    st.metric("p-value(β)", f"{beta_p:.4g}")
    st.metric("R²", f"{r2:.3f}")

    if st.button("Buy hint (-40💰)", key="hint_blue"):
        spend_hint("Check p-value for β. If p < 0.05, reject H0: β=0 → evidence of association (not causality).")

    with st.expander("Why this matters (Econometrics)"):
        st.write(
            "OLS tests whether the slope coefficient differs from zero. A small p-value suggests evidence of association "
            "between markets, but OLS does not prove causality."
        )

    if st.session_state.wires["blue"]:
        if st.button("Cut 🔵", key="cut_blue", use_container_width=True):
            if np.isfinite(beta_p) and beta_p < 0.05:
                st.session_state.wires["blue"] = False
                st.session_state.hero_pos = 2
                correct("✅ Correct. β is significant → evidence of association/spillover (not causality).")
                _rerun()
            else:
                # wrong cut: interpreted as 'significant' when it's not
                boom("Wrong OLS interpretation: β is not significant at 5% in this sample. The witch steals your gold!")
                # show explode image immediately as extra drama
                if bomb_explode.exists():
                    st.image(str(bomb_explode), caption="💥 BOOM 💥")
                _rerun()
    else:
        st.success("Blue wire cut ✅")

# -------------------------
# GREEN wire: GARCH persistence
# -------------------------
with wire_cols[2]:
    st.markdown("### 🟢 Green Wire")
    st.write("**Concept:** GARCH volatility persistence")

    st.metric("α1", f"{alpha1:.4f}")
    st.metric("β1", f"{beta1:.4f}")
    st.metric("α+β", f"{persist:.4f}")

    if st.button("Buy hint (-40💰)", key="hint_green"):
        spend_hint("In GARCH(1,1), α+β measures persistence. Values near 1 → very persistent volatility (slow decay).")

    with st.expander("Why this matters (Econometrics)"):
        st.write(
            "In GARCH(1,1), α+β close to 1 implies long memory in conditional variance: volatility shocks decay slowly. "
            "With weekly data, volatility can look smooth/flat due to high persistence."
        )

    if st.session_state.wires["green"]:
        if st.button("Cut 🟢", key="cut_green", use_container_width=True):
            if np.isfinite(persist) and persist >= 0.95:
                st.session_state.wires["green"] = False
                correct("✅ Correct. α+β close to 1 → volatility is highly persistent.")
                _rerun()
            else:
                boom("Wrong volatility interpretation: α+β is not close enough to 1 for 'high persistence' here.")
                if bomb_explode.exists():
                    st.image(str(bomb_explode), caption="💥 BOOM 💥")
                _rerun()
    else:
        st.success("Green wire cut ✅")


# =========================
# Win condition
# =========================
if not any(st.session_state.wires.values()):
    st.session_state.defused = True
    _rerun()


# =========================
# Optional model evaluation panel (ARIMA + GARCH plots)
# =========================
with st.expander("📈 Model Evaluation Panel (optional)", expanded=False):
    st.markdown("### ARIMA Forecast (SPX returns)")
    if not arima_ok:
        st.warning("Not enough observations for selected ARIMA horizon. Reduce horizon or upload longer series.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ARIMA order", f"({p},{d},{q})")
        m2.metric("RMSE (ARIMA)", f"{arima_rmse:.6f}")
        m3.metric("RMSE (Benchmark)", f"{bench_rmse:.6f}")
        m4.metric("Winner", "ARIMA" if arima_rmse < bench_rmse else "Benchmark")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, mode="lines", name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, mode="lines", name="Test"))
        fig.add_trace(go.Scatter(x=fcst.index, y=fcst, mode="lines", name="ARIMA forecast"))
        fig.add_trace(go.Scatter(
            x=test.index, y=np.repeat(train.mean(), len(test)),
            mode="lines", name="Benchmark (train mean)", line=dict(dash="dot")
        ))
        fig.update_layout(title="Out-of-sample forecast (SPX returns)", height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### GARCH Conditional Volatility (NKY)")
    plot_series(cond_vol, "Conditional Volatility (NKY)", height=340)

st.caption("🎓 OLS for spillover testing • ARIMA for forecast evaluation • GARCH for volatility dynamics — delivered as a game.")
