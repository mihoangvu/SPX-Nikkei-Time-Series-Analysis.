import streamlit as st
st.markdown("### 🧩 Bomb Defusal Scene")

col1, col2, col3 = st.columns([1, 1, 1])

# HERO (Econometrician)
with col1:
    if st.session_state.hero_pos == 0:
        st.image("assets/hero_left.jpg", caption="Econometrician")
    elif st.session_state.hero_pos == 1:
        st.image("assets/hero_mid.jpg", caption="Econometrician")
    else:
        st.image("assets/hero_right.jpg", caption="Econometrician")

# BOMB
with col2:
    if st.session_state.lives > 0:
        st.image("assets/bomb_idle.jpg", caption="Market Bomb 💣")
    else:
        st.image("assets/bomb_explode.jpg", caption="BOOM 💥")

# WITCH
with col3:
    st.image("assets/witch.jpg", caption="Witch of Volatility 🧙‍♀️")
if "hero_pos" not in st.session_state:
    st.session_state.hero_pos = 0  # 0=left, 1=mid, 2=right

if "wires" not in st.session_state:
    st.session_state.wires = {
        "red": True,    # stationarity
        "blue": True,   # OLS
        "green": True   # GARCH
    }
st.markdown("## 💣 Cut the Wires")
st.caption("Each wire corresponds to an econometric concept. Cut the wrong one → BOOM.")

wire_cols = st.columns(3)

# 🔴 RED — Stationarity
with wire_cols[0]:
    if st.session_state.wires["red"]:
        if st.button("🔴 Cut Red Wire (Stationarity)"):
            st.session_state.wires["red"] = False
            st.session_state.hero_pos = 1
            correct_msg("Correct. Returns are used to ensure stationarity.")
            _rerun()

# 🔵 BLUE — OLS
with wire_cols[1]:
    if st.session_state.wires["blue"]:
        if st.button("🔵 Cut Blue Wire (OLS Spillover)"):
            if beta_p < 0.05:
                st.session_state.wires["blue"] = False
                st.session_state.hero_pos = 2
                correct_msg("Correct. OLS shows a statistically significant association.")
                _rerun()
            else:
                big_boom("Wrong OLS interpretation. The witch steals your gold!")
                _rerun()

# 🟢 GREEN — GARCH
with wire_cols[2]:
    if st.session_state.wires["green"]:
        if st.button("🟢 Cut Green Wire (Volatility)"):
            if persist >= 0.95:
                st.session_state.wires["green"] = False
                correct_msg("Correct. Volatility is highly persistent.")
                _rerun()
            else:
                big_boom("Wrong volatility interpretation!")
                _rerun()
if not any(st.session_state.wires.values()):
    st.success("🎉 Bomb fully defused!")
    st.balloons()
    st.markdown("### 🧠 Final Econometric Takeaway")
    st.write("""
    - Returns are modeled to avoid non-stationarity  
    - OLS tests association, not causality  
    - ARIMA predictability is limited  
    - GARCH shows strong volatility persistence  
    """)
    st.stop()
