import streamlit as st

st.set_page_config(page_title="Smart Fusion 5 Baccarat Predictor", layout="centered")

st.title("Smart Fusion 5 - Live Baccarat Predictor")

Initialize session state for storing results

if 'history' not in st.session_state: st.session_state.history = []

Show current history

st.subheader("📊 Current History") if st.session_state.history: st.write(" ".join(st.session_state.history)) else: st.info("No outcomes entered yet.")

Manual input buttons

col1, col2, col3 = st.columns(3) with col1: if st.button("➕ Player (P)"): st.session_state.history.append("P") with col2: if st.button("➕ Banker (B)"): st.session_state.history.append("B") with col3: if st.button("🗑️ Undo"): if st.session_state.history: st.session_state.history.pop()

def get_prediction(history): if len(history) < 5: return "Need at least 5 rounds of history."

last_5 = history[-5:]

# Rule 4: Dragon Slayer
if len(set(last_5[-3:])) == 1:
    return "Player" if last_5[-1] == "B" else "Banker"

# Rule 3: Zigzag Breaker
if last_5[-4:] == ["B", "P", "B", "P"] or last_5[-4:] == ["P", "B", "P", "B"]:
    return last_5[-1]

# Rule 2: OTB4L
if last_5[-1] != last_5[-3]:
    return "Player" if last_5[-3] == "B" else "Banker"

# Rule 5: 2-in-a-row Catcher
if last_5[-2] == last_5[-1] and last_5[-3] != last_5[-2]:
    return last_5[-1]

# Rule 1: Default Banker
return "Banker"

Display prediction

st.subheader("🔮 Prediction") if len(st.session_state.history) >= 5: prediction = get_prediction(st.session_state.history) st.success(f"Next recommended bet: {prediction}") else: st.warning("Enter at least 5 outcomes to get a prediction.")

