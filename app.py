import streamlit as st
import random
import os
import time
from datetime import datetime, timedelta

# Constants
BASE_UNIT = 5  # % of bankroll for initial bet
DATA_FILE = "baccarat_data.txt"
SESSION_TRACK_FILE = "user_sessions.txt"

# Initialize session state
if "sequence" not in st.session_state:
    st.session_state.sequence = []

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 1000.0

if "unit" not in st.session_state:
    st.session_state.unit = round(BASE_UNIT / 100 * st.session_state.bankroll, 2)

if "current_bet" not in st.session_state:
    st.session_state.current_bet = st.session_state.unit

if "wins" not in st.session_state:
    st.session_state.wins = 0

if "losses" not in st.session_state:
    st.session_state.losses = 0

# --- Function Definitions ---

def save_to_file(result):
    with open(DATA_FILE, "a") as f:
        f.write(result + "\n")

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

def predict_next(sequence):
    if len(sequence) < 3:
        return None, 0.0

    recent_sequence = sequence[-6:]
    patterns = {"P": 0, "B": 0}

    for i in range(len(recent_sequence) - 2):
        pattern = tuple(recent_sequence[i:i+2])
        if i < len(recent_sequence) - 3:
            next_outcome = recent_sequence[i+2]
            if pattern == tuple(recent_sequence[-2:]):
                patterns[next_outcome] += 1

    total = sum(patterns.values())
    if total == 0:
        return None, 0.0

    predicted = max(patterns, key=patterns.get)
    confidence = (patterns[predicted] / total) * 100
    return predicted, confidence

def update_bankroll(result, bet):
    if result == st.session_state.last_prediction:
        st.session_state.bankroll += bet
        st.session_state.wins += 1
        st.session_state.current_bet = st.session_state.unit
    else:
        st.session_state.bankroll -= bet
        st.session_state.losses += 1
        st.session_state.current_bet += 1

def reset_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.session_state.sequence = []
    st.session_state.bankroll = 1000.0
    st.session_state.unit = round(BASE_UNIT / 100 * st.session_state.bankroll, 2)
    st.session_state.current_bet = st.session_state.unit
    st.session_state.wins = 0
    st.session_state.losses = 0

def track_user_session_file():
    ip = st.session_state.get("ip_address", f"user_{random.randint(1000, 9999)}")
    current_time = datetime.now()

    if not os.path.exists(SESSION_TRACK_FILE):
        with open(SESSION_TRACK_FILE, "w") as f:
            f.write(f"{ip},{current_time}\n")
    else:
        lines = []
        updated = False
        with open(SESSION_TRACK_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2 and parts[0] == ip:
                    last_seen = datetime.fromisoformat(parts[1])
                    if current_time - last_seen <= timedelta(seconds=30):
                        updated = True
                        lines.append(f"{ip},{current_time}\n")
                    else:
                        lines.append(line)
                else:
                    lines.append(line)
        if not updated:
            lines.append(f"{ip},{current_time}\n")
        with open(SESSION_TRACK_FILE, "w") as f:
            f.writelines(lines)

def count_active_users():
    if not os.path.exists(SESSION_TRACK_FILE):
        return 0
    now = datetime.now()
    with open(SESSION_TRACK_FILE, "r") as f:
        count = 0
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    last_seen = datetime.fromisoformat(parts[1])
                    if now - last_seen <= timedelta(seconds=30):
                        count += 1
                except ValueError:
                    continue
        return count

# --- App Layout ---

st.title("Baccarat Predictor - Algo Z100")

# Track user sessions
track_user_session_file()
active_users = count_active_users()
st.sidebar.markdown(f"**Active Users:** {active_users}")

# --- Game Interaction ---
with st.form("baccarat_input"):
    st.subheader("Enter Baccarat Result")
    result = st.radio("Select Result", ["P", "B", "T"], horizontal=True)
    submitted = st.form_submit_button("Submit Result")

    if submitted:
        if result in ["P", "B"]:
            st.session_state.sequence.append(result)
            save_to_file(result)

            pred, conf = predict_next(st.session_state.sequence)
            st.session_state.last_prediction = pred

            if pred:
                st.success(f"Prediction: {pred} (Confidence: {conf:.2f}%)")
            else:
                st.warning("Not enough data to predict.")

            update_bankroll(result, st.session_state.current_bet)
        else:
            st.info("Tie selected – no impact on bankroll.")

# --- Stats Display ---
st.markdown("### Game Stats")
st.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
st.write(f"Current Bet: ${st.session_state.current_bet:.2f}")
st.write(f"Wins: {st.session_state.wins}")
st.write(f"Losses: {st.session_state.losses}")
st.write(f"Recent Sequence: {'-'.join(st.session_state.sequence[-20:])}")

# --- Reset Option ---
if st.button("Reset All Data"):
    reset_data()
    st.success("Data has been reset.")

# Optional: setup form for bankroll configuration
with st.expander("Setup"):
    with st.form("setup_form"):
        bankroll = st.number_input("Initial Bankroll ($)", min_value=1.0, value=st.session_state.bankroll)
        base_unit_pct = st.slider("Base Unit % of Bankroll", 1, 10, 5)
        save_setup = st.form_submit_button("Save")

        if save_setup:
            st.session_state.bankroll = bankroll
            st.session_state.unit = round(base_unit_pct / 100 * bankroll, 2)
            st.session_state.current_bet = st.session_state.unit
            st.success("Configuration saved.")
