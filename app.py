import streamlit as st
import random

st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 0.0
    st.session_state.base_bet = 0.0
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.strategy = 'T3'
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = ""
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_mode = "Profit %"
    st.session_state.target_value = 10.0
    st.session_state.starting_bankroll = 0.0
    st.session_state.confidence = 0

# --- RESET BUTTON ---
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    strategy = st.selectbox("Choose Strategy", ["T3", "Flatbet"], index=["T3", "Flatbet"].index(st.session_state.strategy))
    target_mode = st.selectbox("Target Mode", ["Profit %", "Units"], index=["Profit %", "Units"].index(st.session_state.target_mode))
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    st.session_state.bankroll = bankroll
    st.session_state.base_bet = base_bet
    st.session_state.strategy = strategy
    st.session_state.target_mode = target_mode
    st.session_state.target_value = target_value
    st.session_state.starting_bankroll = bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = ""
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.success("Session started!")

# --- FUNCTIONS ---
def predict_random():
    return random.choice(["P", "B"]), 50.0

def predict_next():
    if len(st.session_state.sequence) < 1:
        return None, 0
    return predict_random()

def place_result(result):
    bet_amount = 0
    if st.session_state.pending_bet:
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        if win:
            if selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:
                st.session_state.bankroll += bet_amount
            st.session_state.t3_results.append('W')
            st.session_state.wins += 1
        else:
            st.session_state.bankroll -= bet_amount
            st.session_state.t3_results.append('L')
            st.session_state.losses += 1

        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win
        })

        if len(st.session_state.t3_results) == 3:
            w = st.session_state.t3_results.count('W')
            l = st.session_state.t3_results.count('L')
            if w == 3:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif w == 2:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif l == 2:
                st.session_state.t3_level += 1
            elif l == 3:
                st.session_state.t3_level += 2
            st.session_state.t3_results = []

        st.session_state.pending_bet = None

    st.session_state.sequence.append(result)

    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    pred, conf = predict_next()
    st.session_state.confidence = conf
    if pred:
        bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
        if bet_amount <= st.session_state.bankroll:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.0f}%)"
        else:
            st.session_state.advice = "Insufficient bankroll"
    else:
        st.session_state.advice = "Waiting for more entries"

    # Check target hit
    profit = st.session_state.bankroll - st.session_state.starting_bankroll
    if st.session_state.target_mode == "Profit %":
        target_amount = st.session_state.starting_bankroll * (1 + st.session_state.target_value / 100)
    else:
        target_amount = st.session_state.starting_bankroll + (st.session_state.base_bet * st.session_state.target_value)

    if st.session_state.bankroll >= target_amount:
        st.balloons()
        st.success("Target reached! Session complete.")

# --- RESULT INPUT ---
st.subheader("Enter Result")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Player (P)"):
        place_result("P")
with col2:
    if st.button("Banker (B)"):
        place_result("B")
with col3:
    if st.button("Undo Last"):
        if st.session_state.sequence:
            st.session_state.sequence.pop()
            if st.session_state.history:
                last = st.session_state.history.pop()
                if last['Win']:
                    st.session_state.wins -= 1
                    st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                else:
                    st.session_state.losses -= 1
                    st.session_state.bankroll += last['Amount']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence")
latest_sequence = st.session_state.sequence[-20:] if 'sequence' in st.session_state else []
st.text(", ".join(latest_sequence or ["None"]))

# --- PREDICTION BLOCK ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    color = "blue" if side == "P" else "red"
    side_full = "Player" if side == "P" else "Banker"
    confidence = st.session_state.get("confidence", 0)

    if confidence >= 70:
        bg_color = "#d4edda"
    elif confidence >= 50:
        bg_color = "#fff3cd"
    else:
        bg_color = "#f8f9fa"

    st.markdown(
        f"""
        <div style='padding: 12px; border-radius: 10px; background-color: {bg_color}; text-align: center; border: 1px solid #ccc;'>
            <h4 style='color:{color}; margin-bottom: 5px;'>Prediction: <strong>{side_full}</strong></h4>
            <p style='margin: 4px 0;'><strong>Bet Amount:</strong> ${amount:.0f}</p>
            <p style='margin: 4px 0;'><strong>Win Probability:</strong> {confidence:.0f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("No pending bet yet.")

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

# --- UNIT PROFIT ---
if st.session_state.base_bet > 0:
    bankroll_change = 0
    for h in st.session_state.history:
        amt = h["Amount"]
        if h["Win"]:
            bankroll_change += amt if h["Bet"] == "P" else amt * 0.95
        else:
            bankroll_change -= amt
    starting_bankroll = st.session_state.starting_bankroll
    units_profit = int((st.session_state.bankroll - starting_bankroll) // st.session_state.base_bet)
    st.markdown(f"**Units Profit**: {units_profit}")

# --- HISTORY TABLE ---
if st.session_state.history:
    st.subheader("Recent Bet History")
    history_df = st.session_state.history[-10:]
    st.table([
        {
            "Bet": h["Bet"],
            "Result": h["Result"],
            "Amount": f"${h['Amount']:.0f}",
            "Outcome": "Win" if h["Win"] else "Loss"
        }
        for h in history_df
    ])

    st.subheader("Full History")
    st.dataframe([
        {
            "Bet": h["Bet"],
            "Result": h["Result"],
            "Amount": f"${h['Amount']:.0f}",
            "Outcome": "Win" if h["Win"] else "Loss"
        }
        for h in st.session_state.history
    ])
