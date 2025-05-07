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
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    st.session_state.bankroll = bankroll
    st.session_state.base_bet = base_bet
    st.session_state.strategy = strategy
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = ""
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.success("Session started!")

# --- RANDOM AI BET SELECTION ---
def generate_ai_bet():
    return random.choice(['P', 'B'])

# --- PLACE RESULT FUNCTION ---
def place_result(result):
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

    # Random AI bet selection
    pred = generate_ai_bet()
    bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
    if bet_amount <= st.session_state.bankroll:
        st.session_state.pending_bet = (bet_amount, pred)
        st.session_state.advice = f"AI Bet: ${bet_amount:.0f} on {pred}"
    else:
        st.session_state.advice = "Insufficient bankroll"

# --- RESULT INPUT BUTTONS ---
st.subheader("Enter Result")
col1, col2 = st.columns(2)
with col1:
    if st.button("Player (P)"):
        place_result("P")
with col2:
    if st.button("Banker (B)"):
        place_result("B")

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence")
latest_sequence = st.session_state.sequence[-20:] if 'sequence' in st.session_state else []
st.text(", ".join(latest_sequence or ["None"]))

# --- STATUS DISPLAY ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

# --- UNIT PROFIT CALCULATION ---
if st.session_state.base_bet > 0:
    bankroll_change = 0
    for h in st.session_state.history:
        amt = h["Amount"]
        if h["Win"]:
            bankroll_change += amt if h["Bet"] == "P" else amt * 0.95
        else:
            bankroll_change -= amt
    starting_bankroll = st.session_state.bankroll - bankroll_change
    units_profit = int((st.session_state.bankroll - starting_bankroll) // st.session_state.base_bet)
    st.markdown(f"**Units Profit**: {units_profit}")

# --- PENDING BET DISPLAY ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    st.success(f"Pending Bet: ${amount:.0f} on {side}")
else:
    st.info("No pending bet yet.")
st.write(st.session_state.advice)

# --- HISTORY TABLES ---
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
