import streamlit as st
import random
import time

# --- PAGE CONFIG ---
st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 0.0
    st.session_state.base_bet = 0.0
    st.session_state.strategy = 'T3'
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.advice = ""
    st.session_state.target_type = "None"
    st.session_state.target_value = 0.0
    st.session_state.target_hit = False

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
    target_type = st.selectbox("Set Profit Target By", ["None", "Percent", "Units"], index=["None", "Percent", "Units"].index(st.session_state.target_type))
    
    target_value = 0.0
    if target_type == "Percent":
        target_value = st.number_input("Target Profit (%)", min_value=0.0, step=1.0)
    elif target_type == "Units":
        target_value = st.number_input("Target Profit (Units)", min_value=0.0, step=1.0)

    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    st.session_state.bankroll = bankroll
    st.session_state.base_bet = base_bet
    st.session_state.strategy = strategy
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.advice = ""
    st.session_state.target_type = target_type
    st.session_state.target_value = target_value
    st.session_state.target_hit = False
    st.session_state.starting_bankroll = bankroll
    st.success("Session started!")

# --- FUNCTIONS ---
def get_random_bet():
    return random.choice(['P', 'B'])

def predict_next():
    return get_random_bet(), 50

def place_result(result):
    if st.session_state.target_hit:
        st.warning("Target already hit. Reset session to continue.")
        return

    bet_amount = 0
    if st.session_state.pending_bet:
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        if win:
            st.session_state.bankroll += bet_amount if selection == 'P' else bet_amount * 0.95
            st.session_state.wins += 1
            st.session_state.t3_results.append('W')
        else:
            st.session_state.bankroll -= bet_amount
            st.session_state.losses += 1
            st.session_state.t3_results.append('L')

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

    # Trim sequence to last 100
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Check target
    profit = st.session_state.bankroll - st.session_state.starting_bankroll
    if st.session_state.target_type == "Percent":
        if profit >= st.session_state.starting_bankroll * (st.session_state.target_value / 100):
            st.session_state.target_hit = True
    elif st.session_state.target_type == "Units":
        if profit >= st.session_state.base_bet * st.session_state.target_value:
            st.session_state.target_hit = True

    # Prepare next random AI bet
    if not st.session_state.target_hit:
        pred, conf = predict_next()
        bet_amount = st.session_state.base_bet if st.session_state.strategy == "Flatbet" else st.session_state.base_bet * st.session_state.t3_level
        if bet_amount <= st.session_state.bankroll:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} (AI)"
        else:
            st.session_state.advice = "Insufficient bankroll"
    else:
        st.session_state.advice = "Target hit — no further bets."

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
                    st.session_state.bankroll -= last['Amount'] if last['Bet'] == 'P' else last['Amount'] * 0.95
                else:
                    st.session_state.losses -= 1
                    st.session_state.bankroll += last['Amount']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."
            st.session_state.target_hit = False

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence")
st.text(", ".join(st.session_state.sequence[-20:] or ["None"]))

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

# --- PROFIT ---
if st.session_state.base_bet > 0:
    profit = st.session_state.bankroll - st.session_state.starting_bankroll
    units_profit = int(profit // st.session_state.base_bet)
    st.markdown(f"**Units Profit**: {units_profit}")
    st.markdown(f"**Profit**: ${profit:.2f}")

# --- TARGET ---
if st.session_state.target_type != "None":
    st.subheader("Target Status")
    st.markdown(f"**Target Type**: {st.session_state.target_type}")
    st.markdown(f"**Target Value**: {st.session_state.target_value}")
    if st.session_state.target_hit:
        st.success("Target reached!")
    else:
        st.info("Target not yet hit.")

# --- PENDING BET ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    st.success(f"Pending Bet: ${amount:.0f} on {side}")
else:
    st.info("No pending bet.")
st.write(st.session_state.advice)

# --- HISTORY ---
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
