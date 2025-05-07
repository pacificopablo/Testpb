import streamlit as st import random

st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP") st.title("MANG BACCARAT GROUP")

--- SESSION STATE INIT ---

if 'bankroll' not in st.session_state: st.session_state.bankroll = 0.0 st.session_state.base_bet = 0.0 st.session_state.sequence = [] st.session_state.pending_bet = None st.session_state.strategy = 'T3' st.session_state.t3_level = 1 st.session_state.t3_results = [] st.session_state.advice = "" st.session_state.history = [] st.session_state.wins = 0 st.session_state.losses = 0 st.session_state.target_type = 'None' st.session_state.target_value = 0.0 st.session_state.target_hit = False

--- RESET BUTTON ---

if st.button("Reset Session"): for key in list(st.session_state.keys()): del st.session_state[key] st.experimental_rerun()

--- SETUP FORM ---

st.subheader("Setup") with st.form("setup_form"): bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0) base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0) strategy = st.selectbox("Choose Strategy", ["T3", "Flatbet"], index=["T3", "Flatbet"].index(st.session_state.strategy)) target_type = st.selectbox("Set Profit Target By", ["None", "Percent", "Units"], index=["None", "Percent", "Units"].index(st.session_state.target_type)) if target_type == "Percent": target_value = st.number_input("Target Profit (%)", min_value=1.0, value=10.0) elif target_type == "Units": target_value = st.number_input("Target Profit (Units)", min_value=1.0, value=5.0) else: target_value = 0.0 start_clicked = st.form_submit_button("Start Session")

if start_clicked: st.session_state.bankroll = bankroll st.session_state.base_bet = base_bet st.session_state.strategy = strategy st.session_state.sequence = [] st.session_state.pending_bet = None st.session_state.t3_level = 1 st.session_state.t3_results = [] st.session_state.advice = "" st.session_state.history = [] st.session_state.wins = 0 st.session_state.losses = 0 st.session_state.target_type = target_type st.session_state.target_value = target_value st.session_state.target_hit = False st.success("Session started!")

--- PREDICTION LOGIC (RANDOM AI) ---

def predict_next(): if st.session_state.target_hit: return None side = random.choice(['P', 'B']) return side

def place_result(result): bet_amount = 0 if st.session_state.pending_bet: bet_amount, selection = st.session_state.pending_bet win = result == selection if win: if selection == 'B': st.session_state.bankroll += bet_amount * 0.95 else: st.session_state.bankroll += bet_amount st.session_state.t3_results.append('W') st.session_state.wins += 1 else: st.session_state.bankroll -= bet_amount st.session_state.t3_results.append('L') st.session_state.losses += 1

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

# Check target hit
if st.session_state.base_bet > 0:
    bankroll_change = 0
    for h in st.session_state.history:
        amt = h["Amount"]
        if h["Win"]:
            bankroll_change += amt if h["Bet"] == "P" else amt * 0.95
        else:
            bankroll_change -= amt
    starting_bankroll = st.session_state.bankroll - bankroll_change

    if st.session_state.target_type == "Percent":
        gain = ((st.session_state.bankroll - starting_bankroll) / starting_bankroll) * 100
        if gain >= st.session_state.target_value:
            st.session_state.target_hit = True
    elif st.session_state.target_type == "Units":
        units_profit = (st.session_state.bankroll - starting_bankroll) // st.session_state.base_bet
        if units_profit >= st.session_state.target_value:
            st.session_state.target_hit = True

# Predict next
if not st.session_state.target_hit:
    pred = predict_next()
    bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
    if bet_amount <= st.session_state.bankroll:
        st.session_state.pending_bet = (bet_amount, pred)
    else:
        st.session_state.advice = "Insufficient bankroll"

--- RESULT INPUT ---

st.subheader("Enter Result") col1, col2, col3 = st.columns(3) with col1: if st.button("Player (P)"): place_result("P") with col2: if st.button("Banker (B)"): place_result("B") with col3: if st.button("Undo Last"): if st.session_state.sequence: st.session_state.sequence.pop() if st.session_state.history: last = st.session_state.history.pop() if last['Win']: st.session_state.wins -= 1 st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95 else: st.session_state.losses -= 1 st.session_state.bankroll += last['Amount'] st.session_state.pending_bet = None

--- DISPLAY SEQUENCE ---

st.subheader("Current Sequence") latest_sequence = st.session_state.sequence[-20:] if 'sequence' in st.session_state else [] st.text(", ".join(latest_sequence or ["None"]))

--- PREDICTION BOX (Between Sequence and Status) ---

if st.session_state.pending_bet: amount, side = st.session_state.pending_bet color = "#add8e6" if side == "P" else "#f08080" st.markdown( f""" <div style='padding: 12px; border-radius: 10px; background-color: {color}; text-align: center; font-weight: bold; font-size: 18px;'> Next Bet: ${amount:.0f} on {"Player" if side == "P" else "Banker"} </div> """, unsafe_allow_html=True ) else: st.info("No pending bet yet.")

--- STATUS ---

st.subheader("Status") st.markdown(f"Bankroll: ${st.session_state.bankroll:.2f}") st.markdown(f"Base Bet: ${st.session_state.base_bet:.2f}") st.markdown(f"Strategy: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}") st.markdown(f"Wins: {st.session_state.wins} | Losses: {st.session_state.losses}")

if st.session_state.base_bet > 0: bankroll_change = 0 for h in st.session_state.history: amt = h["Amount"] if h["Win"]: bankroll_change += amt if h["Bet"] == "P" else amt * 0.95 else: bankroll_change -= amt starting_bankroll = st.session_state.bankroll - bankroll_change units_profit = int((st.session_state.bankroll - starting_bankroll) // st.session_state.base_bet) st.markdown(f"Units Profit: {units_profit}")

if st.session_state.target_hit: st.success("Target profit reached! No further bets will be made.")

--- HISTORY ---

if st.session_state.history: st.subheader("Recent Bet History") history_df = st.session_state.history[-10:] st.table([ { "Bet": h["Bet"], "Result": h["Result"], "Amount": f"${h['Amount']:.0f}", "Outcome": "Win" if h["Win"] else "Loss" } for h in history_df ])

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

