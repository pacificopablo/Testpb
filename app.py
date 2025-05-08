import streamlit as st
import datetime

# Initialize session state
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 46000
if 'wins' not in st.session_state:
    st.session_state.wins = 0
if 'losses' not in st.session_state:
    st.session_state.losses = 0
if 'prediction_accuracy' not in st.session_state:
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'T': 0, 'total': 0}
if 't3_level' not in st.session_state:
    st.session_state.t3_level = 1
if 't3_results' not in st.session_state:
    st.session_state.t3_results = []
if 'pending_bet' not in st.session_state:
    st.session_state.pending_bet = None
if 'advice' not in st.session_state:
    st.session_state.advice = ""
if 'last_was_tie' not in st.session_state:
    st.session_state.last_was_tie = False
if 'consecutive_losses' not in st.session_state:
    st.session_state.consecutive_losses = 0

# Betting system settings
BASE_UNIT = 0.05  # 5% of bankroll
BET_INCREMENT = 1  # Add $1 after loss
MAX_UNIT_PERCENTAGE = 0.1  # 10% cap

# Title
st.title("Algo Z100 Baccarat AI Strategy Tracker")

# Helper: Calculate next bet
def calculate_bet():
    base_bet = st.session_state.bankroll * BASE_UNIT
    adjusted_bet = base_bet + (st.session_state.consecutive_losses * BET_INCREMENT)
    max_bet = st.session_state.bankroll * MAX_UNIT_PERCENTAGE
    return round(min(adjusted_bet, max_bet), 2)

# Process result
def place_result(result):
    if result == 'T':
        st.session_state.last_was_tie = True
        st.session_state.advice = "Tie recorded. No bankroll change."
        st.session_state.sequence.append(result)
        return

    bet = st.session_state.pending_bet if st.session_state.pending_bet else result
    amount = calculate_bet()

    win = (bet == result)
    if win:
        st.session_state.wins += 1
        gain = amount if result == 'P' else amount * 0.95
        st.session_state.bankroll += gain
        st.session_state.prediction_accuracy[result] += 1
        st.session_state.consecutive_losses = 0
        st.session_state.advice = f"Win! Gained ${gain:.2f}"
    else:
        st.session_state.losses += 1
        st.session_state.bankroll -= amount
        st.session_state.consecutive_losses += 1
        st.session_state.advice = f"Loss. Lost ${amount:.2f}"

    st.session_state.prediction_accuracy['total'] += 1
    st.session_state.history.append({
        'Time': datetime.datetime.now().strftime("%H:%M:%S"),
        'Bet': bet,
        'Result': result,
        'Win': win,
        'Amount': amount,
        'T3_Level': st.session_state.t3_level,
        'T3_Results': st.session_state.t3_results.copy()
    })

    st.session_state.sequence.append(result)
    st.session_state.pending_bet = None
    st.session_state.last_was_tie = False

# UI for result entry
st.subheader("Enter Result")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Player", use_container_width=True):
        place_result("P")

with col2:
    if st.button("Banker", use_container_width=True):
        place_result("B")

with col3:
    if st.button("Tie", use_container_width=True):
        place_result("T")

with col4:
    if st.button("Undo Last", use_container_width=True):
        if st.session_state.history and st.session_state.sequence:
            st.session_state.sequence.pop()
            last = st.session_state.history.pop()
            if last['Win']:
                st.session_state.wins -= 1
                st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                st.session_state.prediction_accuracy[last['Bet']] -= 1
                st.session_state.consecutive_losses = 0
            else:
                st.session_state.bankroll += last['Amount']
                st.session_state.losses -= 1
                st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
            st.session_state.prediction_accuracy['total'] -= 1
            st.session_state.t3_level = last['T3_Level']
            st.session_state.t3_results = last['T3_Results']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."
            st.session_state.last_was_tie = False

# Display stats
st.subheader("Dashboard")

colA, colB, colC = st.columns(3)
colA.metric("Bankroll", f"${st.session_state.bankroll:,.2f}")
colB.metric("Wins", st.session_state.wins)
colC.metric("Losses", st.session_state.losses)

st.progress(min(st.session_state.prediction_accuracy['total'], 100), "Total Entries")

st.write("### Advice")
st.info(st.session_state.advice)

# Show history
st.write("### Game History")
st.dataframe(st.session_state.history[::-1])
