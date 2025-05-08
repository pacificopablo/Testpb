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
if 'consecutive_losses' not in st.session_state:
    st.session_state.consecutive_losses = 0
if 'pending_bet' not in st.session_state:
    st.session_state.pending_bet = None
if 'advice' not in st.session_state:
    st.session_state.advice = ""

BASE_UNIT = 0.05  # 5% of bankroll
BET_INCREMENT = 1  # Add $1 for each consecutive loss
MAX_UNIT_PERCENTAGE = 0.10  # Cap bet at 10% of bankroll

def calculate_bet():
    base_bet = st.session_state.bankroll * BASE_UNIT
    adjusted_bet = base_bet + (st.session_state.consecutive_losses * BET_INCREMENT)
    max_bet = st.session_state.bankroll * MAX_UNIT_PERCENTAGE
    return min(adjusted_bet, max_bet)

def record_result(result):
    bet_amount = calculate_bet()
    win = (result == st.session_state.pending_bet)
    if win:
        st.session_state.wins += 1
        gain = bet_amount if result == "P" else bet_amount * 0.95  # 5% commission on banker
        st.session_state.bankroll += gain
        st.session_state.consecutive_losses = 0
        outcome = "Win"
        st.session_state.advice = f"Win! Gained ${gain:.2f}"
    else:
        st.session_state.losses += 1
        st.session_state.bankroll -= bet_amount
        st.session_state.consecutive_losses += 1
        outcome = "Loss"
        st.session_state.advice = f"Loss. Lost ${bet_amount:.2f}"

    st.session_state.prediction_accuracy[result] += 1
    st.session_state.prediction_accuracy['total'] += 1

    st.session_state.sequence.append(result)
    st.session_state.history.append({
        'Time': datetime.datetime.now().strftime("%H:%M:%S"),
        'Bet': st.session_state.pending_bet,
        'Result': result,
        'Outcome': outcome,
        'Amount': round(bet_amount, 2),
        'Bankroll': round(st.session_state.bankroll, 2)
    })

    st.session_state.pending_bet = None

# Display interface
st.title("Algo Z100 Baccarat AI Strategy")

# Prediction logic placeholder
st.session_state.pending_bet = "P"  # Example: Always bet on Player

st.subheader("Enter Result")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Player"):
        record_result("P")
with col2:
    if st.button("Banker"):
        record_result("B")
with col3:
    if st.button("Tie"):
        st.session_state.sequence.append("T")
        st.session_state.advice = "Tie recorded. No bankroll change."

# Undo last
if st.button("Undo Last"):
    if st.session_state.history:
        last = st.session_state.history.pop()
        st.session_state.sequence.pop()
        if last['Outcome'] == "Win":
            refund = last['Amount'] if last['Bet'] == "P" else last['Amount'] * 0.95
            st.session_state.bankroll -= refund
            st.session_state.wins -= 1
        else:
            st.session_state.bankroll += last['Amount']
            st.session_state.losses -= 1
            st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
        st.session_state.prediction_accuracy[last['Result']] -= 1
        st.session_state.prediction_accuracy['total'] -= 1
        st.session_state.advice = "Last entry undone."

# Show stats
st.subheader("Game Stats")
st.metric("Bankroll", f"${st.session_state.bankroll:,.2f}")
st.metric("Wins", st.session_state.wins)
st.metric("Losses", st.session_state.losses)
st.info(st.session_state.advice)

# Show history
st.subheader("Game History")
st.dataframe(st.session_state.history[::-1])
