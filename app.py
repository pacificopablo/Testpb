import streamlit as st
import datetime

# Initialize session state
for key, value in {
    'sequence': [],
    'history': [],
    'bankroll': 46000,
    'wins': 0,
    'losses': 0,
    'prediction_accuracy': {'P': 0, 'B': 0, 'T': 0, 'total': 0},
    'consecutive_losses': 0,
    'pending_bet': None,
    'advice': '',
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

BASE_UNIT = 0.05  # 5% of bankroll
BET_INCREMENT = 1
MAX_UNIT_PERCENTAGE = 0.10  # 10% max stake

def calculate_bet():
    base_bet = st.session_state.bankroll * BASE_UNIT
    adjusted = base_bet + (st.session_state.consecutive_losses * BET_INCREMENT)
    return round(min(adjusted, st.session_state.bankroll * MAX_UNIT_PERCENTAGE), 2)

def place_result(result):
    bet = result  # We assume user bets based on what button is clicked
    amount = calculate_bet()

    win = (bet == result)
    if win:
        st.session_state.wins += 1
        gain = amount if result == 'P' else round(amount * 0.95, 2)
        st.session_state.bankroll += gain
        st.session_state.consecutive_losses = 0
        st.session_state.advice = f"Win! Gained ${gain}"
    else:
        st.session_state.losses += 1
        st.session_state.bankroll -= amount
        st.session_state.consecutive_losses += 1
        st.session_state.advice = f"Loss. Lost ${amount}"

    st.session_state.prediction_accuracy[result] += 1
    st.session_state.prediction_accuracy['total'] += 1

    st.session_state.sequence.append(result)
    st.session_state.history.append({
        'Time': datetime.datetime.now().strftime("%H:%M:%S"),
        'Bet': bet,
        'Result': result,
        'Win': win,
        'Amount': amount,
        'Bankroll': st.session_state.bankroll
    })

# Interface
st.title("Algo Z100 Baccarat AI Strategy")

st.subheader("Enter Result")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Player", use_container_width=True):
        place_result("P")

with col2:
    if st.button("Banker", use_container_width=True):
        place_result("B")

with col3:
    if st.button("Tie", use_container_width=True):
        st.session_state.sequence.append("T")
        st.session_state.advice = "Tie recorded. No bankroll change."

# Undo feature
if st.button("Undo Last"):
    if st.session_state.history:
        last = st.session_state.history.pop()
        st.session_state.sequence.pop()
        if last['Win']:
            st.session_state.wins -= 1
            refund = last['Amount'] if last['Bet'] == 'P' else round(last['Amount'] * 0.95, 2)
            st.session_state.bankroll -= refund
        else:
            st.session_state.losses -= 1
            st.session_state.bankroll += last['Amount']
            st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
        st.session_state.prediction_accuracy[last['Result']] -= 1
        st.session_state.prediction_accuracy['total'] -= 1
        st.session_state.advice = "Last entry undone."

# Display
st.subheader("Stats")
st.metric("Bankroll", f"${st.session_state.bankroll:,.2f}")
st.metric("Wins", st.session_state.wins)
st.metric("Losses", st.session_state.losses)

st.info(st.session_state.advice)

st.subheader("Game History")
st.dataframe(st.session_state.history[::-1])
