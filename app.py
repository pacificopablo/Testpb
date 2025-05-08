import streamlit as st
import random
from collections import defaultdict

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
    st.session_state.target_mode = 'Profit %'
    st.session_state.target_value = 10.0
    st.session_state.initial_bankroll = 0.0
    st.session_state.target_hit = False
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'T': 0, 'total': 0}
    st.session_state.consecutive_losses = 0
    st.session_state.markov_transitions = defaultdict(lambda: {'P': 0, 'B': 0, 'T': 0})

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
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    if bankroll <= 0:
        st.error("Bankroll must be positive.")
    elif base_bet <= 0:
        st.error("Base bet must be positive.")
    elif base_bet > bankroll:
        st.error("Base bet cannot exceed bankroll.")
    else:
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
        st.session_state.target_mode = target_mode
        st.session_state.target_value = target_value
        st.session_state.initial_bankroll = bankroll
        st.session_state.target_hit = False
        st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'T': 0, 'total': 0}
        st.session_state.consecutive_losses = 0
        st.session_state.markov_transitions = defaultdict(lambda: {'P': 0, 'B': 0, 'T': 0})
        st.success("Session started!")

# --- FUNCTIONS ---
def predict_next():
    sequence = st.session_state.sequence[-50:]  # Analyze last 50 results
    if len(sequence) < 2:
        return random.choice(['P', 'B']), 50  # Default if insufficient history

    # Update Markov transitions
    transitions = st.session_state.markov_transitions
    for i in range(len(sequence) - 1):
        current, next_outcome = sequence[i], sequence[i + 1]
        transitions[current][next_outcome] += 1

    # Get probabilities for the last outcome
    last_outcome = sequence[-1]
    counts = transitions[last_outcome]
    total = sum(counts.values())
    if total == 0:
        return random.choice(['P', 'B']), 50

    probs = {k: v / total for k, v in counts.items()}
    p_prob = probs.get('P', 0)
    b_prob = probs.get('B', 0)
    t_prob = probs.get('T', 0)

    # Apply Banker bias (slight preference for B if probabilities are close)
    if abs(p_prob - b_prob) < 0.05 and b_prob > 0:
        b_prob += 0.05
        p_prob = max(0, p_prob - 0.05)

    # Calculate confidence based on max probability and prediction accuracy
    accuracy = st.session_state.prediction_accuracy['total']
    accuracy_rate = st.session_state.prediction_accuracy['P'] / accuracy if accuracy > 0 else 0.5
    max_prob = max(p_prob, b_prob)
    confidence = 50 + (max_prob * 40) * (0.5 + accuracy_rate / 2)
    confidence = min(confidence, 90)

    # Choose prediction
    if b_prob > p_prob:
        return 'B', confidence
    elif p_prob > b_prob:
        return 'P', confidence
    else:
        return 'B', confidence  # Default to Banker in ties

def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        if st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit:
            return True
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.base_bet
        if unit_profit >= st.session_state.target_value:
            return True
    return False

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = "Session reset: Target reached."
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_hit = False
    st.session_state.consecutive_losses = 0

def place_result(result):
    if st.session_state.target_hit:
        reset_session_auto()
        return

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
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
        else:
            st.session_state.bankroll -= bet_amount
            st.session_state.t3_results.append('L')
            st.session_state.losses += 1
            st.session_state.consecutive_losses += 1
        st.session_state.prediction_accuracy['total'] += 1

        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level,
            "T3_Results": st.session_state.t3_results.copy()
        })
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]

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

    # Reset consecutive losses for ties or non-bet results
    if result == 'T' or not st.session_state.pending_bet:
        st.session_state.consecutive_losses = 0

    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    # Check loss avoidance conditions
    pred, conf = predict_next()
    bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
    if st.session_state.consecutive_losses >= 4:
        st.session_state.pending_bet = None
        st.session_state.advice = f"Skip bet: {st.session_state.consecutive_losses} consecutive losses"
    elif bet_amount > st.session_state.bankroll:
        st.session_state.pending_bet = None
        st.session_state.advice = "Skip bet: Insufficient bankroll"
    elif conf < 55:
        st.session_state.pending_bet = None
        st.session_state.advice = f"Skip bet: Low confidence ({conf:.0f}%)"
    else:
        st.session_state.pending_bet = (bet_amount, pred)
        st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.0f}%)"

# --- RESULT INPUT ---
st.subheader("Enter Result")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Player (P)"):
        place_result("P")
with col2:
    if st.button("Banker (B)"):
        place_result("B")
with col3:
    if st.button("Tie (T)"):
        place_result("T")
with col4:
    if st.button("Undo Last"):
        if st.session_state.history and st.session_state.sequence:
            st.session_state.sequence.pop()
            last = st.session_state.history.pop()
            if last['Win']:
                st.session_state.wins -= 1
                st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                st.session_state.prediction_accuracy[last['Bet']] -= 1
                st.session_state.consecutive_losses = 0
            else:
                st.session_state.losses -= 1
                st.session_state.bankroll += last['Amount']
                st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
            st.session_state.prediction_accuracy['total'] -= 1
            st.session_state.t3_level = last['T3_Level']
            st.session_state.t3_results = last['T3_Results']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence")
latest_sequence = st.session_state.sequence[-20:] if 'sequence' in st.session_state else []
st.text(", ".join(latest_sequence or ["None"]))

# --- PREDICTION DISPLAY ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    color = 'blue' if side == 'P' else 'red'
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.0f} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
else:
    if not st.session_state.target_hit:
        st.info(st.session_state.advice)

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
st.markdown(f"**Consecutive Losses**: {st.session_state.consecutive_losses}")

# --- PREDICTION ACCURACY ---
st.subheader("Prediction Accuracy")
total = st.session_state.prediction_accuracy['total']
if total > 0:
    p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
    b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
    st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
    st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

# --- UNIT PROFIT ---
if st.session_state.base_bet > 0:
    units_profit = int((st.session_state.bankroll - st.session_state.initial_bankroll) // st.session_state.base_bet)
    st.markdown(f"**Units Profit**: {units_profit}")

# --- HISTORY TABLE ---
if st.session_state.history:
    st.subheader("Bet History")
    n = st.slider("Show last N bets", 5, 50, 10)
    st.dataframe([
        {
            "Bet": h["Bet"],
            "Result": h["Result"],
            "Amount": f"${h['Amount']:.0f}",
            "Outcome": "Win" if h["Win"] else "Loss",
            "T3 Level": h["T3_Level"]
        }
        for h in st.session_state.history[-n:]
    ])
