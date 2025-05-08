import streamlit as st
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.button_action = None  # New: Track button actions

# --- RESET BUTTON ---
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()  # Improvement 7: Use st.rerun

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet"],
        index=["T3", "Flatbet"].index(st.session_state.strategy),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Uses a fixed bet size."
    )
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    if bankroll <= 0 or base_bet <= 0 or base_bet > bankroll:
        st.error("Invalid inputs: Bankroll and base bet must be positive, and base bet cannot exceed bankroll.")  # Improvement 6
    else:
        st.session_state.bankroll = bankroll
        st.session_state.base_bet = base_bet
        st.session_state.strategy = betting_strategy
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
        st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
        st.session_state.consecutive_losses = 0
        st.session_state.loss_log = []
        st.session_state.last_was_tie = False
        st.session_state.button_action = None
        st.success("Session started!")

# --- FUNCTIONS ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 5:  # Improvement 3: Minimum 5 non-Tie outcomes
        return 'B', 45.86
    bigram = sequence[-2:]
    transitions = defaultdict(int)
    for i in range(len(sequence) - 2):
        if sequence[i:i+2] == bigram:
            next_outcome = sequence[i+2]
            transitions[next_outcome] += 1
    total_transitions = sum(transitions.values())
    if total_transitions > 0:
        prob_p = (transitions['P'] / total_transitions) * 100
        prob_b = (transitions['B'] / total_transitions) * 100
    else:
        prob_p = 44.62
        prob_b = 45.86
    return ('P', prob_p) if prob_p > prob_b else ('B', prob_b)

def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.base_bet
    return unit_profit >= st.session_state.target_value

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = "Session reset: Target reached."
    st.session_state.history = []
    st.session_state.wis = 0
    st.session_state.losses = 0
    st.session_state.target_hit = False
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.button_action = None

def place_result(result):
    if st.session_state.target_hit:
        reset_session_auto()
        return

    st.session_state.last_was_tie = (result == 'T')

    bet_amount = 0
    if st.session_state.pending_bet and result != 'T':
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        old_bankroll = st.session_state.bankroll
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
            st.session_state.loss_log.append({
                'sequence': st.session_state.sequence[-10:],
                'prediction': selection,
                'result': result,
                'confidence': st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
            })
            if len(st.session_state.loss_log) > 50:
                st.session_state.loss_log = st.session_state.loss_log[-50:]
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

        st.session_state.pending_bet = None

    if not st.session_state.pending_bet and result != 'T':
        st.session_state.consecutive_losses = 0

    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    pred, conf = predict_next()
    if conf < 50.5:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet (Confidence: {conf:.1f}% < 50.5%)"
    else:
        bet_amount = st.session_state.base_bet * st.session_state.t3_level
        if bet_amount > st.session_state.bankroll:  # Improvement 6
            st.session_state.pending_bet = None
            st.session_state.advice = "No bet: Insufficient bankroll."
        else:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"

    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = st.session_state.t3_level + 1
        elif losses == 3:
            st.session_state.t3_level = st.session_state.t3_level + 2
        st.session_state.t3_results = []

# --- RESULT INPUT WITH NATIVE BUTTONS --- (Improvement 1, 8)
st.subheader("Enter Result")
col1, col2, col3, col4 = st.columns(4)  # Improvement 8: Responsive layout
with col1:
    if st.button("Player", use_container_width=True):
        st.session_state.button_action = "P"
with col2:
    if st.button("Banker", use_container_width=True):
        st.session_state.button_action = "B"
with col3:
    if st.button("Tie", use_container_width=True):
        st.session_state.button_action = "T"
with col4:
    if st.button("Undo Last", use_container_width=True):
        st.session_state.button_action = "undo"

# Process button action (Improvement 2)
if st.session_state.button_action:
    action = st.session_state.button_action
    st.session_state.button_action = None  # Reset to avoid reprocessing
    if action in ["P", "B", "T"]:
        place_result(action)
    elif action == "undo":
        if not st.session_state.history or not st.session_state.sequence:
            st.session_state.advice = "Nothing to undo."
        else:
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

# --- BEAD PLATE WITH MATPLOTLIB --- (Updated for smaller size)
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-100:]
grid = []
current_col = []
for result in sequence:
    if len(current_col) < 6:
        current_col.append(result)
    else:
        grid.append(current_col)
        current_col = [result]
if current_col:
    grid.append(current_col)
if grid and len(grid[-1]) < 6:
    grid[-1] += [''] * (6 - len(grid[-1]))

if grid:
    fig, ax = plt.subplots(figsize=(max(1.5, len(grid) * 0.2), 1.0))
    for i, col in enumerate(grid):
        for j, result in enumerate(col):
            if result == 'P':
                ax.add_patch(plt.Circle((i, 5-j), 0.15, color='blue'))
            elif result == 'B':
                ax.add_patch(plt.Circle((i, 5-j), 0.15, color='red'))
            elif result == 'T':
                ax.add_patch(plt.Circle((i, 5-j), 0.15, color='green'))
    ax.set_xlim(-0.5, len(grid) - 0.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(grid)))
    ax.set_yticks(range(6))
    ax.set_xticklabels([])
    ax.set_yticklabels(['6', '5', '4', '3', '2', '1'], fontsize=6)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
else:
    st.write("No results yet.")

# --- PREDICTION DISPLAY ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    color = 'blue' if side == 'P' else 'red'
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.0f} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
else:
    if not st.session_state.target_hit:
        st.info(st.session_state.advice)

# --- UNIT PROFIT ---
if st.session_state.base_bet > 0 and st.session_state.initial_bankroll > 0:
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    units_profit = profit / st.session_state.base_bet
    st.markdown(f"**Units Profit**: {units_profit:.2f} units (${profit:.2f})")
else:
    st.markdown("**Units Profit**: 0.00 units ($0.00)")

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Betting Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

# --- PREDICTION ACCURACY ---
st.subheader("Prediction Accuracy")
total = st.session_state.prediction_accuracy['total']
if total > 0:
    p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
    b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
    st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
    st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

# --- LOSS LOG ---
if st.session_state.loss_log:
    st.subheader("Recent Losses")
    st.dataframe([
        {
            "Sequence": ", ".join(log['sequence']),
            "Prediction": log['prediction'],
            "Result": log['result'],
            "Confidence": log['confidence'] + "%"
        }
        for log in st.session_state.loss_log[-5:]
    ])

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
            "T3_Level": h["T3_Level"]
        }
        for h in st.session_state.history[-n:]
    ])
