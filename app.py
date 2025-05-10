import streamlit as st
import random
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta

# --- FILE-BASED SESSION TRACKING ---
SESSION_FILE = "online_users.txt"

def track_user_session_file():
    # Generate a unique session ID if not already set
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())
    
    # Read and clean up expired sessions
    sessions = {}
    current_time = datetime.now()
    try:
        with open(SESSION_FILE, 'r') as f:
            for line in f:
                session_id, timestamp = line.strip().split(',')
                last_seen = datetime.fromisoformat(timestamp)
                if current_time - last_seen <= timedelta(seconds=30):
                    sessions[session_id] = last_seen
    except FileNotFoundError:
        pass
    
    # Add or update current session
    sessions[st.session_state.session_id] = current_time
    
    # Write back active sessions
    try:
        with open(SESSION_FILE, 'w') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except:
        return 0  # Fallback if file access fails
    
    return len(sessions)

# --- APP CONFIG ---
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
    st.session_state.t3_level_changes = 0  # Track T3 level changes
    st.session_state.parlay_step = 1
    st.session_state.parlay_wins = 0
    st.session_state.parlay_using_base = True
    st.session_state.parlay_step_changes = 0  # Track Parlay step changes
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

# Validate strategy on every load
if 'strategy' in st.session_state and st.session_state.strategy not in ['T3', 'Flatbet', 'Parlay16']:
    st.session_state.strategy = 'T3'

# --- RESET BUTTON ---
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.t3_level_changes = 0
    st.session_state.parlay_step_changes = 0
    st.experimental_rerun()

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet", "Parlay16"],
        index={'T3': 0, 'Flatbet': 1, 'Parlay16': 2}.get(st.session_state.strategy, 0),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Uses a fixed bet size. Parlay16: Follows a 16-step Parlay progression based on wins/losses, resets after 2 wins."
    )
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
        st.session_state.strategy = betting_strategy
        st.session_state.sequence = []
        st.session_state.pending_bet = None
        st.session_state.t3_level = 1
        st.session_state.t3_results = [] if betting_strategy == 'T3' else []
        st.session_state.t3_level_changes = 0  # Reset T3 level changes
        st.session_state.parlay_step = 1
        st.session_state.parlay_wins = 0
        st.session_state.parlay_using_base = True
        st.session_state.parlay_step_changes = 0  # Reset Parlay step changes
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
        if betting_strategy == 'Flatbet':
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.t3_level_changes = 0
            st.session_state.parlay_step = 1
            st.session_state.parlay_wins = 0
            st.session_state.parlay_using_base = True
            st.session_state.parlay_step_changes = 0
        elif betting_strategy == 'Parlay16':
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.t3_level_changes = 0
            st.session_state.parlay_step = 1
            st.session_state.parlay_wins = 0
            st.session_state.parlay_using_base = True
            st.session_state.parlay_step_changes = 0
        st.success(f"Session started with {betting_strategy} strategy!")

# --- PARLAY TABLE ---
PARLAY_TABLE = {
    1: {'base': 10, 'parlay': 20},
    2: {'base': 10, 'parlay': 20},
    3: {'base': 10, 'parlay': 20},
    4: {'base': 20, 'parlay': 40},
    5: {'base': 30, 'parlay': 60},
    6: {'base': 40, 'parlay': 80},
    7: {'base': 60, 'parlay': 120},
    8: {'base': 80, 'parlay': 160},
    9: {'base': 120, 'parlay': 240},
    10: {'base': 160, 'parlay': 320},
    11: {'base': 220, 'parlay': 440},
    12: {'base': 300, 'parlay': 600},
    13: {'base': 400, 'parlay': 800},
    14: {'base': 520, 'parlay': 1040},
    15: {'base': 700, 'parlay': 1400},
    16: {'base': 950, 'parlay': 1900}
}

# --- FUNCTIONS ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 2:
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
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.base_bet
        return unit_profit >= st.session_state.target_value

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.t3_level_changes = 0
    st.session_state.parlay_step = 1
    st.session_state.parlay_wins = 0
    st.session_state.parlay_using_base = True
    st.session_state.parlay_step_changes = 0
    st.session_state.advice = "Session reset: Target reached."
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_hit = False
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False

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
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('W')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins += 1
                if st.session_state.parlay_wins == 2:
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = 1
                    st.session_state.parlay_wins = 0
                    st.session_state.parlay_using_base = True
                    if old_step != st.session_state.parlay_step:
                        st.session_state.parlay_step_changes += 1
                else:
                    st.session_state.parlay_using_base = False
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
        else:
            st.session_state.bankroll -= bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('L')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins = 0
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
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
            "T3_Level": st.session_state.t3_level if st.session_state.strategy == 'T3' else 1,
            "T3_Results": st.session_state.t3_results.copy() if st.session_state.strategy == 'T3' else [],
            "Parlay_Step": st.session_state.parlay_step if st.session_state.strategy == 'Parlay16' else 1
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
        if st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
        elif st.session_state.strategy == 'Parlay16':
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = (st.session_state.base_bet / 10) * PARLAY_TABLE[st.session_state.parlay_step][key]
            if bet_amount > st.session_state.bankroll:
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
                bet_amount = (st.session_state.base_bet / 10) * PARLAY_TABLE[st.session_state.parlay_step]['base']
        if bet_amount > st.session_state.bankroll:
            st.session_state.pending_bet = None
            st.session_state.advice = "No bet: Insufficient bankroll."
            if st.session_state.strategy == 'Parlay16':
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
        else:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"
    if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses DASH = st.session_state.t3_results.count('L')
        old_level = st.session_state.t3_level
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = st.session_state.t3_level + 1
        elif losses == 3:
            st.session_state.t3_level = st.session_state.t3_level + 2
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_results = []

# --- RESULT INPUT ---
st.subheader("Enter Result")
st.markdown("""
<style>
div.stButton > button {
    width: 90px;
    height: 35px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 6px;
    border: 1px solid;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
}
div.stButton > button:active {
    transform: scale(0.95);
    box-shadow: none;
}
div.stButton > button[kind="player_btn"] {
    background: linear-gradient(to bottom, #007bff, #0056b3);
    border-color: #0056b3;
    color: white;
}
div.stButton > button[kind="player_btn"]:hover {
    background: linear-gradient(to bottom, #339cff, #007bff);
}
div.stButton > button[kind="banker_btn"] {
    background: linear-gradient(to bottom, #dc3545, #a71d2a);
    border-color: #a71d2a;
    color: white;
}
div.stButton > button[kind="banker_btn"]:hover {
    background: linear-gradient(to bottom, #ff6666, #dc3545);
}
div.stButton > button[kind="tie_btn"] {
    background: linear-gradient(to bottom, #28a745, #1e7e34);
    border-color: #1e7e34;
    color: white;
}
div.stButton > button[kind="tie_btn"]:hover {
    background: linear-gradient(to bottom, #4caf50, #28a745);
}
div.stButton > button[kind="undo_btn"] {
    background: linear-gradient(to bottom, #6c757d, #545b62);
    border-color: #545b62;
    color: white;
}
div.stButton > button[kind="undo_btn"]:hover {
    background: linear-gradient(to bottom, #8e959c, #6c757d);
}
@media (max-width: 600px) {
    div.stButton > button {
        width: 80%;
        max-width: 150px;
        height: 40px;
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Player", key="player_btn"):
        place_result("P")
with col2:
    if st.button("Banker", key="banker_btn"):
        place_result("B")
with col3:
    if st.button("Tie", key="tie_btn"):
        place_result("T")
with col4:
    if st.button("Undo Last", key="undo_btn"):
        if st.session_state.history and st.session_state.sequence:
            st.session_state.sequence.pop()
            last = st.session_state.history.pop()
            if last['Win']:
                st.session_state.wins -= 1
                st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                st.session_state.prediction_accuracy[last['Bet']] -= 1
                st.session_state.consecutive_losses = 0
                if st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins = max(0, st.session_state.parlay_wins - 1)
                    st.session_state.parlay_step = max(1, last['Parlay_Step'] - 1 if last['Parlay_Step'] > 1 else 1)
                    st.session_state.parlay_using_base = True
            else:
                st.session_state.bankroll += last['Amount']
                st.session_state.losses -= 1
                st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
                if st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins = 0
                    st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                    st.session_state.parlay_using_base = True
            st.session_state.prediction_accuracy['total'] -= 1
            if st.session_state.strategy == 'T3':
                st.session_state.t3_level = last['T3_Level']
                st.session_state.t3_results = last['T3_Results']
            else:
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."
            st.session_state.last_was_tie = False

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-90:] if 'sequence' in st.session_state else []
grid = [[] for _ in range(15)]
for i, result in enumerate(sequence):
    col_index = i // 6
    if col_index < 15:
        grid[col_index].append(result)
for col in grid:
    while len(col) < 6:
        col.append('')
bead_plate_html = "<div style='display: flex; flex-direction: row; gap: 5px; max-width: 100%; overflow-x: auto;'>"
for col in grid:
    col_html = "<div style='display: flex; flex-direction: column; gap: 5px;'>"
    for result in col:
        if result == '':
            col_html += "<div style='width: 20px; height: 20px; border: 1px solid #ddd; border-radius: 50%;'></div>"
        elif result == 'P':
            col_html += "<div style='width: 20px; height: 20px; background-color: blue; border-radius: 50%;'></div>"
        elif result == 'B':
            col_html += "<div style='width: 20px; height: 20px; background-color: red; border-radius: 50%;'></div>"
        elif result == 'T':
            col_html += "<div style='width: 20px; height: 20px; background-color: green; border-radius: 50%;'></div>"
    col_html += "</div>"
    bead_plate_html += col_html
bead_plate_html += "</div>"
st.markdown(bead_plate_html, unsafe_allow_html=True)

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
strategy_status = f"**Betting Strategy**: {st.session_state.strategy}"
if st.session_state.strategy == 'T3':
    strategy_status += f" | T3 Level: {st.session_state.t3_level} | Level Changes: {st.session_state.t3_level_changes}"
elif st.session_state.strategy == 'Parlay16':
    strategy_status += f" | Parlay Step: {st.session_state.parlay_step}/16 | Step Changes: {st.session_state.parlay_step_changes} | Consecutive Wins: {st.session_state.parlay_wins}"
st.markdown(strategy_status)
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
online_users = track_user_session_file()
st.markdown(f"**Online Users**: {online_users}")

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
            "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-",
            "Parlay_Step": h["Parlay_Step"] if st.session_state.strategy == 'Parlay16' else "-"
        }
        for h in st.session_state.history[-n:]
    ])
