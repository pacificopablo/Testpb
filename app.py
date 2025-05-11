import streamlit as st
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta
import numpy as np

# --- FILE-BASED SESSION TRACKING ---
SESSION_FILE = "online_users.txt"

def track_user_session_file():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())
    
    sessions = {}
    current_time = datetime.now()
    try:
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    session_id, timestamp = line.strip().split(',')
                    last_seen = datetime.fromisoformat(timestamp)
                    if current_time - last_seen <= timedelta(seconds=30):
                        sessions[session_id] = last_seen
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    except PermissionError:
        st.error("Unable to access session file. Online user count unavailable.")
        return 0
    
    sessions[st.session_state.session_id] = current_time
    
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except PermissionError:
        st.error("Unable to write to session file. Online user count may be inaccurate.")
        return 0
    
    return len(sessions)

# --- APP CONFIG ---
st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 0.0
    st.session_state.base_bet = 0.0
    st.session_state.initial_base_bet = 0.0
    st.session_state.sequence = []
    st.session_state.strategy = 'T3'
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.t3_level_changes = 0
    st.session_state.parlay_step = 1
    st.session_state.parlay_wins = 0
    st.session_state.parlay_using_base = True
    st.session_state.parlay_step_changes = 0
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
    st.session_state.recovery_mode = False
    st.session_state.insights = {}
    st.session_state.recovery_threshold = 15.0
    st.session_state.recovery_bet_scale = 0.6
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)

# Validate strategy
if 'strategy' in st.session_state and st.session_state.strategy not in ['T3', 'Flatbet', 'Parlay16']:
    st.session_state.strategy = 'T3'

# --- PARLAY TABLE ---
PARLAY_TABLE = {
    1: {'base': 1, 'parlay': 2},
    2: {'base': 1, 'parlay': 2},
    3: {'base': 1, 'parlay': 2},
    4: {'base': 2, 'parlay': 4},
    5: {'base': 3, 'parlay': 6},
    6: {'base': 4, 'parlay': 8},
    7: {'base': 6, 'parlay': 12},
    8: {'base': 8, 'parlay': 16},
    9: {'base': 12, 'parlay': 24},
    10: {'base': 16, 'parlay': 32},
    11: {'base': 22, 'parlay': 44},
    12: {'base': 30, 'parlay': 60},
    13: {'base': 40, 'parlay': 80},
    14: {'base': 52, 'parlay': 104},
    15: {'base': 70, 'parlay': 140},
    16: {'base': 95, 'parlay': 190}
}

# --- FUNCTIONS ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 3:
        return 'B', 45.86, {}  # Default with empty insights

    window_size = 50
    recent_sequence = sequence[-window_size:]

    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    pattern_transitions = defaultdict(lambda: defaultdict(int))
    streak_count = 0
    current_streak = None
    chop_count = 0
    double_count = 0
    insights = {}
    pattern_changes = 0
    last_pattern = None

    for i in range(len(recent_sequence) - 1):
        if i < len(recent_sequence) - 2:
            bigram = tuple(recent_sequence[i:i+2])
            next_outcome = recent_sequence[i+2]
            bigram_transitions[bigram][next_outcome] += 1

        if i < len(recent_sequence) - 3:
            trigram = tuple(recent_sequence[i:i+3])
            next_outcome = recent_sequence[i+3]
            trigram_transitions[trigram][next_outcome] += 1

        if i > 0:
            if recent_sequence[i] == recent_sequence[i-1]:
                if current_streak == recent_sequence[i]:
                    streak_count += 1
                else:
                    current_streak = recent_sequence[i]
                    streak_count = 1
                if i > 1 and recent_sequence[i-1] == recent_sequence[i-2]:
                    double_count += 1
            else:
                current_streak = None
                streak_count = 0
                if i > 1 and recent_sequence[i] != recent_sequence[i-2]:
                    chop_count += 1

        if i < len(recent_sequence) - 2:
            current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
            if last_pattern and last_pattern != current_pattern:
                pattern_changes += 1
            last_pattern = current_pattern
            next_outcome = recent_sequence[i+2]
            pattern_transitions[current_pattern][next_outcome] += 1

    st.session_state.pattern_volatility = pattern_changes / max(len(recent_sequence) - 2, 1)

    prior_p = 44.62 / 100
    prior_b = 45.86 / 100

    total_bets = max(st.session_state.pattern_attempts['bigram'], 1)
    weights = {
        'bigram': 0.4 * (st.session_state.pattern_success['bigram'] / total_bets if st.session_state.pattern_attempts['bigram'] > 0 else 0.5),
        'trigram': 0.3 * (st.session_state.pattern_success['trigram'] / total_bets if st.session_state.pattern_attempts['trigram'] > 0 else 0.5),
        'streak': 0.2 if streak_count >= 2 else 0.05,
        'chop': 0.05 if chop_count >= 2 else 0.01,
        'double': 0.05 if double_count >= 1 else 0.01
    }
    if sum(weights.values()) == 0:
        weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
    total_w = sum(weights.values())
    for k in weights:
        weights[k] = max(weights[k] / total_w, 0.05)

    prob_p = 0
    prob_b = 0
    total_weight = 0

    if len(recent_sequence) >= 2:
        bigram = tuple(recent_sequence[-2:])
        total_transitions = sum(bigram_transitions[bigram].values())
        if total_transitions > 0:
            p_prob = bigram_transitions[bigram]['P'] / total_transitions
            b_prob = bigram_transitions[bigram]['B'] / total_transitions
            prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total_transitions)
            prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total_transitions)
            total_weight += weights['bigram']
            insights['Bigram'] = f"{weights['bigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    if len(recent_sequence) >= 3:
        trigram = tuple(recent_sequence[-3:])
        total_transitions = sum(trigram_transitions[trigram].values())
        if total_transitions > 0:
            p_prob = trigram_transitions[trigram]['P'] / total_transitions
            b_prob = trigram_transitions[trigram]['B'] / total_transitions
            prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total_transitions)
            prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total_transitions)
            total_weight += weights['trigram']
            insights['Trigram'] = f"{weights['trigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"

    if chop_count >= 2:
        next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
        if next_pred == 'P':
            prob_p += weights['chop'] * 0.6
            prob_b += weights['chop'] * 0.4
        else:
            prob_b += weights['chop'] * 0.6
            prob_p += weights['chop'] * 0.4
        total_weight += weights['chop']
        insights['Chop'] = f"{weights['chop']*100:.0f}% ({chop_count} alternations)"

    if double_count >= 1 and len(recent_sequence) >= 2 and recent_sequence[-1] == recent_sequence[-2]:
        double_prob = 0.6
        if recent_sequence[-1] == 'P':
            prob_p += weights['double'] * double_prob
            prob_b += weights['double'] * (1 - double_prob)
        else:
            prob_b += weights['double'] * double_prob
            prob_p += weights['double'] * (1 - double_prob)
        total_weight += weights['double']
        insights['Double'] = f"{weights['double']*100:.0f}% ({recent_sequence[-1]}{recent_sequence[-1]})"

    if total_weight > 0:
        prob_p = (prob_p / total_weight) * 100
        prob_b = (prob_b / total_weight) * 100
    else:
        prob_p = 44.62
        prob_b = 45.86

    if abs(prob_p - prob_b) < 2:
        prob_p += 0.5
        prob_b -= 0.5

    current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
    total_transitions = sum(pattern_transitions[current_pattern].values())
    if total_transitions > 0:
        p_prob = pattern_transitions[current_pattern]['P'] / total_transitions
        b_prob = pattern_transitions[current_pattern]['B'] / total_transitions
        prob_p = 0.9 * prob_p + 0.1 * p_prob * 100
        prob_b = 0.9 * prob_b + 0.1 * b_prob * 100
        insights['Pattern Transition'] = f"10% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
    base_threshold = 50.0 if st.session_state.recovery_mode else 52.0
    threshold = base_threshold + (st.session_state.consecutive_losses * 1.0) - (recent_accuracy * 1.5)
    threshold = min(max(threshold, 48.0 if st.session_state.recovery_mode else 50.0), 60.0)
    insights['Threshold'] = f"{threshold:.1f}%"

    if st.session_state.pattern_volatility > 0.5:
        threshold += 2.0
        insights['Volatility'] = f"High (Adjustment: +2% threshold)"

    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights
    elif prob_b >= threshold:
        return 'B', prob_b, insights
    else:
        return None, max(prob_p, prob_b), insights

def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        return unit_profit >= st.session_state.target_value

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
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
    st.session_state.recovery_mode = False
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)

def log_result(result):
    if st.session_state.target_hit:
        reset_session_auto()
        return
    st.session_state.last_was_tie = (result == 'T')

    # Check recovery mode
    loss_percentage = (st.session_state.initial_bankroll - st.session_state.bankroll) / st.session_state.initial_bankroll if st.session_state.initial_bankroll > 0 else 0
    st.session_state.recovery_mode = loss_percentage >= st.session_state.recovery_threshold / 100

    # Store state
    previous_state = {
        "bankroll": st.session_state.bankroll,
        "t3_level": st.session_state.t3_level,
        "t3_results": st.session_state.t3_results.copy(),
        "parlay_step": st.session_state.parlay_step,
        "parlay_wins": st.session_state.parlay_wins,
        "parlay_using_base": st.session_state.parlay_using_base,
        "wins": st.session_state.wins,
        "losses": st.session_state.losses,
        "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
        "consecutive_losses": st.session_state.consecutive_losses,
        "t3_level_changes": st.session_state.t3_level_changes,
        "parlay_step_changes": st.session_state.parlay_step_changes,
        "recovery_mode": st.session_state.recovery_mode,
        "pattern_volatility": st.session_state.pattern_volatility,
        "pattern_success": st.session_state.pattern_success.copy(),
        "pattern_attempts": st.session_state.pattern_attempts.copy()
    }

    # Append to sequence
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Store history
    st.session_state.history.append({
        "Result": result,
        "Previous_State": previous_state
    })
    if len(st.session_state.history) > 1000:
        st.session_state.history = st.session_state.history[-1000:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    # Generate prediction
    pred, conf, insights = predict_next()

    if st.session_state.pattern_volatility > 0.5:
        st.session_state.advice = f"No prediction: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.insights = insights
        return

    if pred is None or conf < 48.0:
        st.session_state.advice = f"No prediction (Confidence: {conf:.1f}% too low)"
        st.session_state.insights = insights
    else:
        st.session_state.advice = f"Prediction: {pred} ({conf:.1f}%)"
        st.session_state.insights = insights

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet", "Parlay16"],
        index={'T3': 0, 'Flatbet': 1, 'Parlay16': 2}.get(st.session_state.strategy, 0),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression."
    )
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    recovery_threshold = st.slider("Recovery Mode Threshold (% Loss)", 10.0, 30.0, st.session_state.recovery_threshold, step=5.0)
    recovery_bet_scale = st.slider("Recovery Mode Bet Scaling", 0.5, 1.0, st.session_state.recovery_bet_scale, step=0.1)
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
        st.session_state.initial_base_bet = base_bet
        st.session_state.strategy = betting_strategy
        st.session_state.sequence = []
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.t3_level_changes = 0
        st.session_state.parlay_step = 1
        st.session_state.parlay_wins = 0
        st.session_state.parlay_using_base = True
        st.session_state.parlay_step_changes = 0
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
        st.session_state.recovery_mode = False
        st.session_state.insights = {}
        st.session_state.recovery_threshold = recovery_threshold
        st.session_state.recovery_bet_scale = recovery_bet_scale
        st.session_state.pattern_volatility = 0.0
        st.session_state.pattern_success = defaultdict(int)
        st.session_state.pattern_attempts = defaultdict(int)
        st.success(f"Session started with {betting_strategy} strategy!")

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
if not st.session_state.target_hit:
    st.info(st.session_state.advice)

# --- PREDICTION INSIGHTS ---
st.subheader("Prediction Insights")
if st.session_state.insights:
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"**{factor}**: {contribution}")
if st.session_state.recovery_mode:
    st.warning("Recovery Mode: Significant losses detected.")
if st.session_state.pattern_volatility > 0.5:
    st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Predictions paused)")

# --- UNIT PROFIT ---
if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    units_profit = profit / st.session_state.initial_base_bet
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
    st.markdown(f"**Player Predictions**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
    st.markdown(f"**Banker Predictions**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

# --- HISTORY TABLE ---
if st.session_state.history:
    st.subheader("History")
    n = st.slider("Show last N results", 5, 50, 10)
    st.dataframe([
        {
            "Result": h["Result"]
        }
        for h in st.session_state.history[-n:]
    ])

# --- EXPORT SESSION ---
st.subheader("Export Session")
if st.button("Download Session Data"):
    csv_data = "Result\n"
    for h in st.session_state.history:
        csv_data += f"{h['Result']}\n"
    st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")
