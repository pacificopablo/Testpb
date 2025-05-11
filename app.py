import streamlit as st
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta
import numpy as np

def calculate_stake(confidence):
    strategy = st.session_state.get("management_strategy", "Flat")
    bankroll = st.session_state.get("bankroll", 1000)
    base_bet = st.session_state.get("base_bet", 10)

    if strategy == "Flat":
        return base_bet

    elif strategy == "Kelly %":
        edge = (confidence / 100) - 0.5  # Assumes breakeven at 50%
        kelly_fraction = edge / 0.5  # Based on fair odds
        stake = bankroll * kelly_fraction
        return max(base_bet, round(stake, 2))

    elif strategy == "Progressive":
        losses = st.session_state.get("loss_streak", 0)
        stake = base_bet + losses * base_bet
        return min(stake, bankroll)

    return base_bet


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
if 'sequence' not in st.session_state:
    st.session_state.sequence = []  # Start with empty sequence
    st.session_state.advice = ""
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)
    st.session_state.bankroll = 1000.0
    st.session_state.base_bet = 10.0
    st.session_state.loss_streak = 0
    st.session_state.target_mode = "Percentage"  # Default initialization
    st.session_state.target_value = 10.0  # Default initialization

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

    
    # Highlight strongest factor used for prediction
    strongest_factor = max(weights.items(), key=lambda x: x[1])
    insights['Top Factor'] = f"Bet driven by: {strongest_factor[0].capitalize()} Pattern (Weight: {strongest_factor[1]*100:.1f}%)"
    insights['Top Factor Raw'] = strongest_factor[0].capitalize() + " Pattern"  # for advice message

    threshold = 50.0
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

def log_result(result):
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    pred, conf, insights = predict_next()
    st.session_state.insights = insights

    stake = calculate_stake(conf)

    # Track loss streak for Progressive strategy and update bankroll
    if pred and result != pred:
        st.session_state.loss_streak = st.session_state.get("loss_streak", 0) + 1
        st.session_state.bankroll = max(st.session_state.bankroll - stake, 0)
    else:
        st.session_state.loss_streak = 0
        # Assumes winning returns the stake amount (can adjust multiplier if needed)
        st.session_state.bankroll += stake

    if st.session_state.pattern_volatility > 0.5:
        st.session_state.advice = f"No prediction: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
    elif pred is None or conf < 48.0:
        st.session_state.advice = f"No prediction (Confidence: {conf:.1f}% too low)"
    else:
        top_factor = st.session_state.insights.get('Top Factor Raw', 'Mixed Factors')
        st.session_state.advice = (
            f"Recommended Bet: {pred} (${stake:.2f} stake at {conf:.1f}% confidence)\n"
            f"Reason: Driven by {top_factor} — based on recent pattern behavior."
        )

# --- SETUP FORM ---
st.subheader("Setup")

with st.form("setup_form"):
    bankroll = st.number_input("Initial Bankroll ($)", min_value=1.0, value=1000.0, step=10.0)
    base_bet = st.number_input("Base Bet Unit ($)", min_value=1.0, value=10.0, step=1.0)

    col1, col2 = st.columns([1, 2])
    with col1:
        target_mode = st.radio("Target Mode", ["Percentage", "Units"], horizontal=True)
    management_strategy = st.selectbox("Money Management Strategy", ["Flat", "Kelly %", "Progressive"])
    with col2:
        if target_mode == "Percentage":
            target_value = st.number_input("Target Profit (%)", min_value=1.0, value=10.0, step=1.0)
        else:
            target_value = st.number_input("Target Profit (Units)", min_value=1.0, value=10.0, step=1.0)

    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    st.session_state.sequence = []
    st.session_state.advice = ""
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)
    st.session_state.loss_streak = 0

    st.session_state.bankroll = bankroll
    st.session_state.base_bet = base_bet
    st.session_state.target_mode = target_mode
    st.session_state.target_value = target_value
    st.session_state.management_strategy = management_strategy

    st.success("Session started!")

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

col1, col2 = st.columns(2)
with col1:
    if st.button("Player", key="player_btn"):
        log_result("P")
        # Some Streamlit versions support st.experimental_rerun() for rerun; if not, fallback to st.rerun()
        try:
            st.experimental_rerun()
        except AttributeError:
            st.rerun()
with col2:
    if st.button("Banker", key="banker_btn"):
        log_result("B")
        try:
            st.experimental_rerun()
        except AttributeError:
            st.rerun()

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-90:] if 'sequence' in st.session_state else []
if not sequence:
    st.info("No sequence available. Enter results using the Player or Banker buttons.")
else:
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
        col_html += "</div>"
        bead_plate_html += col_html
    bead_plate_html += "</div>"
    st.markdown(bead_plate_html, unsafe_allow_html=True)

# --- PREDICTION DISPLAY ---
st.subheader("Prediction")
st.info(st.session_state.advice)

# --- PREDICTION INSIGHTS ---
st.subheader("Prediction Insights")
if st.session_state.insights:
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"**{factor}**: {contribution}")
if st.session_state.pattern_volatility > 0.5:
    st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Predictions paused)")

# --- STATUS ---
st.subheader("Status")
online_users = track_user_session_file()
st.markdown(f"**Online Users**: {online_users}")
st.markdown(f"**Bankroll**: ${st.session_state.get('bankroll', 0):,.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.get('base_bet', 0):,.2f}")
if st.session_state.get("target_mode") == "Percentage":
    st.markdown(f"**Target**: {st.session_state['target_value']}%")
else:
    st.markdown(f"**Target**: {st.session_state['target_value']} units")

