import streamlit as st
from collections import defaultdict

# --- APP CONFIG ---
st.set_page_config(layout="centered", page_title="BACCARAT PLAYER/BANKER PREDICTOR")

# --- SESSION STATE INIT ---
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
    st.session_state.pending_prediction = None
    st.session_state.advice = ""
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.base_bet = 10.0  # Base bet for both strategies
    st.session_state.flat_bet_amount = 10.0  # Fixed bet for FlatBet
    st.session_state.t3_level = 1  # Current level for T3 (no upper limit)
    st.session_state.bet_history = []  # List of (prediction, actual_outcome, bet_amount) for T3
    st.session_state.betting_strategy = "T3"  # Default strategy

# --- PREDICTION FUNCTION ---
def predict_next():
    sequence = st.session_state.sequence  # Contains only P, B
    base_bet = st.session_state.base_bet
    t3_level = st.session_state.t3_level
    strategy = st.session_state.betting_strategy
    flat_bet_amount = st.session_state.flat_bet_amount
    # Define default probabilities (normalized P and B probabilities)
    total_p_b = 0.4462 + 0.4586
    default_p = 0.4462 / total_p_b  # ~0.4931
    default_b = 0.4586 / total_p_b  # ~0.5069

    if len(sequence) < 2:
        insights = {
            "Overall": f"No prediction: Need at least 2 outcomes (Current: {len(sequence)})",
            "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
        }
        return None, 0, insights, 0.0  # Return four values with default bet_amount
    elif len(sequence) < 3:
        insights = {
            "Overall": f"No prediction: Need at least 3 outcomes for trigram (Current: {len(sequence)})",
            "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
        }
        return None, 0, insights, 0.0  # Return four values with default bet_amount

    # Sliding window of 50 hands
    window_size = 50
    recent_sequence = sequence[-window_size:]

    # Initialize data structures
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    pattern_transitions = defaultdict(lambda: defaultdict(int))
    streak_count = 0
    current_streak = None
    chop_count = 0
    double_count = 0
    pattern_changes = 0
    last_pattern = None

    # Analyze patterns
    for i in range(len(recent_sequence) - 1):
        # Bigram transitions
        if i < len(recent_sequence) - 2:
            bigram = tuple(recent_sequence[i:i+2])
            next_outcome = recent_sequence[i+2]
            bigram_transitions[bigram][next_outcome] += 1

        # Trigram transitions
        if i < len(recent_sequence) - 3:
            trigram = tuple(recent_sequence[i:i+3])
            next_outcome = recent_sequence[i+3]
            trigram_transitions[trigram][next_outcome] += 1

        # Pattern detection
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

        # Pattern transitions and volatility
        if i < len(recent_sequence) - 2:
            current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
            if last_pattern and last_pattern != current_pattern:
                pattern_changes += 1
            last_pattern = current_pattern
            next_outcome = recent_sequence[i+2]
            pattern_transitions[current_pattern][next_outcome] += 1

    # Calculate volatility (pattern changes per hand)
    st.session_state.pattern_volatility = pattern_changes / max(len(recent_sequence) - 2, 1)

    # Compute bigram probabilities for the last two outcomes
    bigram = tuple(recent_sequence[-2:])
    total_transitions = sum(bigram_transitions[bigram].values())
    if total_transitions > 0:
        bigram_p_prob = bigram_transitions[bigram]['P'] / total_transitions
        bigram_b_prob = bigram_transitions[bigram]['B'] / total_transitions
    else:
        bigram_p_prob = default_p
        bigram_b_prob = default_b
    bigram_pred = 'P' if bigram_p_prob > bigram_b_prob else 'B'

    # Compute trigram probabilities for the last three outcomes
    trigram = tuple(recent_sequence[-3:])
    total_transitions = sum(trigram_transitions[trigram].values())
    if total_transitions > 0:
        trigram_p_prob = trigram_transitions[trigram]['P'] / total_transitions
        trigram_b_prob = trigram_transitions[trigram]['B'] / total_transitions
    else:
        trigram_p_prob = default_p
        trigram_b_prob = default_b
    trigram_pred = 'P' if trigram_p_prob > trigram_b_prob else 'B'

    # Combine probabilities only if predictions agree
    if bigram_pred == trigram_pred:
        pred = bigram_pred
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
        conf = max(overall_p, overall_b) * 100
        if strategy == "T3":
            bet_amount = base_bet * t3_level
            bet_info = f"T3: Bet ${bet_amount:.2f} (Level {t3_level})"
        else:  # FlatBet
            bet_amount = flat_bet_amount
            bet_info = f"FlatBet: ${bet_amount:.2f}"
    else:
        pred = None
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
        conf = max(overall_p, overall_b) * 100
        bet_amount = 0.0
        bet_info = f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else "")

    # Insights
    insights = {
        'Bigram': f"Prediction: {bigram_pred}, P: {bigram_p_prob*100:.1f}%, B: {bigram_b_prob*100:.1f}%",
        'Trigram': f"Prediction: {trigram_pred}, P: {trigram_p_prob*100:.1f}%, B: {trigram_b_prob*100:.1f}%",
        'Overall': f"P: {overall_p*100:.1f}%, B: {overall_b*100:.1f}%",
        'Volatility': f"{st.session_state.pattern_volatility:.2f}",
        'Betting Strategy': bet_info,
    }
    if strategy == "T3" and len(st.session_state.bet_history) > 0:
        wins = sum(1 for pred, actual, _ in st.session_state.bet_history[-3:] if pred == actual)
        total_bets = min(3, len(st.session_state.bet_history))
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        insights['Bet History'] = f"Last 3 Bets: Win Rate: {win_rate:.1f}% ({wins}/{total_bets})"
    if pred is None:
        insights['Status'] = "No prediction: Bigram and trigram predictions differ"

    return pred, conf, insights, bet_amount

# --- PROCESS RESULT ---
def place_result(result):
    # Append to sequence (only P or B)
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Calculate next prediction
    pred, conf, insights, bet_amount = predict_next()

    # Update advice and history
    if pred is None:
        st.session_state.pending_prediction = None
        st.session_state.advice = f"No prediction: Bigram and trigram predictions differ, {insights['Betting Strategy']}"
        if st.session_state.pattern_volatility > 0.5:
            st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
    else:
        st.session_state.pending_prediction = pred
        st.session_state.advice = f"Prediction: {pred} ({conf:.1f}%), {insights['Betting Strategy']}"
        if st.session_state.pattern_volatility > 0.5:
            st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        # Update bet history based on strategy
        if st.session_state.betting_strategy == "T3":
            st.session_state.bet_history.append((pred, result, bet_amount))
            # Evaluate T3 level change if 3 bets are completed
            if len(st.session_state.bet_history) >= 3:
                wins = sum(1 for p, a, _ in st.session_state.bet_history[-3:] if p == a)
                losses = 3 - wins
                if wins == 3:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 2)  # 3 Wins: Go back 2 levels
                elif wins == 2 and losses == 1:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)  # 2 Wins, 1 Loss: Go back 1 level
                elif wins == 1 and losses == 2:
                    st.session_state.t3_level += 1  # 2 Losses, 1 Win: Go to next level
                elif losses == 3:
                    st.session_state.t3_level += 2  # 3 Losses: Go forward 2 levels
                st.session_state.bet_history = []  # Reset for next level
    st.session_state.insights = insights

# --- UI ---
st.title("BACCARAT PLAYER/BANKER PREDICTOR")

# Result Input and Betting Strategy Selection
st.subheader("Enter Game Result")
st.session_state.betting_strategy = st.selectbox(
    "Select Betting Strategy", ["T3", "FlatBet"], index=["T3", "FlatBet"].index(st.session_state.betting_strategy)
)
st.session_state.base_bet = st.number_input(
    "Base Bet Amount ($)", min_value=0.01, value=st.session_state.base_bet, step=1.0, format="%.2f"
)
if st.session_state.betting_strategy == "FlatBet":
    st.session_state.flat_bet_amount = st.number_input(
        "Flat Bet Amount ($)", min_value=0.01, value=st.session_state.flat_bet_amount, step=1.0, format="%.2f"
    )

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
        place_result("P")
with col2:
    if st.button("Banker", key="banker_btn"):
        place_result("B")

# Bead Plate
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-90:]  # Limit to 90 for display
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

# Prediction Display
if st.session_state.pending_prediction:
    side = st.session_state.pending_prediction
    color = 'blue' if side == 'P' else 'red'
    advice_parts = st.session_state.advice.split(', ')
    prob = advice_parts[0].split('(')[-1].split('%')[0] if '(' in advice_parts[0] else '0'
    bet_info = advice_parts[1] if len(advice_parts) > 1 else 'No bet'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Prob: {prob}% | {bet_info}</h4>", unsafe_allow_html=True)
else:
    st.info(st.session_state.advice)

# Prediction Insights
st.subheader("Prediction Insights")
if st.session_state.insights:
    st.markdown("**Factors Contributing to Prediction:**")
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"- **{factor}**: {contribution}")
    if st.session_state.pattern_volatility > 0.5:
        st.warning(f"**High Pattern Volatility**: {st.session_state.pattern_volatility:.2f}")
else:
    st.markdown("No insights available yet. Enter at least 3 Player or Banker results to generate predictions.")
