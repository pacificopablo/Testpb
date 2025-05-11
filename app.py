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
    st.session_state.flat_bet_amount = 10.0  # Default flat bet amount

# --- PREDICTION FUNCTION ---
def predict_next():
    sequence = st.session_state.sequence  # Contains only P, B
    flat_bet = st.session_state.flat_bet_amount
    # Define default probabilities (normalized P and B probabilities)
    total_p_b = 0.4462 + 0.4586
    default_p = 0.4462 / total_p_b  # ~0.4931
    default_b = 0.4586 / total_p_b  # ~0.5069

    if len(sequence) < 2:
        insights = {
            "Overall": f"No prediction: Need at least 2 outcomes (Current: {len(sequence)})",
            "Betting Strategy": "No bet",
        }
        return None, 0, insights
    elif len(sequence) < 3:
        insights = {
            "Overall": f"No prediction: Need at least 3 outcomes for trigram (Current: {len(sequence)})",
            "Betting Strategy": "No bet",
        }
        return None, 0, insights

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
        bet_info = f"Flat bet: ${flat_bet:.2f} on {pred}"
    else:
        pred = None
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
        conf = max(overall_p, overall_b) * 100
        bet_info = "No bet"

    # Insights
    insights = {
        'Bigram': f"Prediction: {bigram_pred}, P: {bigram_p_prob*100:.1f}%, B: {bigram_b_prob*100:.1f}%",
        'Trigram': f"Prediction: {trigram_pred}, P: {trigram_p_prob*100:.1f}%, B: {trigram_b_prob*100:.1f}%",
        'Overall': f"P: {overall_p*100:.1f}%, B: {overall_b*100:.1f}%",
        'Volatility': f"{st.session_state.pattern_volatility:.2f}",
        'Betting Strategy': bet_info,
    }
    if pred is None:
        insights['Status'] = "No prediction: Bigram and trigram predictions differ"

    return pred, conf, insights

# --- PROCESS RESULT ---
def place_result(result):
    # Append to sequence (only P or B)
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Calculate next prediction
    pred, conf, insights = predict_next()

    if pred is None:
        st.session_state.pending_prediction = None
        st.session_state.advice = f"No prediction: Bigram and trigram predictions differ, No bet"
        if st.session_state.pattern_volatility > 0.5:
            st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
    else:
        st.session_state.pending_prediction = pred
        st.session_state.advice = f"Prediction: {pred} ({conf:.1f}%), Bet: ${st.session_state.flat_bet_amount:.2f}"
        if st.session_state.pattern_volatility > 0.5:
            st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
    st.session_state.insights = insights

# --- UI ---
st.title("BACCARAT PLAYER/BANKER PREDICTOR")

# Result Input and Bet Amount
st.subheader("Enter Game Result")
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
    bet = advice_parts[1].split(': ')[1] if len(advice_parts) > 1 and 'Bet' in advice_parts[1] else 'No bet'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Prob: {prob}% | Bet: {bet}</h4>", unsafe_allow_html=True)
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
