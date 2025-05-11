import streamlit as st
from collections import defaultdict

# --- SESSION STATE INIT ---
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
    st.session_state.pending_prediction = None
    st.session_state.advice = ""
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0

# --- PREDICTION FUNCTION ---
def predict_next():
    sequence = st.session_state.sequence  # Contains only P, B
    if len(sequence) < 3:
        return 'B', 45.86, {}

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
    insights = {}
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

    # Bayesian priors
    prior_p = 44.62 / 100
    prior_b = 45.86 / 100

    # Fixed weights
    weights = {
        'bigram': 0.4,
        'trigram': 0.3,
        'streak': 0.2 if streak_count >= 2 else 0.05,
        'chop': 0.05 if chop_count >= 2 else 0.01,
        'double': 0.05 if double_count >= 1 else 0.01
    }
    total_w = sum(weights.values())
    for k in weights:
        weights[k] = max(weights[k] / total_w, 0.05)  # Ensure non-zero weights

    # Calculate probabilities
    prob_p = 0
    prob_b = 0
    total_weight = 0

    # Bigram contribution
    if len(recent_sequence) >= 2:
        bigram = tuple(recent_sequence[-2:])
        total_transitions = sum(bigram_transitions[bigram].values())
        if total_transitions > 0:
            p_prob = bigram_transitions[bigram]['P'] / total_transitions
            b_prob = bigram_transitions[bigram]['B'] / total_transitions
            prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total_transitions)
            prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total_transitions)  # Corrected
            total_weight += weights['bigram']
            insights['Bigram'] = f"{weights['bigram']*100:.0f}% weight (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Trigram contribution
    if len(recent_sequence) >= 3:
        trigram = tuple(recent_sequence[-3:])
        total_transitions = sum(trigram_transitions[trigram].values())
        if total_transitions > 0:
            p_prob = trigram_transitions[trigram]['P'] / total_transitions
            b_prob = trigram_transitions[trigram]['B'] / total_transitions
            prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total_transitions)
            prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total_transitions)
            total_weight += weights['trigram']
            insights['Trigram'] = f"{weights['trigram']*100:.0f}% weight (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Streak contribution with anti-streak bias
    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% weight ({streak_count} {current_streak})"

    # Chop contribution
    if chop_count >= 2:
        next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
        if next_pred == 'P':
            prob_p += weights['chop'] * 0.6
            prob_b += weights['chop'] * 0.4
        else:
            prob_b += weights['chop'] * 0.6
            prob_p += weights['chop'] * 0.4
        total_weight += weights['chop']
        insights['Chop'] = f"{weights['chop']*100:.0f}% weight ({chop_count} alternations)"

    # Double contribution
    if double_count >= 1 and len(recent_sequence) >= 2 and recent_sequence[-1] == recent_sequence[-2]:
        double_prob = 0.6
        if recent_sequence[-1] == 'P':
            prob_p += weights['double'] * double_prob
            prob_b += weights['double'] * (1 - double_prob)
        else:
            prob_b += weights['double'] * double_prob
            prob_p += weights['double'] * (1 - double_prob)
        total_weight += weights['double']
        insights['Double'] = f"{weights['double']*100:.0f}% weight ({recent_sequence[-1]}{recent_sequence[-1]})"

    # Normalize probabilities
    if total_weight > 0:
        prob_p = (prob_p / total_weight) * 100
        prob_b = (prob_b / total_weight) * 100
    else:
        prob_p = 44.62
        prob_b = 45.86

    # Adjust for Banker commission
    if abs(prob_p - prob_b) < 2:
        prob_p += 0.5
        prob_b -= 0.5

    # Pattern transition adjustment
    current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
    total_transitions = sum(pattern_transitions[current_pattern].values())
    if total_transitions > 0:
        p_prob = pattern_transitions[current_pattern]['P'] / total_transitions
        b_prob = pattern_transitions[current_pattern]['B'] / total_transitions
        prob_p = 0.9 * prob_p + 0.1 * p_prob * 100
        prob_b = 0.9 * prob_b + 0.1 * b_prob * 100
        insights['Pattern Transition'] = f"10% weight (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Confidence threshold
    threshold = 52.0
    insights['Confidence Threshold'] = f"{threshold:.1f}%"

    # Volatility adjustment
    if st.session_state.pattern_volatility > 0.5:
        threshold += 2.0
        insights['Volatility'] = f"High (Adjustment: +2% threshold)"

    # Determine prediction
    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights
    elif prob_b >= threshold:
        return 'B', prob_b, insights
    else:
        return None, max(prob_p, prob_b), insights

# --- PROCESS RESULT ---
def place_result(result):
    # Append to sequence (only P or B)
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Calculate next prediction
    pred, conf, insights = predict_next()

    if pred is None or conf < 48.0:
        st.session_state.pending_prediction = None
        st.session_state.advice = f"No prediction (Confidence: {conf:.1f}% too low)"
        st.session_state.insights = insights
    else:
        st.session_state.pending_prediction = pred
        st.session_state.advice = f"Prediction: {pred} ({conf:.1f}%)"
        st.session_state.insights = insights

    # Volatility check
    if st.session_state.pattern_volatility > 0.5:
        st.session_state.advice = f"No prediction: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.pending_prediction = None
        st.session_state.insights = insights

# --- UI ---
st.title("BACCARAT PLAYER/BANKER PREDICTOR")

# Result Input
st.subheader("Enter Game Result")
col1, col2 = st.columns(2)
with col1:
    if st.button("Player"):
        place_result("P")
with col2:
    if st.button("Banker"):
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
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
else:
    st.info(st.session_state.advice)

# Prediction Insights
st.subheader("Prediction Insights")
if st.session_state.insights:
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"- **{factor}**: {contribution}")
    if st.session_state.pattern_volatility > 0.5:
        st.warning(f"**High Pattern Volatility**: {st.session_state.pattern_volatility:.2f} (Predicting paused)")
else:
    st.markdown("No insights available yet. Enter at least 3 Player or Banker results to generate predictions.")
