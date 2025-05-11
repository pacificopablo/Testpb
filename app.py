def predict_next():
    sequence = st.session_state.sequence  # Contains only P, B
    # Define default probabilities (normalized P and B probabilities)
    total_p_b = 0.4462 + 0.4586
    default_p = 0.4462 / total_p_b  # ~0.4931
    default_b = 0.4586 / total_p_b  # ~0.5069

    if len(sequence) < 2:
        return None, max(default_p, default_b) * 100, {"Overall": f"P: {default_p*100:.1f}%, B: {default_b*100:.1f}%"}

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

    # Compute trigram probabilities for the last three outcomes (if available)
    if len(recent_sequence) >= 3:
        trigram = tuple(recent_sequence[-3:])
        total_transitions = sum(trigram_transitions[trigram].values())
        if total_transitions > 0:
            trigram_p_prob = trigram_transitions[trigram]['P'] / total_transitions
            trigram_b_prob = trigram_transitions[trigram]['B'] / total_transitions
        else:
            trigram_p_prob = default_p
            trigram_b_prob = default_b
        # Combine bigram and trigram probabilities
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
    else:
        # Use only bigram probabilities
        overall_p = bigram_p_prob
        overall_b = bigram_b_prob

    # Determine prediction with threshold
    threshold = 52.0
    if st.session_state.pattern_volatility > 0.5:
        threshold += 2.0
    max_prob = max(overall_p, overall_b)
    conf = max_prob * 100
    if max_prob >= threshold / 100:
        if overall_p > overall_b:
            pred = 'P'
        else:
            pred = 'B'
    else:
        pred = None

    # Insights
    insights = {
        'Bigram': f"P: {bigram_p_prob*100:.1f}%, B: {bigram_b_prob*100:.1f}%",
    }
    if len(recent_sequence) >= 3:
        insights['Trigram'] = f"P: {trigram_p_prob*100:.1f}%, B: {trigram_b_prob*100:.1f}%"
    insights['Overall'] = f"P: {overall_p*100:.1f}%, B: {overall_b*100:.1f}%"
    insights['Volatility'] = f"{st.session_state.pattern_volatility:.2f}"
    insights['Threshold'] = f"{threshold:.1f}%"
    if pred is None:
        insights['Reason'] = "Confidence below threshold"

    return pred, conf, insights
