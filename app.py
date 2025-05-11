def predict_next():
    sequence = st.session_state.sequence
    base_bet = st.session_state.base_bet
    t3_level = st.session_state.t3_level
    strategy = st.session_state.betting_strategy
    flat_bet_amount = st.session_state.flat_bet_amount
    # Define default probabilities (normalized P and B probabilities)
    total_p_b = 0.4462 + 0.4586  # Fixed: Correct variable name
    default_p = 0.4462 / total_p_b  # ~0.4931
    default_b = 0.4586 / total_p_b  # ~0.5069

    if len(sequence) < 2:
        insights = {
            "Overall": f"No prediction: Need at least 2 outcomes (Current: {len(sequence)})",
            "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
        }
        return None, 0, insights, 0.0
    elif len(sequence) < 3:
        insights = {
            "Overall": f"No prediction: Need at least 3 outcomes for trigram (Current: {len(sequence)})",
            "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
        }
        return None, 0, insights, 0.0

    window_size = 50
    recent_sequence = sequence[-window_size:]

    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    pattern_transitions = defaultdict(lambda: defaultdict(int))
    streak_count = 0
    current_streak = None
    chop_count = 0
    double_count = 0
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

    bigram = tuple(recent_sequence[-2:])
    total_transitions = sum(bigram_transitions[bigram].values())
    if total_transitions > 0:
        bigram_p_prob = bigram_transitions[bigram]['P'] / total_transitions
        bigram_b_prob = bigram_transitions[bigram]['B'] / total_transitions
    else:
        bigram_p_prob = default_p
        bigram_b_prob = default_b
    bigram_pred = 'P' if bigram_p_prob > bigram_b_prob else 'B'

    trigram = tuple(recent_sequence[-3:])
    total_transitions = sum(trigram_transitions[trigram].values())
    if total_transitions > 0:
        trigram_p_prob = trigram_transitions[trigram]['P'] / total_transitions
        trigram_b_prob = trigram_transitions[trigram]['B'] / total_transitions
    else:
        trigram_p_prob = default_p
        trigram_b_prob = default_b
    trigram_pred = 'P' if trigram_p_prob > trigram_b_prob else 'B'

    if bigram_pred == trigram_pred:
        pred = bigram_pred
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
        conf = max(overall_p, overall_b) * 100
        if strategy == "T3":
            bet_amount = base_bet * t3_level
            bet_info = f"T3: Bet ${bet_amount:.2f} (Level {t3_level})"
        else:
            bet_amount = flat_bet_amount
            bet_info = f"FlatBet: ${bet_amount:.2f}"
    else:
        pred = None
        overall_p = (bigram_p_prob + trigram_p_prob) / 2
        overall_b = (bigram_b_prob + trigram_b_prob) / 2
        conf = max(overall_p, overall_b) * 100
        bet_amount = 0.0
        bet_info = f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else "")

    insights = {
        'Overall': f"P: {overall_p*100:.1f}%, B: {overall_b*100:.1f}%",
        'Volatility': f"{st.session_state.pattern_volatility:.2f}",
        'Betting Strategy': bet_info,
    }
    if strategy == "T3":
        if len(st.session_state.bet_history) > 0:
            wins = sum(1 for pred, actual, _ in st.session_state.bet_history[-3:] if pred == actual)
            total_bets = min(3, len(st.session_state.bet_history))
            losses = total_bets - wins
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            insights['Current T3 Cycle'] = f"W: {wins}, L: {losses}"
            insights['Bet History'] = f"Last 3 Bets: Win Rate: {win_rate:.1f}% ({wins}/{total_bets})"
        else:
            insights['Current T3 Cycle'] = "W: 0, L: 0"
    if pred is None:
        insights['Status'] = "No prediction: Bigram and trigram predictions differ"

    return pred, conf, insights, bet_amount
