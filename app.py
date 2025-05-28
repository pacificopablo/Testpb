import streamlit as st

# Normalize input tokens
def normalize_result(s):
    s = s.strip().lower()
    if s == 'banker' or s == 'b':
        return 'Banker'
    if s == 'player' or s == 'p':
        return 'Player'
    if s == 'tie' or s == 't':
        return 'Tie'
    return None

def detect_streak(results):
    if not results:
        return None, 0
    last = results[-1]
    count = 1
    for i in range(len(results) - 2, -1, -1):
        if results[i] == last:
            count += 1
        else:
            break
    return last, count

def is_alternating_pattern(arr):
    if len(arr) < 4:
        return False
    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1]:
            return False
    return True

def recent_trend_analysis(results, window=10):
    """
    Analyze recent trends within a specified window to detect dominance of Banker or Player.
    Returns a score favoring the dominant side or None if balanced.
    """
    recent = results[-window:] if len(results) >= window else results
    if not recent:
        return None, 0
    freq = frequency_count(recent)
    total = len(recent)
    if total == 0:
        return None, 0
    banker_ratio = freq['Banker'] / total
    player_ratio = freq['Player'] / total
    if banker_ratio > player_ratio + 0.2:  # Banker dominates
        return 'Banker', banker_ratio * 50
    elif player_ratio > banker_ratio + 0.2:  # Player dominates
        return 'Player', player_ratio * 50
    return None, 0

def frequency_count(arr):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in arr:
        if r in count:
            count[r] += 1
    return count

def advanced_bet_selection(results):
    """
    Emotion-driven bet selection logic with enhanced heuristics:
    - Streak detection with dynamic confidence
    - Alternating pattern detection
    - Short-term trend analysis (last 10 hands)
    - Long-term frequency analysis
    - Emotion influences betting: skip bets if Cautious/Hesitant
    Returns bet choice (or 'Pass'), confidence, reason, and emotional tone.
    """
    max_recent_count = 30
    recent = results[-max_recent_count:]
    if not recent:
        return 'Pass', 0, "I need some past results to make a bet. Let’s wait for now!", "Cautious"

    streak_value, streak_length = detect_streak(recent)
    freq = frequency_count(recent)
    trend_bet, trend_score = recent_trend_analysis(recent)

    # Initialize scores
    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    emotional_tone = "Neutral"
    confidence = 0

    # Streak detection (highest priority)
    if streak_length >= 3 and streak_value != "Tie":
        confidence = min(75 + (streak_length - 3) * 10, 95)
        scores[streak_value] += confidence
        reason_parts.append(f"Strong streak detected: {streak_length} {streak_value} wins in a row! I’m super confident about this one!")
        emotional_tone = "Very Confident" if confidence > 85 else "Confident"

    # Alternating pattern detection
    elif len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        confidence = 70
        scores[alternate_bet] += confidence
        reason_parts.append(f"Alternating pattern in the last 4 hands. I’m betting on {alternate_bet}! This is exciting!")
        emotional_tone = "Excited"

    # Short-term trend analysis
    elif trend_bet:
        confidence = round(trend_score)
        scores[trend_bet] += confidence
        reason_parts.append(f"In the last 10 hands, {trend_bet} is dominating. This looks like a good bet!")
        emotional_tone = "Hopeful"

    # Long-term frequency analysis as fallback
    else:
        total = len(recent)
        scores['Banker'] += (freq['Banker'] / total * 0.9) * 50  # Adjust for commission
        scores['Player'] += (freq['Player'] / total * 1.0) * 50
        scores['Tie'] += (freq['Tie'] / total * 0.5) * 50  # Lower weight for Tie
        reason_parts.append(
            f"Based on {total} hands: Banker {freq['Banker']} ({scores['Banker']:.1f} points), "
            f"Player {freq['Player']} ({scores['Player']:.1f} points), "
            f"Tie {freq['Tie']} ({scores['Tie']:.1f} points). No clear pattern, so I’m playing it safe."
        )
        confidence = round(max(scores['Banker'], scores['Player'], scores['Tie']))

    # Determine best choice
    bet_choice = max(scores, key=scores.get)

    # Emotion-driven decision: skip bet if Cautious/Hesitant
    if confidence < 50:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append("My confidence is too low. I’m passing to stay safe!")
    elif confidence < 60 and emotional_tone == "Neutral":
        emotional_tone = "Cautious"
        reason_parts.append("I’m a bit unsure, but let’s try this bet anyway.")

    # Avoid Tie unless very confident
    if bet_choice == 'Tie' and confidence < 80:
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice]), 95)
        reason_parts.append("Ties are too risky unless I’m really sure. Switching to a safer bet.")
        emotional_tone = "Cautious" if emotional_tone != "Hesitant" else "Hesitant"

    # Adjust confidence for conflicting signals
    if streak_length >= 3 and is_alternating_pattern(recent[-4:]):
        confidence = max(confidence - 10, 40)
        reason_parts.append("There’s a conflict between a streak and an alternating pattern, so I’m lowering my confidence.")
        emotional_tone = "Cautious" if emotional_tone != "Hesitant" else "Hesitant"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone

def money_management(bankroll, base_bet, strategy, confidence=None):
    """
    Calculate bet size based on selected money management strategy.
    - Fixed 5% of Bankroll: 5% of current bankroll, rounded to multiple of base_bet.
    - Flat Betting: Fixed bet equal to base_bet.
    - Confidence-Based: 2% + up to 3% based on confidence, rounded to multiple of base_bet.
    - Enforce minimum and maximum bet limits.
    """
    min_bet = max(1.0, base_bet)  # Minimum bet is at least base_bet or $1
    max_bet = bankroll  # Max bet is the entire bankroll to prevent bankruptcy

    if strategy == "Fixed 5% of Bankroll":
        calculated_bet = bankroll * 0.05
    elif strategy == "Flat Betting":
        calculated_bet = base_bet
    elif strategy == "Confidence-Based":
        if confidence is None:
            confidence = 50  # Default confidence if not provided
        confidence_factor = confidence / 100.0
        bet_percentage = 0.02 + (confidence_factor * 0.03)  # 2% + up to 3%
        calculated_bet = bankroll * bet_percentage
    else:
        calculated_bet = base_bet  # Fallback to base_bet

    # Round to nearest multiple of base_bet
    bet_size = round(calculated_bet / base_bet) * base_bet
    # Ensure bet size is at least base_bet (or $1) and does not exceed bankroll
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

def calculate_bankroll(history, base_bet, strategy):
    """
    Calculate bankroll after each round, using selected money management strategy.
    """
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []  # Track bet sizes for display
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        # Use advanced bet selection to predict before current round
        bet, confidence, _, _ = advanced_bet_selection(current_rounds[:-1]) if i != 0 else ('Pass', 0, '', 'Neutral')
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        # Calculate bet size based on selected strategy
        bet_size = money_management(current_bankroll, base_bet, strategy, confidence)
        bet_sizes.append(bet_size)
        if actual_result == bet:
            if bet == 'Banker':
                # Win with 5% commission
                win_amount = bet_size * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += bet_size
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def main():
    st.set_page_config(page_title="Smart Baccarat Predictor with Emotions", page_icon="🎲", layout="centered")
    st.title("Smart Baccarat Predictor with Emotions")

    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.session_state.money_management_strategy = "Fixed 5% of Bankroll"

    # Input fields
    col_init, col_base, col_strategy = st.columns(3)
    with col_init:
        initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
    with col_base:
        base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
    with col_strategy:
        strategy_options = ["Fixed 5% of Bankroll", "Flat Betting", "Confidence-Based"]
        money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))

    st.session_state.initial_bankroll = initial_bankroll
    st.session_state.base_bet = base_bet
    st.session_state.money_management_strategy = money_management_strategy

    # Game input buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Banker"):
            st.session_state.history.append("Banker")
    with col2:
        if st.button("Player"):
            st.session_state.history.append("Player")
    with col3:
        if st.button("Tie"):
            st.session_state.history.append("Tie")

    # Bead Plate History (Adapted from origmangbacc)
    st.markdown("### Bead Plate")
    sequence = [r for r in st.session_state.history]  # Copy history
    sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence][-84:]  # Map to P/B/T and limit to 84
    grid = [['' for _ in range(14)] for _ in range(6)]
    for i, result in enumerate(sequence):
        if result in ['P', 'B', 'T']:
            col = i // 6
            row = i % 6
            if col < 14:
                color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
    for row in grid:
        st.markdown(' '.join(row), unsafe_allow_html=True)
    if not st.session_state.history:
        st.write("_No results yet. Click the buttons above to add results._")

    st.markdown("---")

    # Display selected money management strategy
    st.markdown(f"**Selected Money Management Strategy:** {money_management_strategy}")

    # Bet prediction
    bet, confidence, reason, emotional_tone = advanced_bet_selection(st.session_state.history)
    st.markdown("### Prediction for Next Bet")
    if bet == 'Pass':
        st.warning("I’m not betting this time! The pattern is too unclear.")
        st.info(reason)
        recommended_bet_size = 0.0
    else:
        current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else initial_bankroll
        recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy, confidence)
        st.success(f"Predicted Bet: **{bet}**    Confidence: **{confidence}%**    Recommended Bet Size: **${recommended_bet_size:.2f}**    Emotion: **{emotional_tone}**")
        st.write(reason)

    # Bankroll progression
    bankroll_progress, bet_sizes = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)
    if bankroll_progress:
        st.markdown("### Bankroll and Bet Size Progression")
        for i, (val, bet_size) in enumerate(zip(bankroll_progress, bet_sizes), 1):
            bet_display = f"Bet: ${bet_size:.2f}" if bet_size > 0 else "Bet: None (No prediction, Tie, or Pass)"
            st.write(f"After hand {i}: Bankroll ${val:.2f}, {bet_display}")
        st.markdown(f"**Current Bankroll:** ${bankroll_progress[-1]:.2f}")
    else:
        st.markdown(f"**Current Bankroll:** ${initial_bankroll:.2f}")

    if st.button("Reset History and Bankroll"):
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.session_state.money_management_strategy = "Fixed 5% of Bankroll"
        st.experimental_rerun()

if __name__ == "__main__":
    main()
