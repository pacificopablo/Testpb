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

def frequency_count(arr):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in arr:
        if r in count:
            count[r] += 1
    return count

def advanced_bet_selection(results):
    """
    Improved bet selection logic based on multiple heuristics:
    - Streak detection
    - Alternating pattern detection
    - Frequency analysis with weighted scores
    - Penalizing ties less, considering their rarity
    Returns bet choice, confidence, and explanation.
    """
    max_recent_count = 30
    recent = results[-max_recent_count:]
    if not recent:
        return None, 0, "Please enter past results."

    streak_value, streak_length = detect_streak(recent)
    freq = frequency_count(recent)

    # If strong streak detected (3 or more)
    if streak_length >= 3 and streak_value != "Tie":
        confidence = min(75 + (streak_length - 3) * 8, 98)
        reason = f"Strong streak detected: {streak_length} consecutive wins by {streak_value}. Betting on the streak continuation."
        return streak_value, confidence, reason

    # Detect alternating pattern on last 4 results
    if len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        reason = "Alternating pattern identified. Suggest betting on the opposite side of the last result."
        return alternate_bet, 70, reason

    # Weighted frequency scoring
    # Tie is rare but can disrupt pattern, give it a small penalty but not ignore entirely
    total = len(recent)
    scores = {}
    scores['Banker'] = freq['Banker'] / total * 0.9  # Slightly penalize because of commission in real game
    scores['Player'] = freq['Player'] / total * 1.0
    scores['Tie'] = freq['Tie'] / total * 0.6  # Least weight given to Tie

    # Determine best choice
    bet_choice = max(scores, key=scores.get)
    confidence = round(scores[bet_choice] * 100)

    # Adjust confidence upwards if difference between top and others is large
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) > 0.15:
        confidence = min(confidence + 10, 98)

    reason = (
        f"Frequency-based analysis over the last {total} hands:\n"
        f"Banker: {freq['Banker']} ({scores['Banker']*100:.1f}% weighted), "
        f"Player: {freq['Player']} ({scores['Player']*100:.1f}% weighted), "
        f"Tie: {freq['Tie']} ({scores['Tie']*100:.1f}% weighted).\n"
        f"Selected bet: {bet_choice} with confidence {confidence}%."
    )

    return bet_choice, confidence, reason

# Calculate bankroll after each round assuming user bets predicted bet with base bet
def calculate_bankroll(history, base_bet):
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        # Use improved bet selection to predict before current round
        bet, _, _ = advanced_bet_selection(current_rounds[:-1]) if i != 0 else (None, 0, '')
        actual_result = history[i]
        if bet is None:
            bankroll_progress.append(current_bankroll)
            continue
        if bet == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        if actual_result == bet:
            if bet == 'Banker':
                # Win with 5% commission
                win_amount = base_bet * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += base_bet
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= base_bet
        bankroll_progress.append(current_bankroll)
    return bankroll_progress

def main():
    st.set_page_config(page_title="Baccarat Interactive Predictor with Bankroll", page_icon="🎲", layout="centered")
    st.title("Baccarat Interactive Predictor with Bankroll")

    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0

    col_init, col_base = st.columns(2)
    with col_init:
        initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
    with col_base:
        base_bet = st.number_input("Base Bet", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")

    st.session_state.initial_bankroll = initial_bankroll
    st.session_state.base_bet = base_bet

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

    st.markdown("### Game History")
    if st.session_state.history:
        for i, result in enumerate(reversed(st.session_state.history), 1):
            st.write(f"{len(st.session_state.history) - i + 1}. {result}")
    else:
        st.write("_No results yet. Click the buttons above to enter results._")

    st.markdown("---")

    bet, confidence, reason = advanced_bet_selection(st.session_state.history)
    st.markdown("### Prediction for Next Bet")
    if bet is None:
        st.warning("No confident prediction available yet.")
        st.info(reason)
    else:
        st.success(f"Predicted Bet: **{bet}**    Confidence: **{confidence}%**")
        st.write(reason.replace('\n', '  \n'))

    bankroll_progress = calculate_bankroll(st.session_state.history, st.session_state.base_bet)
    if bankroll_progress:
        st.markdown("### Bankroll Progression")
        for i, val in enumerate(bankroll_progress, 1):
            st.write(f"After hand {i}: ${val:.2f}")
        st.markdown(f"**Current Bankroll:** ${bankroll_progress[-1]:.2f}")
    else:
        st.markdown(f"**Current Bankroll:** ${initial_bankroll:.2f}")

    if st.button("Reset History and Bankroll"):
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.experimental_rerun()

if __name__ == "__main__":
    main()

