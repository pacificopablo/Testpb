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

def advanced_bet_selection(results):
    """
    Smarter bet selection logic with enhanced heuristics:
    - Streak detection with dynamic confidence
    - Alternating pattern detection
    - Short-term trend analysis (last 10 hands)
    - Long-term frequency analysis
    - Conservative fallback to avoid Ties unless strong evidence
    Returns bet choice, confidence, reason, and emotional tone.
    """
    max_recent_count = 30
    recent = results[-max_recent_count:]
    if not recent:
        return None, 0, "Kailangan ng past results para magbigay ng taya. Subukan ulit!", "Cautious"

    streak_value, streak_length = detect_streak(recent)
    freq = frequency_count(recent)
    trend_bet, trend_score = recent_trend_analysis(recent)

    # Initialize scores
    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    emotional_tone = "Neutral"

    # Streak detection (highest priority)
    if streak_length >= 3 and streak_value != "Tie":
        confidence = min(75 + (streak_length - 3) * 10, 95)
        scores[streak_value] += confidence
        reason_parts.append(f"Malakas na streak: {streak_length} sunod-sunod na {streak_value}! Malaki ang tiwala ko dito.")
        emotional_tone = "Confident" if confidence > 85 else "Optimistic"

    # Alternating pattern detection
    elif len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += 70
        reason_parts.append(f"May alternating pattern sa huling 4 na kamay. Kaya taya ko sa {alternate_bet} ngayon!")
        emotional_tone = "Excited"

    # Short-term trend analysis
    elif trend_bet:
        scores[trend_bet] += trend_score
        reason_parts.append(f"Sa huling 10 kamay, mas malakas ang {trend_bet}. Mukhang maganda itong taya.")
        emotional_tone = "Hopeful"

    # Long-term frequency analysis as fallback
    if not reason_parts:
        total = len(recent)
        scores['Banker'] += (freq['Banker'] / total * 0.9) * 50  # Adjust for commission
        scores['Player'] += (freq['Player'] / total * 1.0) * 50
        scores['Tie'] += (freq['Tie'] / total * 0.5) * 50  # Lower weight for Tie
        reason_parts.append(
            f"Base sa {total} kamay: Banker {freq['Banker']} ({scores['Banker']:.1f} puntos), "
            f"Player {freq['Player']} ({scores['Player']:.1f} puntos), "
            f"Tie {freq['Tie']} ({scores['Tie']:.1f} puntos). Walang klarong pattern, pero may maliit na edge.")
        emotional_tone = "Cautious"

    # Determine best choice
    bet_choice = max(scores, key=scores.get)
    confidence = min(round(scores[bet_choice]), 95)

    # Avoid Tie unless very strong evidence
    if bet_choice == 'Tie' and confidence < 80:
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice]), 95)
        reason_parts.append("Masyadong risky ang Tie maliban kung sigurado. Nagpalit sa mas ligtas na taya.")
        emotional_tone = "Cautious"

    # Adjust confidence based on pattern strength
    if streak_length >= 5:
        confidence = min(confidence + 10, 95)
        emotional_tone = "Very Confident"
    elif confidence < 60:
        emotional_tone = "Hesitant"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone

def calculate_bet_size(bankroll, base_bet):
    """
    Money management: Calculate bet size as 5% of current bankroll.
    - Round to nearest multiple of base_bet.
    - Enforce minimum and maximum bet limits.
    """
    min_bet = max(1.0, base_bet)  # Minimum bet is at least base_bet or $1
    max_bet = bankroll  # Max bet is the entire bankroll to prevent bankruptcy
    calculated_bet = bankroll * 0.05  # 5% of current bankroll
    # Round to nearest multiple of base_bet
    bet_size = round(calculated_bet / base_bet) * base_bet
    # Ensure bet size is at least base_bet (or $1) and does not exceed bankroll
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

def calculate_bankroll(history, base_bet):
    """
    Calculate bankroll after each round, using dynamic bet sizing based on 5% of current bankroll.
    """
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []  # Track bet sizes for display
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        # Use advanced bet selection to predict before current round
        bet, _, _, _ = advanced_bet_selection(current_rounds[:-1]) if i != 0 else (None, 0, '', 'Neutral')
        actual_result = history[i]
        if bet is None:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        if bet == 'Tie':
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        # Calculate bet size as 5% of current bankroll
        bet_size = calculate_bet_size(current_bankroll, base_bet)
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
    st.set_page_config(page_title="Matalinong Baccarat Predictor na May Emosyon", page_icon="🎲", layout="centered")
    st.title("Matalinong Baccarat Predictor na May Emosyon")

    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0

    col_init, col_base = st.columns(2)
    with col_init:
        initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
    with col_base:
        base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")

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

    st.markdown("### Kasaysayan ng Laro")
    if st.session_state.history:
        for i, result in enumerate(reversed(st.session_state.history), 1):
            st.write(f"{len(st.session_state.history) - i + 1}. {result}")
    else:
        st.write("_Wala pang results. Pindutin ang mga button sa itaas para magdagdag._")

    st.markdown("---")

    bet, confidence, reason, emotional_tone = advanced_bet_selection(st.session_state.history)
    st.markdown("### Hula para sa Susunod na Taya")
    if bet is None:
        st.warning("Wala pang sapat na datos para magbigay ng matibay na hula.")
        st.info(reason)
        recommended_bet_size = 0.0
    else:
        current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet)[0][-1] if st.session_state.history else initial_bankroll
        recommended_bet_size = calculate_bet_size(current_bankroll, st.session_state.base_bet)
        st.success(f"Hinulaang Taya: **{bet}**    Kumpiyansa: **{confidence}%**    Inirerekomendang Halaga ng Taya: **${recommended_bet_size:.2f}**    Emosyon: **{emotional_tone}**")
        st.write(reason)

    bankroll_progress, bet_sizes = calculate_bankroll(st.session_state.history, st.session_state.base_bet)
    if bankroll_progress:
        st.markdown("### Progresyon ng Bankroll at Halaga ng Taya")
        for i, (val, bet_size) in enumerate(zip(bankroll_progress, bet_sizes), 1):
            bet_display = f"Taya: ${bet_size:.2f}" if bet_size > 0 else "Taya: Wala (Walang hula o Tie)"
            st.write(f"Pagkatapos ng kamay {i}: Bankroll ${val:.2f}, {bet_display}")
        st.markdown(f"**Kasalukuyang Bankroll:** ${bankroll_progress[-1]:.2f}")
    else:
        st.markdown(f"**Kasalukuyang Bankroll:** ${initial_bankroll:.2f}")

    if st.button("I-reset ang Kasaysayan at Bankroll"):
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.experimental_rerun()

if __name__ == "__main__":
    main()
