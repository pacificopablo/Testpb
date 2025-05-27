import streamlit as st
&nbsp;
&nbsp;

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
&nbsp;
&nbsp;

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
&nbsp;
&nbsp;

def is_alternating_pattern(arr):
    if len(arr) < 4:
        return False
    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1]:
            return False
    return True
&nbsp;
&nbsp;

def frequency_count(arr):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in arr:
        if r in count:
            count[r] += 1
    return count
&nbsp;
&nbsp;

def predict_next(results):
    max_recent_count = 20
    recent = results[-max_recent_count:]
&nbsp;
&nbsp;

    if not recent:
        return None, 0, "Please enter past results."
&nbsp;
&nbsp;

    # 1) Detect streaks
    last_result, streak_len = detect_streak(recent)
    if streak_len >= 3 and last_result != "Tie":
        confidence = min(80 + (streak_len - 3) * 7, 95)
        reason = f"Detected a strong streak ({streak_len}) of {last_result}. Bet on continuation."
        return last_result, confidence, reason
&nbsp;
&nbsp;

    # 2) Detect alternating pattern
    if len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        bet = 'Player' if last == 'Banker' else 'Banker'
        reason = "Detected an alternating pattern (Banker-Player-Banker-Player). Bet on continuation."
        return bet, 65, reason
&nbsp;
&nbsp;

    # 3) Calculate frequencies ignoring ties mostly
    freq = frequency_count(recent)
    total_non_tie = freq['Banker'] + freq['Player']
    if total_non_tie == 0:
        # Only ties or no results
        if freq['Tie'] > 0:
            return 'Tie', 30, "Only ties found but they are very rare outcomes."
        else:
            return None, 0, "No valid results provided."
&nbsp;
&nbsp;

    banker_rate = freq['Banker'] / total_non_tie
    player_rate = freq['Player'] / total_non_tie
&nbsp;
&nbsp;

    bet = 'Banker' if banker_rate >= player_rate else 'Player'
    diff = abs(banker_rate - player_rate)
&nbsp;
&nbsp;

    # House edge penalty for Banker - reduce confidence slightly
    base_confidence = round(max(banker_rate, player_rate) * 100 * 0.85)
    if diff > 0.2:
        base_confidence += 7
    base_confidence = min(base_confidence, 85)
&nbsp;
&nbsp;

    reason = (
        f"Based on the frequency in last {len(recent)} rounds:\n"
        f"Banker: {freq['Banker']}, Player: {freq['Player']}, Tie: {freq['Tie']}\n"
        "Bet on the side with higher frequency, adjusted for house edge and pattern detection."
    )
    return bet, base_confidence, reason
&nbsp;
&nbsp;

# Calculate bankroll after each round assuming user bets predicted bet with base bet
# Payout rules:
# Banker win pays 0.95x (5% commission)
# Player win pays 1x
# Tie pays no win or loss (push)
def calculate_bankroll(history, base_bet):
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _ = predict_next(current_rounds[:-1]) if i != 0 else (None, 0, '')  # Predict bet before current round
        actual_result = history[i]
        # If no prediction before first round, just no bet placed
        if bet is None:
            bankroll_progress.append(current_bankroll)
            continue
        # If bet was Tie (rare in prediction), treat as no bet (no loss or gain)
        if bet == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        # Calculate win/loss
        if actual_result == bet:
            # Win
            if bet == 'Banker':
                # 5% commission on winnings
                win_amount = base_bet * 0.95
                current_bankroll += win_amount
            else:
                # Player win pays 1:1
                current_bankroll += base_bet
        elif actual_result == 'Tie':
            # Tie push - bet returned (no change)
            bankroll_progress.append(current_bankroll)
            continue
        else:
            # Lose bet
            current_bankroll -= base_bet
&nbsp;
&nbsp;

        bankroll_progress.append(current_bankroll)
    return bankroll_progress
&nbsp;
&nbsp;

def main():
    st.set_page_config(page_title="Baccarat Interactive Predictor with Bankroll", page_icon="🎲", layout="centered")
    st.title("Baccarat Interactive Predictor with Bankroll")
&nbsp;
&nbsp;

    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
&nbsp;
&nbsp;

    # Bankroll and base bet inputs
    col_init, col_base = st.columns(2)
    with col_init:
        initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value= ⬤
