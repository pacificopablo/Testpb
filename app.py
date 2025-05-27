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
    for i in range(len(arr) -1):
        if arr[i] == arr[i+1]:
            return False
    return True

def frequency_count(arr):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in arr:
        if r in count:
            count[r] += 1
    return count

def predict_next(results):
    max_recent_count = 20
    recent = results[-max_recent_count:]

    if not recent:
        return None, 0, "Please enter past results."

    # 1) Detect streaks
    last_result, streak_len = detect_streak(recent)
    if streak_len >= 3 and last_result != "Tie":
        confidence = min(80 + (streak_len - 3) * 7, 95)
        reason = f"Detected a strong streak ({streak_len}) of {last_result}. Bet on continuation."
        return last_result, confidence, reason

    # 2) Detect alternating pattern
    if len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        bet = 'Player' if last == 'Banker' else 'Banker'
        reason = "Detected an alternating pattern (Banker-Player-Banker-Player). Bet on continuation."
        return bet, 65, reason

    # 3) Calculate frequencies ignoring ties mostly
    freq = frequency_count(recent)
    total_non_tie = freq['Banker'] + freq['Player']
    if total_non_tie == 0:
        # Only ties or no results
        if freq['Tie'] > 0:
            return 'Tie', 30, "Only ties found but they are very rare outcomes."
        else:
            return None, 0, "No valid results provided."

    banker_rate = freq['Banker'] / total_non_tie
    player_rate = freq['Player'] / total_non_tie

    bet = 'Banker' if banker_rate >= player_rate else 'Player'
    diff = abs(banker_rate - player_rate)

    # House edge penalty for Banker - reduce confidence slightly
    base_confidence = round(max(banker_rate, player_rate) * 100 * 0.85)
    if diff > 0.2:
        base_confidence += 7
    base_confidence = min(base_confidence, 85)

    reason = (
        f"Based on the frequency in last {len(recent)} rounds:\n"
        f"Banker: {freq['Banker']}, Player: {freq['Player']}, Tie: {freq['Tie']}\n"
        "Bet on the side with higher frequency, adjusted for house edge and pattern detection."
    )
    return bet, base_confidence, reason

def main():
    st.set_page_config(page_title="Baccarat Interactive Predictor", page_icon="🎲", layout="centered")
    st.title("Baccarat Interactive Predictor")

    if 'history' not in st.session_state:
        st.session_state.history = []

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
        # Show history reversed with latest on top for better UX
        for i, result in enumerate(reversed(st.session_state.history), 1):
            st.write(f"{len(st.session_state.history) - i + 1}. {result}")
    else:
        st.write("_No results yet. Click the buttons above to enter results._")

    st.markdown("---")
    bet, confidence, reason = predict_next(st.session_state.history)

    st.markdown("### Prediction for Next Bet")
    if bet is None:
        st.warning("No confident prediction available yet.")
        st.info(reason)
    else:
        st.success(f"Predicted Bet: **{bet}**    Confidence: **{confidence}%**")
        st.write(reason.replace('\n', '  \n'))

    if st.button("Reset History"):
        st.session_state.history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
