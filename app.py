import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List

# --- Constants ---
SESSION_FILE = "online_users.txt"
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["T3", "Flatbet", "Parlay16"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50

# --- Session Tracking ---
def track_user_session() -> int:
    """Track active user sessions using a file-based approach."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())

    sessions = {}
    current_time = datetime.now()

    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        session_id, timestamp = line.strip().split(',')
                        last_seen = datetime.fromisoformat(timestamp)
                        if current_time - last_seen <= timedelta(seconds=30):
                            sessions[session_id] = last_seen
                    except ValueError:
                        continue
    except PermissionError:
        st.error("Unable to access session file. Online user count unavailable.")
        return 0

    sessions[st.session_state.session_id] = current_time

    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except PermissionError:
        st.error("Unable to write to session file. Online user count may be inaccurate.")
        return 0

    return len(sessions)

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_base_bet': 0.0,
        'sequence': [],
        'pending_bet': None,
        'strategy': 'T3',
        't3_level': 1,
        't3_results': [],
        't3_level_changes': 0,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'advice': "",
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_mode': 'Profit %',
        'target_value': 10.0,
        'initial_bankroll': 0.0,
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Validate strategy
    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'T3'

def reset_session():
    """Reset session state to initial values after target is hit."""
    initialize_session_state()
    st.session_state.update({
        'bankroll': st.session_state.initial_bankroll,
        'sequence': [],
        'pending_bet': None,
        't3_level': 1,
        't3_results': [],
        't3_level_changes': 0,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'advice': "Session reset: Target reached.",
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_hit': False,
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int)
    })

# --- Prediction Logic ---
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, int, int, int, float]:
    """Analyze sequence patterns for bigram, trigram, and pattern transitions."""
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    pattern_transitions = defaultdict(lambda: defaultdict(int))
    streak_count = chop_count = double_count = pattern_changes = 0
    current_streak = last_pattern = None

    for i in range(len(sequence) - 1):
        if i < len(sequence) - 2:
            bigram = tuple(sequence[i:i+2])
            trigram = tuple(sequence[i:i+3])
            next_outcome = sequence[i+2]
            bigram_transitions[bigram][next_outcome] += 1
            if i < len(sequence) - 3:
                trigram_transitions[trigram][next_outcome] += 1

        if i > 0:
            if sequence[i] == sequence[i-1]:
                if current_streak == sequence[i]:
                    streak_count += 1
                else:
                    current_streak = sequence[i]
                    streak_count = 1
                if i > 1 and sequence[i-1] == sequence[i-2]:
                    double_count += 1
            else:
                current_streak = None
                streak_count = 0
                if i > 1 and sequence[i] != sequence[i-2]:
                    chop_count += 1

        if i < len(sequence) - 2:
            current_pattern = (
                'streak' if streak_count >= 2 else
                'chop' if chop_count >= 2 else
                'double' if double_count >= 1 else 'other'
            )
            if last_pattern and last_pattern != current_pattern:
                pattern_changes += 1
            last_pattern = current_pattern
            next_outcome = sequence[i+2]
            pattern_transitions[current_pattern][next_outcome] += 1

    volatility = pattern_changes / max(len(sequence) - 2, 1)
    return (bigram_transitions, trigram_transitions, pattern_transitions,
            streak_count, chop_count, double_count, volatility)

def calculate_weights(streak_count: int, chop_count: int, double_count: int) -> Dict[str, float]:
    """Calculate dynamic weights for prediction factors."""
    total_bets = max(st.session_state.pattern_attempts['bigram'], 1)
    weights = {
        'bigram': 0.4 * (st.session_state.pattern_success['bigram'] / total_bets
                         if st.session_state.pattern_attempts['bigram'] > 0 else 0.5),
        'trigram': 0.3 * (st.session_state.pattern_success['trigram'] / total_bets
                          if st.session_state.pattern_attempts['trigram'] > 0 else 0.5),
        'streak': 0.2 if streak_count >= 2 else 0.05,
        'chop': 0.05 if chop_count >= 2 else 0.01,
        'double': 0.05 if double_count >= 1 else 0.01
    }
    if sum(weights.values()) == 0:
        weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
    total_w = sum(weights.values())
    return {k: max(w / total_w, 0.05) for k, w in weights.items()}

def predict_next() -> Tuple[Optional[str], float, Dict]:
    """Predict the next outcome based on sequence patterns."""
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 3:
        return 'B', 45.86, {}

    recent_sequence = sequence[-WINDOW_SIZE:]
    (bigram_transitions, trigram_transitions, pattern_transitions,
     streak_count, chop_count, double_count, volatility) = analyze_patterns(recent_sequence)
    st.session_state.pattern_volatility = volatility

    prior_p, prior_b = 44.62 / 100, 45.86 / 100
    weights = calculate_weights(streak_count, chop_count, double_count)
    prob_p = prob_b = total_weight = 0
    insights = {}

    # Bigram contribution
    if len(recent_sequence) >= 2:
        bigram = tuple(recent_sequence[-2:])
        total = sum(bigram_transitions[bigram].values())
        if total > 0:
            p_prob = bigram_transitions[bigram]['P'] / total
            b_prob = bigram_transitions[bigram]['B'] / total
            prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total)
            prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total)
            total_weight += weights['bigram']
            insights['Bigram'] = f"{weights['bigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Trigram contribution
    if len(recent_sequence) >= 3:
        trigram = tuple(recent_sequence[-3:])
        total = sum(trigram_transitions[trigram].values())
        if total > 0:
            p_prob = trigram_transitions[trigram]['P'] / total
            b_prob = trigram_transitions[trigram]['B'] / total
            prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total)
            prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total)
            total_weight += weights['trigram']
            insights['Trigram'] = f"{weights['trigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Streak contribution
    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
        current_streak = recent_sequence[-1]
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"

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
        insights['Chop'] = f"{weights['chop']*100:.0f}% ({chop_count} alternations)"

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
        insights['Double'] = f"{weights['double']*100:.0f}% ({recent_sequence[-1]}{recent_sequence[-1]})"

    # Normalize probabilities
    if total_weight > 0:
        prob_p = (prob_p / total_weight) * 100
        prob_b = (prob_b / total_weight) * 100
    else:
        prob_p, prob_b = 44.62, 45.86

    # Adjust for Banker commission
    if abs(prob_p - prob_b) < 2:
        prob_p += 0.5
        prob_b -= 0.5

    # Pattern transition adjustment
    current_pattern = (
        'streak' if streak_count >= 2 else
        'chop' if chop_count >= 2 else
        'double' if double_count >= 1 else 'other'
    )
    total = sum(pattern_transitions[current_pattern].values())
    if total > 0:
        p_prob = pattern_transitions[current_pattern]['P'] / total
        b_prob = pattern_transitions[current_pattern]['B'] / total
        prob_p = 0.9 * prob_p + 0.1 * p_prob * 100
        prob_b = 0.9 * prob_b + 0.1 * b_prob * 100
        insights['Pattern Transition'] = f"10% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Adaptive confidence threshold
    recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
    threshold = 41.0 + (st.session_state.consecutive_losses * 0.5) - (recent_accuracy * 1.0)
    threshold = min(max(threshold, 41.0), 51.0)
    insights['Threshold'] = f"{threshold:.1f}%"

    if st.session_state.pattern_volatility > 0.5:
        threshold += 2.0
        insights['Volatility'] = f"High (Adjustment: +2% threshold)"

    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights
    elif prob_b >= threshold:
        return 'B', prob_b, insights
    return None, max(prob_p, prob_b), insights

# --- Betting Logic ---
def check_target_hit() -> bool:
    """Check if the profit target has been reached."""
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
    return unit_profit >= st.session_state.target_value

def update_t3_level():
    """Update T3 betting level based on recent results."""
    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        old_level = st.session_state.t3_level
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = st.session_state.t3_level + 1
        elif losses == 3:
            st.session_state.t3_level = st.session_state.t3_level + 2
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_results = []

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate the next bet amount based on strategy and conditions."""
    if st.session_state.consecutive_losses >= 3 and conf < 45.0:
        return None, f"No bet: Paused after {st.session_state.consecutive_losses} losses"
    if st.session_state.pattern_volatility > 0.5:
        return None, f"No bet: High pattern volatility"
    if pred is None or conf < 41.0:
        return None, f"No bet: Confidence too low"

    if st.session_state.strategy == 'Flatbet':
        bet_amount = st.session_state.base_bet
    elif st.session_state.strategy ==nhance the user experience by making the interface cleaner and more focused on actionable information.

### Testing Instructions

1. Save the code as `app.py` in `/mount/src/testpb/app.py`.
2. Install dependencies: `pip install streamlit numpy`.
3. Run: `streamlit run app.py`.
4. Test scenarios:
   - **Low Confidence**: Start a session (e.g., $1000 bankroll, $10 base bet, Flatbet). Simulate a sequence yielding low confidence (e.g., enter 'P', 'B', 'P' with weak patterns). When confidence < 41% (e.g., 22.5%), verify the message in the “Prediction” section is `"No bet: Confidence too low"`.
   - **Consecutive Losses**: Simulate 3 losses (e.g., predict 'P', enter 'B' three times). If confidence < 45% (e.g., 43%), confirm the message is `"No bet: Paused after 3 losses"`.
   - **High Volatility**: Enter a sequence with frequent pattern changes (e.g., 'P', 'B', 'P', 'B'). If `pattern_volatility > 0.5`, check for `"No bet: High pattern volatility"`.
   - **Bankroll Risk**: Reduce bankroll (e.g., set $20 bankroll, $10 bet) and verify `"No bet: Risk too high for current bankroll."`.
   - **Successful Bet**: Place a bet with confidence ≥ 41% (e.g., 45%). Confirm the message shows confidence, e.g., `"Next Bet: $10 on P (45.0%)"`, as this is necessary for bet details.
   - **UI Consistency**: Ensure “Prediction Insights” still shows the threshold (e.g., “Threshold: 41.0%”) and confidence details for transparency, but pause messages remain simple.
   - **Loss Log**: Check “Recent Losses” to ensure logged confidence values are still recorded (for your analysis, not displayed in UI messages).

### Considerations

- **User Experience**: The simplified messages make the UI cleaner, focusing on why a bet is skipped without technical details, aligning with your request for necessary information only. Confidence percentages remain in “Prediction Insights” and “Recent Losses” for debugging or analysis.
- **Conservative Strategy**: The change is cosmetic and doesn’t affect your conservative safeguards (41%/45% thresholds, volatility pause, bankroll checks), ensuring risk management remains intact.
- **Performance Monitoring**: With your noted performance (3 wins, 13 losses), the clearer messages may help you focus on betting decisions. If losses persist, share loss log details from “Recent Losses” (sequence, prediction, confidence), and I can suggest adjustments (e.g., raise thresholds to 43%/47%).
- **Further Simplification**: If you want other messages simplified (e.g., remove loss count in `"Paused after X losses"`), let me know.
- **Additional Features**: If you want to enhance the UI further (e.g., a “skip bet” button, toggle to show/hide confidence in messages, or performance alerts), I can implement them.

If you encounter issues, want to tweak other messages, or need analysis of specific loss patterns, please share details, and I’ll assist!
