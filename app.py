import streamlit as st
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta
import numpy as np

# --- FILE-BASED SESSION TRACKING ---
SESSION_FILE = "online_users.txt"

def track_user_session_file():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())
    
    sessions = {}
    current_time = datetime.now()
    try:
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    session_id, timestamp = line.strip().split(',')
                    last_seen = datetime.fromisoformat(timestamp)
                    if current_time - last_seen <= timedelta(seconds=30):
                        sessions[session_id] = last_seen
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
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

# --- APP CONFIG ---
st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 0.0
    st.session_state.base_bet = 0.0
    st.session_state.initial_base_bet = 0.0
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.strategy = 'T3'
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.t3_level_changes = 0
    st.session_state.parlay_step = 1
    st.session_state.parlay_wins = 0
    st.session_state.parlay_using_base = True
    st.session_state.parlay_step_changes = 0
    st.session_state.advice = ""
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_mode = 'Profit %'
    st.session_state.target_value = 10.0
    st.session_state.initial_bankroll = 0.0
    st.session_state.target_hit = False
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)

# Validate strategy
if 'strategy' in st.session_state and st.session_state.strategy not in ['T3', 'Flatbet', 'Parlay16']:
    st.session_state.strategy = 'T3'

# --- PARLAY TABLE ---
PARLAY_TABLE = {
    1: {'base': 1, 'parlay': 2},
    2: {'base': 1, 'parlay': 2},
    3: {'base': 1, 'parlay': 2},
    4: {'base': 2, 'parlay': 4},
    5: {'base': 3, 'parlay': 6},
    6: {'base': 4, 'parlay': 8},
    7: {'base': 6, 'parlay': 12},
    8: {'base': 8, 'parlay': 16},
    9: {'base': 12, 'parlay': 24},
    10: {'base': 16, 'parlay': 32},
    11: {'base': 22, 'parlay': 44},
    12: {'base': 30, 'parlay': 60},
    13: {'base': 40, 'parlay': 80},
    14: {'base': 52, 'parlay': 104},
    15: {'base': 70, 'parlay': 140},
    16: {'base': 95, 'parlay': 190}
}

# --- FUNCTIONS ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 4:  # Require at least 4 hands for quadgram
        return 'B', 45.86, {'Default': 'Insufficient data, default to Banker (45.86%)'}

    # Sliding window increased to 60 hands
    window_size = 60
    recent_sequence = sequence[-window_size:]

    # Initialize data structures
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    quadgram_transitions = defaultdict(lambda: defaultdict(int))
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

        # Quadgram transitions
        if i < len(recent_sequence) - 4:
            quadgram = tuple(recent_sequence[i:i+4])
            next_outcome = recent_sequence[i+4]
            quadgram_transitions[quadgram][next_outcome] += 1

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

    # Dynamic weights with feedback
    total_bets = max(sum(st.session_state.pattern_attempts.values()), 1)
    weights = {
        'bigram': 0.3 * (st.session_state.pattern_success['bigram'] / st.session_state.pattern_attempts['bigram'] if st.session_state.pattern_attempts['bigram'] > 0 else 0.5),
        'trigram': 0.4 * (st.session_state.pattern_success['trigram'] / st.session_state.pattern_attempts['trigram'] if st.session_state.pattern_attempts['trigram'] > 0 else 0.5),
        'quadgram': 0.2 * (st.session_state.pattern_success['quadgram'] / st.session_state.pattern_attempts['quadgram'] if st.session_state.pattern_attempts['quadgram'] > 0 else 0.5),
        'streak': 0.15 if streak_count >= 2 else 0.05,
        'chop': 0.05 if chop_count >= 2 else 0.01,
        'double': 0.05 if double_count >= 1 else 0.01
    }
    if sum(weights.values()) == 0:
        weights = {'bigram': 0.3, 'trigram': 0.4, 'quadgram': 0.2, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}
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
            prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total_transitions)
            total_weight += weights['bigram']
            insights['Bigram'] = f"{weights['bigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

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
            insights['Trigram'] = f"{weights['trigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Quadgram contribution
    if len(recent_sequence) >= 4:
        quadgram = tuple(recent_sequence[-4:])
        total_transitions = sum(quadgram_transitions[quadgram].values())
        if total_transitions > 0:
            p_prob = quadgram_transitions[quadgram]['P'] / total_transitions
            b_prob = quadgram_transitions[quadgram]['B'] / total_transitions
            prob_p += weights['quadgram'] * (prior_p + p_prob) / (1 + total_transitions)
            prob_b += weights['quadgram'] * (prior_b + b_prob) / (1 + total_transitions)
            total_weight += weights['quadgram']
            insights['Quadgram'] = f"{weights['quadgram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Streak contribution with stronger anti-streak bias
    if streak_count >= 2:
        streak_prob = min(0.6, 0.5 + streak_count * 0.03) * (0.5 if streak_count > 5 else 0.8 if streak_count > 4 else 1.0)
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
        double_prob = 0.55
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
        prob_p = 44.62
        prob_b = 45.86

    # Adjust for Banker commission
    if abs(prob_p - prob_b) < 1.5:
        prob_p += 0.3
        prob_b -= 0.3

    # Pattern transition adjustment
    current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
    total_transitions = sum(pattern_transitions[current_pattern].values())
    if total_transitions > 0:
        p_prob = pattern_transitions[current_pattern]['P'] / total_transitions
        b_prob = pattern_transitions[current_pattern]['B'] / total_transitions
        prob_p = 0.85 * prob_p + 0.15 * p_prob * 100
        prob_b = 0.85 * prob_b + 0.15 * b_prob * 100
        insights['Pattern Transition'] = f"15% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    # Adaptive confidence threshold
    recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
    base_threshold = 52.0
    threshold = base_threshold + (st.session_state.consecutive_losses * 0.8) - (recent_accuracy * 1.2)
    threshold = min(max(threshold, 50.0), 62.0)
    insights['Threshold'] = f"{threshold:.1f}%"

    # Volatility adjustment
    if st.session_state.pattern_volatility > 0.6:
        threshold += 3.0
        insights['Volatility'] = f"High (Adjustment: +3% threshold)"

    # Pattern success rate
    for pattern in ['bigram', 'trigram', 'quadgram', 'streak', 'chop', 'double']:
        if st.session_state.pattern_attempts[pattern] > 0:
            success_rate = (st.session_state.pattern_success[pattern] / st.session_state.pattern_attempts[pattern]) * 100
            insights[f"{pattern.capitalize()} Success"] = f"{success_rate:.1f}% ({st.session_state.pattern_success[pattern]}/{st.session_state.pattern_attempts[pattern]})"

    # Determine prediction
    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights
    elif prob_b >= threshold:
        return 'B', prob_b, insights
    else:
        return None, max(prob_p, prob_b), insights

def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        return unit_profit >= st.session_state.target_value

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.t3_level_changes = 0
    st.session_state.parlay_step = 1
    st.session_state.parlay_wins = 0
    st.session_state.parlay_using_base = True
    st.session_state.parlay_step_changes = 0
    st.session_state.advice = "Session reset: Target reached."
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_hit = False
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)

def place_result(result, manual_selection=None):
    if st.session_state.target_hit:
        reset_session_auto()
        return
    st.session_state.last_was_tie = (result == 'T')
    bet_amount = 0
    bet_placed = False
    selection = None
    win = False

    # Store state
    previous_state = {
        "bankroll": st.session_state.bankroll,
        "t3_level": st.session_state.t3_level,
        "t3_results": st.session_state.t3_results.copy(),
        "parlay_step": st.session_state.parlay_step,
        "parlay_wins": st.session_state.parlay_wins,
        "parlay_using_base": st.session_state.parlay_using_base,
        "pending_bet": st.session_state.pending_bet,
        "wins": st.session_state.wins,
        "losses": st.session_state.losses,
        "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
        "consecutive_losses": st.session_state.consecutive_losses,
        "t3_level_changes": st.session_state.t3_level_changes,
        "parlay_step_changes": st.session_state.parlay_step_changes,
        "pattern_volatility": st.session_state.pattern_volatility,
        "pattern_success": st.session_state.pattern_success.copy(),
        "pattern_attempts": st.session_state.pattern_attempts.copy()
    }

    if st.session_state.pending_bet and result != 'T':
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        bet_placed = True
        if win:
            if selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:
                st.session_state.bankroll += bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('W')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins += 1
                if st.session_state.parlay_wins == 2:
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = 1
                    st.session_state.parlay_wins = 0
                    st.session_state.parlay_using_base = True
                    if old_step != st.session_state.parlay_step:
                        st.session_state.parlay_step_changes += 1
                else:
                    st.session_state.parlay_using_base = False
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
            # Update pattern success
            for pattern in ['bigram', 'trigram', 'quadgram', 'streak', 'chop', 'double']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_success[pattern] += 1
                    st.session_state.pattern_attempts[pattern] += 1
        else:
            st.session_state.bankroll -= bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('L')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins = 0
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
            st.session_state.losses += 1
            st.session_state.consecutive_losses += 1
            st.session_state.loss_log.append({
                'sequence': st.session_state.sequence[-10:],
                'prediction': selection,
                'result': result,
                'confidence': st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0',
                'insights': st.session_state.insights.copy()
            })
            if len(st.session_state.loss_log) > 50:
                st.session_state.loss_log = st.session_state.loss_log[-50:]
            # Update pattern attempts
            for pattern in ['bigram', 'trigram', 'quadgram', 'streak', 'chop', 'double']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_attempts[pattern] += 1
        st.session_state.prediction_accuracy['total'] += 1
        st.session_state.pending_bet = None

    # Append to sequence
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Store history
    st.session_state.history.append({
        "Bet": selection,
        "Result": result,
        "Amount": bet_amount,
        "Win": win,
        "T3_Level": st.session_state.t3_level,
        "Parlay_Step": st.session_state.parlay_step,
        "Parlay_Wins": st.session_state.parlay_wins,
        "Parlay_Using_Base": st.session_state.parlay_using_base,
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed
    })
    if len(st.session_state.history) > 1000:
        st.session_state.history = st.session_state.history[-1000:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    # Calculate next bet
    pred, conf, insights = predict_next()
    if manual_selection in ['P', 'B']:
        pred = manual_selection
        conf = max(conf, 50.0)
        st.session_state.advice = f"Manual Bet: {pred} (User override)"

    # Dynamic bet sizing
    bet_scaling = 1.0
    if st.session_state.consecutive_losses >= 2:
        bet_scaling *= 0.85
    if conf < 55.0:
        bet_scaling *= 0.9

    # Loss streak pause
    if st.session_state.consecutive_losses >= 3 and conf < 60.0:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet: Paused after {st.session_state.consecutive_losses} losses (Confidence: {conf:.1f}% < 60%)"
        st.session_state.insights = insights
        return

    # Volatility check
    if st.session_state.pattern_volatility > 0.6 and conf < 60.0:
        st.session_state.advice = f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.pending_bet = None
        st.session_state.insights = insights
        return

    if pred is None or conf < 50.0:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet (Confidence: {conf:.1f}% too low)"
        st.session_state.insights = insights
    else:
        if st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet * bet_scaling
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level * bet_scaling
        elif st.session_state.strategy == 'Parlay16':
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key] * bet_scaling
            if bet_amount > st.session_state.bankroll:
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
                bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step]['base'] * bet_scaling

        # Bankroll-aware filtering
        safe_bankroll = st.session_state.initial_bankroll * 0.2
        max_bet_percent = 0.05
        if (bet_amount > st.session_state.bankroll or
            st.session_state.bankroll - bet_amount < safe_bankroll or
            bet_amount > st.session_state.bankroll * max_bet_percent):
            st.session_state.pending_bet = None
            st.session_state.advice = "No bet: Risk too high for current bankroll."
            st.session_state.insights = insights
            if st.session_state.strategy == 'Parlay16':
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
        else:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"
            st.session_state.insights = insights

    if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
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

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet", "Parlay16"],
        index={'T3': 0, 'Flatbet': 1, 'Parlay16': 2}.get(st.session_state.strategy, 0),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression."
    )
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    if bankroll <= 0:
        st.error("Bankroll must be positive.")
    elif base_bet <= 0:
        st.error("Base bet must be positive.")
    elif base_bet > bankroll:
        st.error("Base bet cannot exceed bankroll.")
    else:
        st.session_state.bankroll = bankroll
        st.session_state.base_bet = base_bet
        st.session_state.initial_base_bet = base_bet
        st.session_state.strategy = betting_strategy
        st.session_state.sequence = []
        st.session_state.pending_bet = None
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.t3_level_changes = 0
        st.session_state.parlay_step = 1
        st.session_state.parlay_wins = 0
        st.session_state.parlay_using_base = True
        st.session_state.parlay_step_changes = 0
        st.session_state.advice = ""
        st.session_state.history = []
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.session_state.target_mode = target_mode
        st.session_state.target_value = target_value
        st.session_state.initial_bankroll = bankroll
        st.session_state.target_hit = False
        st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
        st.session_state.consecutive_losses = 0
        st.session_state.loss_log = []
        st.session_state.last_was_tie = False
        st.session_state.insights = {}
        st.session_state.pattern_volatility = 0.0
        st.session_state.pattern_success = defaultdict(int)
        st.session_state.pattern_attempts = defaultdict(int)
        st.success(f"Session started with {betting_strategy} strategy!")

# --- MANUAL OVERRIDE ---
st.subheader("Manual Bet Override")
manual_selection = st.radio("Override Next Bet", ["Auto", "Player", "Banker", "Skip"], index= Stuart at Home (https://www.stuartathom.com) - By default, radio buttons are styled with a simple circle for the unchecked state and a filled circle for the checked state. For a more customized appearance, you can use custom CSS to style radio buttons, but this requires careful consideration to maintain accessibility.

For this example, we’ll stick with the default Streamlit radio button styling, which is clean and functional. If you want to apply custom styling like the buttons above, you can inject CSS via `st.markdown` with the `unsafe_allow_html=True` option.

Here’s how you can structure the manual override section:

```python
st.subheader("Manual Bet Override")
manual_selection = st.radio(
    "Override Next Bet",
    ["Auto", "Player", "Banker", "Skip"],
    index=0,
    horizontal=True
)
if manual_selection == "Skip":
    st.session_state.pending_bet = None
    st.session_state.advice = "Bet skipped by user."
elif manual_selection in ["Player", "Banker"]:
    st.info(f"Next bet will be placed on {manual_selection}.")
