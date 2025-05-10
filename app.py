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

# --- NEW FUNCTION: MANAGE PROFIT AND LOSS ---
def manage_profit_and_loss(session_profit):
    if session_profit > 0:
        saved = session_profit * 0.5
        reinvested = session_profit * 0.5
        st.session_state.saved_profits += saved
        st.session_state.total_bankroll += reinvested
        st.session_state.session_profit += session_profit
        st.info(f"Profit: ${session_profit:.2f} | Saved: ${saved:.2f} | Reinvested: ${reinvested:.2f}")
    elif session_profit < 0:
        st.session_state.session_profit += session_profit
        st.session_state.total_bankroll += session_profit
        st.warning(f"Loss: ${-session_profit:.2f}")

    win_limit = st.session_state.session_bankroll * st.session_state.win_limit_percent
    loss_limit = st.session_state.session_bankroll * st.session_state.loss_limit_percent
    if st.session_state.session_profit >= win_limit:
        st.session_state.session_active = False
        st.success(f"Win limit reached: ${st.session_state.session_profit:.2f} (Target: ${win_limit:.2f}). Session paused.")
    elif -st.session_state.session_profit >= loss_limit:
        st.session_state.session_active = False
        st.error(f"Loss limit reached: ${-st.session_state.session_profit:.2f} (Target: ${loss_limit:.2f}). Session paused.")

    if st.session_state.total_bankroll < st.session_state.initial_total_bankroll * 0.8:
        new_session_bankroll = max(st.session_state.total_bankroll / 10, 10.0)
        st.session_state.session_bankroll = new_session_bankroll
        st.session_state.base_bet = max(new_session_bankroll * 0.02, 1.0)
        st.session_state.initial_base_bet = st.session_state.base_bet
        st.warning(f"Bankroll low: ${st.session_state.total_bankroll:.2f}. Adjusted session bankroll to ${new_session_bankroll:.2f}, base bet to ${st.session_state.base_bet:.2f}")

# --- NEW FUNCTION: AUTO-OPTIMIZE PREDICTION WEIGHTS ---
def auto_optimize_weights():
    if len(st.session_state.sequence) % 10 == 0 and len(st.session_state.sequence) > 0:
        total_attempts = sum(st.session_state.pattern_attempts.values())
        if total_attempts < 10:
            return
        weights = {}
        for pattern in ['bigram', 'trigram', 'streak', 'chop', 'double']:
            success = st.session_state.pattern_success[pattern]
            attempts = st.session_state.pattern_attempts[pattern]
            if attempts > 0:
                accuracy = success / attempts
                weights[pattern] = max(0.05, 0.4 * accuracy)  # Base weight scaled by accuracy
            else:
                weights[pattern] = 0.05
        total_w = sum(weights.values())
        for k in weights:
            weights[k] = weights[k] / total_w
        st.session_state.optimized_weights = weights
        st.session_state.optimization_status.append(f"Weights updated: Bigram: {weights['bigram']:.2f}, Trigram: {weights['trigram']:.2f}, Streak: {weights['streak']:.2f}, Chop: {weights['chop']:.2f}, Double: {weights['double']:.2f}")

    # Adjust thresholds and scaling based on win/loss ratio
    if st.session_state.losses >= st.session_state.wins * 1.5 and st.session_state.losses > 0:
        st.session_state.dynamic_threshold += 2.0
        st.session_state.dynamic_bet_scale *= 0.9
        st.session_state.optimization_status.append(f"Losses high ({st.session_state.losses}/{st.session_state.wins}). Threshold: {st.session_state.dynamic_threshold:.1f}%, Bet Scale: {st.session_state.dynamic_bet_scale:.2f}")

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
    st.session_state.recovery_mode = False
    st.session_state.insights = {}
    st.session_state.recovery_threshold = 15.0
    st.session_state.recovery_bet_scale = 0.6
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)
    st.session_state.total_bankroll = 0.0
    st.session_state.initial_total_bankroll = 0.0
    st.session_state.session_bankroll = 0.0
    st.session_state.saved_profits = 0.0
    st.session_state.session_profit = 0.0
    st.session_state.session_active = True
    st.session_state.win_limit_percent = 0.25
    st.session_state.loss_limit_percent = 0.5
    # New optimization variables
    st.session_state.optimized_weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
    st.session_state.optimization_status = []
    st.session_state.dynamic_threshold = 52.0  # Base threshold
    st.session_state.dynamic_bet_scale = 1.0   # Base bet scale
    st.session_state.cool_off_counter = 0      # Tracks cool-off period after losses

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

# --- PREDICTION FUNCTION ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 3:
        return 'B', 45.86, {}

    window_size = 50
    recent_sequence = sequence[-window_size:]

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

    prior_p = 44.62 / 100
    prior_b = 45.86 / 100

    # Use optimized weights
    weights = st.session_state.optimized_weights
    total_w = sum(weights.values())
    for k in weights:
        weights[k] = max(weights[k] / total_w, 0.05)

    prob_p = 0
    prob_b = 0
    total_weight = 0

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

    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"

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

    if total_weight > 0:
        prob_p = (prob_p / total_weight) * 100
        prob_b = (prob_b / total_weight) * 100
    else:
        prob_p = 44.62
        prob_b = 45.86

    if abs(prob_p - prob_b) < 2:
        prob_p += 0.5
        prob_b -= 0.5

    current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
    total_transitions = sum(pattern_transitions[current_pattern].values())
    if total_transitions > 0:
        p_prob = pattern_transitions[current_pattern]['P'] / total_transitions
        b_prob = pattern_transitions[current_pattern]['B'] / total_transitions
        prob_p = 0.9 * prob_p + 0.1 * p_prob * 100
        prob_b = 0.9 * prob_b + 0.1 * b_prob * 100
        insights['Pattern Transition'] = f"10% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"

    recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
    threshold = st.session_state.dynamic_threshold + (st.session_state.consecutive_losses * 1.0) - (recent_accuracy * 1.5)
    threshold = min(max(threshold, 50.0 if st.session_state.recovery_mode else 52.0), 65.0)
    insights['Threshold'] = f"{threshold:.1f}%"

    if st.session_state.pattern_volatility > 0.6:  # Stricter volatility threshold
        threshold += 2.0
        insights['Volatility'] = f"High (Adjustment: +2% threshold)"

    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights
    elif prob_b >= threshold:
        return 'B', prob_b, insights
    else:
        return None, max(prob_p, prob_b), insights

# --- TARGET CHECK AND RESET FUNCTIONS ---
def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        return unit_profit >= st.session_state.target_value

def reset_session_auto():
    st.session_state.bankroll = st.session_state.session_bankroll
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
    st.session_state.recovery_mode = False
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)
    st.session_state.session_profit = 0.0
    st.session_state.session_active = True
    st.session_state.optimized_weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
    st.session_state.optimization_status = []
    st.session_state.dynamic_threshold = 52.0
    st.session_state.dynamic_bet_scale = 1.0
    st.session_state.cool_off_counter = 0

# --- PLACE RESULT FUNCTION ---
def place_result(result, manual_selection=None):
    if st.session_state.target_hit or not st.session_state.session_active:
        reset_session_auto()
        return
    st.session_state.last_was_tie = (result == 'T')
    bet_amount = 0
    bet_placed = False
    selection = None
    win = False

    # Check recovery mode
    loss_percentage = (st.session_state.initial_bankroll - st.session_state.bankroll) / st.session_state.initial_bankroll if st.session_state.initial_bankroll > 0 else 0
    st.session_state.recovery_mode = loss_percentage >= st.session_state.recovery_threshold / 100

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
        "recovery_mode": st.session_state.recovery_mode,
        "pattern_volatility": st.session_state.pattern_volatility,
        "pattern_success": st.session_state.pattern_success.copy(),
        "pattern_attempts": st.session_state.pattern_attempts.copy(),
        "session_profit": st.session_state.session_profit,
        "cool_off_counter": st.session_state.cool_off_counter
    }

    if st.session_state.pending_bet and result != 'T':
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        bet_placed = True
        previous_bankroll = st.session_state.bankroll
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
            st.session_state.cool_off_counter = 0
            for pattern in ['bigram', 'trigram', 'streak', 'chop', 'double']:
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
            if st.session_state.consecutive_losses >= 2:
                st.session_state.cool_off_counter = 2
            st.session_state.loss_log.append({
                'sequence': st.session_state.sequence[-10:],
                'prediction': selection,
                'result': result,
                'confidence': st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0',
                'insights': st.session_state.insights.copy()
            })
            if len(st.session_state.loss_log) > 50:
                st.session_state.loss_log = st.session_state.loss_log[-50:]
            for pattern in ['bigram', 'trigram', 'streak', 'chop', 'double']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_attempts[pattern] += 1
        st.session_state.prediction_accuracy['total'] += 1
        st.session_state.pending_bet = None

        # Manage profit/loss
        session_profit = st.session_state.bankroll - previous_bankroll
        manage_profit_and_loss(session_profit)

    # Append to sequence
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Auto-optimize weights
    auto_optimize_weights()

    # Store history
    st.session_state.history.append({
        "Bet": selection,
        "Result": result,
        "Amount": bet_amount,
        "Win": win,
        "T3_Level": st.session_state.t3_level,
        "Parlay_Step": st.session_state.parlay_step,
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed,
        "Session_Profit": st.session_state.session_profit
    })
    if len(st.session_state.history) > 1000:
        st.session_state.history = st.session_state.history[-1000:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    # Manage cool-off period
    if st.session_state.cool_off_counter > 0:
        st.session_state.cool_off_counter -= 1
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet: Cool-off period ({st.session_state.cool_off_counter} hands remaining)"
        st.session_state.insights = {}
        return

    # Calculate next bet
    pred, conf, insights = predict_next()
    if manual_selection in ['P', 'B']:
        pred = manual_selection
        conf = max(conf, 50.0)
        st.session_state.advice = f"Manual Bet: {pred} (User override)"

    # Dynamic bet sizing with stricter loss adjustments
    bet_scaling = st.session_state.dynamic_bet_scale
    if st.session_state.recovery_mode:
        bet_scaling *= st.session_state.recovery_bet_scale
    if st.session_state.consecutive_losses >= 1:
        reduction = min(0.5, 0.2 * st.session_state.consecutive_losses)  # Reduce by 20% per loss, max 50%
        bet_scaling *= (1 - reduction)
    if conf < 55.0:
        bet_scaling *= 0.9

    # Stricter loss streak pause
    if st.session_state.consecutive_losses >= 2 and conf < 65.0:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet: Paused after {st.session_state.consecutive_losses} losses (Confidence: {conf:.1f}% < 65%)"
        st.session_state.insights = insights
        return

    # Stricter volatility check
    if st.session_state.pattern_volatility > 0.6:
        st.session_state.advice = f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.pending_bet = None
        st.session_state.insights = insights
        return

    if pred is None or conf < 52.0:
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
    total_bankroll = st.number_input("Enter Total Bankroll ($)", min_value=0.0, value=st.session_state.total_bankroll, step=10.0)
    session_bankroll = st.number_input("Enter Session Bankroll ($)", min_value=0.0, value=st.session_state.session_bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet", "Parlay16"],
        index={'T3': 0, 'Flatbet': 1, 'Parlay16': 2}.get(st.session_state.strategy, 0),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression."
    )
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    recovery_threshold = st.slider("Recovery Mode Threshold (% Loss)", 10.0, 30.0, st.session_state.recovery_threshold, step=5.0)
    recovery_bet_scale = st.slider("Recovery Mode Bet Scaling", 0.5, 1.0, st.session_state.recovery_bet_scale, step=0.1)
    win_limit_percent = st.slider("Win Limit (% of Session Bankroll)", 20.0, 30.0, st.session_state.win_limit_percent * 100, step=5.0)
    loss_limit_percent = st.slider("Loss Limit (% of Session Bankroll)", 40.0, 60.0, st.session_state.loss_limit_percent * 100, step=5.0)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    if total_bankroll <= 0:
        st.error("Total bankroll must be positive.")
    elif session_bankroll <= 0:
        st.error("Session bankroll must be positive.")
    elif session_bankroll > total_bankroll:
        st.error("Session bankroll cannot exceed total bankroll.")
    elif base_bet <= 0:
        st.error("Base bet must be positive.")
    elif base_bet > session_bankroll * 0.02:
        st.error("Base bet cannot exceed 2% of session bankroll.")
    else:
        st.session_state.total_bankroll = total_bankroll
        st.session_state.initial_total_bankroll = total_bankroll
        st.session_state.session_bankroll = session_bankroll
        st.session_state.bankroll = session_bankroll
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
        st.session_state.initial_bankroll = session_bankroll
        st.session_state.target_hit = False
        st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
        st.session_state.consecutive_losses = 0
        st.session_state.loss_log = []
        st.session_state.last_was_tie = False
        st.session_state.recovery_mode = False
        st.session_state.insights = {}
        st.session_state.recovery_threshold = recovery_threshold
        st.session_state.recovery_bet_scale = recovery_bet_scale
        st.session_state.pattern_volatility = 0.0
        st.session_state.pattern_success = defaultdict(int)
        st.session_state.pattern_attempts = defaultdict(int)
        st.session_state.saved_profits = 0.0
        st.session_state.session_profit = 0.0
        st.session_state.session_active = True
        st.session_state.win_limit_percent = win_limit_percent / 100
        st.session_state.loss_limit_percent = loss_limit_percent / 100
        st.session_state.optimized_weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
        st.session_state.optimization_status = []
        st.session_state.dynamic_threshold = 52.0
        st.session_state.dynamic_bet_scale = 1.0
        st.session_state.cool_off_counter = 0
        st.success(f"Session started with {betting_strategy} strategy!")

# --- MANUAL OVERRIDE ---
st.subheader("Manual Bet Override")
manual_selection = st.radio("Override Next Bet", ["Auto", "Player", "Banker", "Skip"], index=0, horizontal=True)
if manual_selection == "Skip":
    st.session_state.pending_bet = None
    st.session_state.advice = "Bet skipped by user."
elif manual_selection in ["Player", "Banker"]:
    st.info(f"Next bet will be placed on {manual_selection}.")

# --- RESULT INPUT ---
st.subheader("Enter Result")
st.markdown("""
<style>
div.stButton > button {
    width: 90px;
    height: 35px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 6px;
    border: 1px solid;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
}
div.stButton > button:active {
    transform: scale(0.95);
    box-shadow: none;
}
div.stButton > button[kind="player_btn"] {
    background: linear-gradient(to bottom, #007bff, #0056b3);
    border-color: #0056b3;
    color: white;
}
div.stButton > button[kind="player_btn"]:hover {
    background: linear-gradient(to bottom, #339cff, #007bff);
}
div.stButton > button[kind="banker_btn"] {
    background: linear-gradient(to bottom, #dc3545, #a71d2a);
    border-color: #a71d2a;
    color: white;
}
div.stButton > button[kind="banker_btn"]:hover {
    background: linear-gradient(to bottom, #ff6666, #dc3545);
}
div.stButton > button[kind="tie_btn"] {
    background: linear-gradient(to bottom, #28a745, #1e7e34);
    border-color: #1e7e34;
    color: white;
}
div.stButton > button[kind="tie_btn"]:hover {
    background: linear-gradient(to bottom, #4caf50, #28a745);
}
div.stButton > button[kind="undo_btn"] {
    background: linear-gradient(to bottom, #6c757d, #545b62);
    border-color: #545b62;
    color: white;
}
div.stButton > button[kind="undo_btn"]:hover {
    background: linear-gradient(to bottom, #8e959c, #6c757d);
}
@media (max-width: 600px) {
    div.stButton > button {
        width: 80%;
        max-width: 150px;
        height: 40px;
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Player", key="player_btn"):
        place_result("P", manual_selection if manual_selection in ['P', 'B'] else None)
with col2:
    if st.button("Banker", key="banker_btn"):
        place_result("B", manual_selection if manual_selection in ['P', 'B'] else None)
with col3:
    if st.button("Tie", key="tie_btn"):
        place_result("T", manual_selection if manual_selection in ['P', 'B'] else None)
with col4:
    if st.button("Undo Last", key="undo_btn"):
        if not st.session_state.sequence:
            st.warning("No results to undo.")
        else:
            try:
                if st.session_state.history:
                    last = st.session_state.history.pop()
                    previous_state = last['Previous_State']
                    for key, value in previous_state.items():
                        st.session_state[key] = value
                    st.session_state.sequence.pop()
                    if last['Bet_Placed'] and not last['Win'] and st.session_state.loss_log:
                        if st.session_state.loss_log[-1]['result'] == last['Result']:
                            st.session_state.loss_log.pop()
                    if st.session_state.pending_bet:
                        amount, pred = st.session_state.pending_bet
                        conf = predict_next()[1]
                        st.session_state.advice = f"Next Bet: ${amount:.0f} on {pred} ({conf:.1f}%)"
                    else:
                        st.session_state.advice = "No bet pending."
                    st.session_state.last_was_tie = False
                    st.success("Undone last action.")
                    st.rerun()
                else:
                    st.session_state.sequence.pop()
                    st.session_state.pending_bet = None
                    st.session_state.advice = "No bet pending."
                    st.session_state.last_was_tie = False
                    st.success("Undone last result.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error undoing last action: {str(e)}")

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-90:] if 'sequence' in st.session_state else []
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
        elif result == 'T':
            col_html += "<div style='width: 20px; height: 20px; background-color: green; border-radius: 50%;'></div>"
    col_html += "</div>"
    bead_plate_html += col_html
bead_plate_html += "</div>"
st.markdown(bead_plate_html, unsafe_allow_html=True)

# --- PREDICTION DISPLAY ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    color = 'blue' if side == 'P' else 'red'
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.0f} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
else:
    if not st.session_state.target_hit:
        st.info(st.session_state.advice)

# --- PREDICTION INSIGHTS ---
st.subheader("Prediction Insights")
if st.session_state.insights:
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"**{factor}**: {contribution}")
if st.session_state.recovery_mode:
    st.warning("Recovery Mode: Reduced bet sizes due to significant losses.")
if st.session_state.pattern_volatility > 0.6:
    st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Betting paused)")

# --- OPTIMIZATION STATUS ---
st.subheader("Optimization Status")
if st.session_state.optimization_status:
    for status in st.session_state.optimization_status[-5:]:
        st.markdown(f"- {status}")
st.markdown(f"**Current Threshold**: {st.session_state.dynamic_threshold:.1f}%")
st.markdown(f"**Current Bet Scale**: {st.session_state.dynamic_bet_scale:.2f}")

# --- UNIT PROFIT ---
if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    units_profit = profit / st.session_state.initial_base_bet
    st.markdown(f"**Units Profit**: {units_profit:.2f} units (${profit:.2f})")
else:
    st.markdown("**Units Profit**: 0.00 units ($0.00)")

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Total Bankroll**: ${st.session_state.total_bankroll:.2f}")
st.markdown(f"**Session Bankroll**: ${st.session_state.session_bankroll:.2f}")
st.markdown(f"**Current Session Balance**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Saved Profits**: ${st.session_state.saved_profits:.2f}")
st.markdown(f"**Session Profit/Loss**: ${st.session_state.session_profit:.2f}")
st.markdown(f"**Win Limit**: ${st.session_state.session_bankroll * st.session_state.win_limit_percent:.2f} ({st.session_state.win_limit_percent*100:.0f}%)")
st.markdown(f"**Loss Limit**: ${st.session_state.session_bankroll * st.session_state.loss_limit_percent:.2f} ({st.session_state.loss_limit_percent*100:.0f}%)")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
strategy_status = f"**Betting Strategy**: {st.session_state.strategy}"
if st.session_state.strategy == 'T3':
    strategy_status += f" | T3 Level: {st.session_state.t3_level} | Level Changes: {st.session_state.t3_level_changes}"
elif st.session_state.strategy == 'Parlay16':
    strategy_status += f" | Parlay Step: {st.session_state.parlay_step}/16 | Step Changes: {st.session_state.parlay_step_changes} | Consecutive Wins: {st.session_state.parlay_wins}"
st.markdown(strategy_status)
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
online_users = track_user_session_file()
st.markdown(f"**Online Users**: {online_users}")
if not st.session_state.session_active:
    st.error("Session paused due to win/loss limit. Start a new session to continue.")

# --- PREDICTION ACCURACY ---
st.subheader("Prediction Accuracy")
total = st.session_state.prediction_accuracy['total']
if total > 0:
    p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
    b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
    st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
    st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

# --- PREDICTION ACCURACY CHART ---
st.subheader("Prediction Accuracy Trend")
if st.session_state.history:
    accuracy_data = []
    correct = 0
    total = 0
    for h in st.session_state.history[-50:]:
        if h['Bet_Placed'] and h['Bet'] in ['P', 'B']:
            total += 1
            if h['Win']:
                correct += 1
            accuracy_data.append(correct / max(total, 1) * 100)
    if accuracy_data:
        st.line_chart(accuracy_data, use_container_width=True)

# --- LOSS LOG ---
if st.session_state.loss_log:
    st.subheader("Recent Losses")
    st.dataframe([
        {
            "Sequence": ", ".join(log['sequence']),
            "Prediction": log['prediction'],
            "Result": log['result'],
            "Confidence": log['confidence'] + "%",
            "Insights": "; ".join([f"{k}: {v}" for k, v in log['insights'].items()])
        }
        for log in st.session_state.loss_log[-5:]
    ])

# --- HISTORY TABLE ---
if st.session_state.history:
    st.subheader("Bet History")
    n = st.slider("Show last N bets", 5, 50, 10)
    st.dataframe([
        {
            "Bet": h["Bet"] if h["Bet"] else "-",
            "Result": h["Result"],
            "Amount": f"${h['Amount']:.0f}" if h["Bet_Placed"] else "-",
            "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-",
            "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-",
            "Parlay_Step": h["Parlay_Step"] if st.session_state.strategy == 'Parlay16' else "-",
            "Session_Profit": f"${h['Session_Profit']:.2f}"
        }
        for h in st.session_state.history[-n:]
    ])

# --- EXPORT SESSION ---
st.subheader("Export Session")
if st.button("Download Session Data"):
    csv_data = "Bet,Result,Amount,Win,T3_Level,Parlay_Step,Session_Profit\n"
    for h in st.session_state.history:
        csv_data += f"{h['Bet'] or '-'},{h['Result']},${h['Amount']:.0f},{h['Win']},{h['T3_Level']},{h['Parlay_Step']},${h['Session_Profit']:.2f}\n"
    st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")
