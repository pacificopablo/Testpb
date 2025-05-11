import streamlit as st
import numpy as np
from collections import defaultdict
import time
import os
from datetime import datetime

# --- INITIALIZE SESSION STATE ---
if 'total_bankroll' not in st.session_state:
    st.session_state.total_bankroll = 100.0
if 'initial_total_bankroll' not in st.session_state:
    st.session_state.initial_total_bankroll = 100.0
if 'session_bankroll' not in st.session_state:
    st.session_state.session_bankroll = 10.0
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 10.0
if 'base_bet' not in st.session_state:
    st.session_state.base_bet = 1.0
if 'initial_base_bet' not in st.session_state:
    st.session_state.initial_base_bet = 1.0
if 'strategy' not in st.session_state:
    st.session_state.strategy = 'Flatbet'
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'pending_bet' not in st.session_state:
    st.session_state.pending_bet = None
if 't3_level' not in st.session_state:
    st.session_state.t3_level = 1
if 't3_results' not in st.session_state:
    st.session_state.t3_results = []
if 't3_level_changes' not in st.session_state:
    st.session_state.t3_level_changes = 0
if 'parlay_step' not in st.session_state:
    st.session_state.parlay_step = 1
if 'parlay_wins' not in st.session_state:
    st.session_state.parlay_wins = 0
if 'parlay_using_base' not in st.session_state:
    st.session_state.parlay_using_base = True
if 'parlay_step_changes' not in st.session_state:
    st.session_state.parlay_step_changes = 0
if 'advice' not in st.session_state:
    st.session_state.advice = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'wins' not in st.session_state:
    st.session_state.wins = 0
if 'losses' not in st.session_state:
    st.session_state.losses = 0
if 'target_mode' not in st.session_state:
    st.session_state.target_mode = "Profit %"
if 'target_value' not in st.session_state:
    st.session_state.target_value = 10.0
if 'initial_bankroll' not in st.session_state:
    st.session_state.initial_bankroll = 10.0
if 'target_hit' not in st.session_state:
    st.session_state.target_hit = False
if 'prediction_accuracy' not in st.session_state:
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
if 'consecutive_losses' not in st.session_state:
    st.session_state.consecutive_losses = 0
if 'loss_log' not in st.session_state:
    st.session_state.loss_log = []
if 'last_was_tie' not in st.session_state:
    st.session_state.last_was_tie = False
if 'recovery_mode' not in st.session_state:
    st.session_state.recovery_mode = False  # Already disabled
if 'insights' not in st.session_state:
    st.session_state.insights = {}
if 'pattern_volatility' not in st.session_state:
    st.session_state.pattern_volatility = 0.0
if 'pattern_success' not in st.session_state:
    st.session_state.pattern_success = defaultdict(int)
if 'pattern_attempts' not in st.session_state:
    st.session_state.pattern_attempts = defaultdict(int)
if 'saved_profits' not in st.session_state:
    st.session_state.saved_profits = 0.0
if 'session_profit' not in st.session_state:
    st.session_state.session_profit = 0.0
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'optimized_weights' not in st.session_state:
    st.session_state.optimized_weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
if 'optimization_status' not in st.session_state:
    st.session_state.optimization_status = []
if 'dynamic_threshold' not in st.session_state:
    st.session_state.dynamic_threshold = 53.0
if 'cool_off_counter' not in st.session_state:
    st.session_state.cool_off_counter = 0
if 'last_bet_outcomes' not in st.session_state:
    st.session_state.last_bet_outcomes = {'P': [], 'B': []}
if 'pattern_accuracy' not in st.session_state:
    st.session_state.pattern_accuracy = defaultdict(lambda: {'success': 0, 'attempts': 0})
if 'pattern_transitions' not in st.session_state:
    st.session_state.pattern_transitions = defaultdict(lambda: defaultdict(int))
if 'last_pattern' not in st.session_state:
    st.session_state.last_pattern = None

# Removed: win_limit_percent, loss_limit_percent, recovery_threshold, recovery_bet_scale from session state

# --- PARLAY TABLE ---
PARLAY_TABLE = {
    1: {'base': 1.0, 'parlay': 1.0},
    2: {'base': 1.0, 'parlay': 2.0},
    3: {'base': 1.0, 'parlay': 4.0},
    4: {'base': 1.0, 'parlay': 8.0},
    5: {'base': 1.0, 'parlay': 16.0},
    6: {'base': 1.0, 'parlay': 32.0},
    7: {'base': 1.0, 'parlay': 64.0},
    8: {'base': 1.0, 'parlay': 128.0},
    9: {'base': 1.0, 'parlay': 256.0},
    10: {'base': 1.0, 'parlay': 512.0},
    11: {'base': 1.0, 'parlay': 1024.0},
    12: {'base': 1.0, 'parlay': 2048.0},
    13: {'base': 1.0, 'parlay': 4096.0},
    14: {'base': 1.0, 'parlay': 8192.0},
    15: {'base': 1.0, 'parlay': 16384.0},
    16: {'base': 1.0, 'parlay': 32768.0}
}

# --- CHECK TARGET HIT ---
def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.session_profit >= target
    else:  # Units
        return st.session_state.wins - st.session_state.losses >= st.session_state.target_value

# --- RESET SESSION ---
def reset_session_auto():
    st.session_state.session_active = False
    st.session_state.target_hit = False
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
    st.session_state.session_profit = 0.0
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.recovery_mode = False  # Already disabled
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.pattern_success = defaultdict(int)
    st.session_state.pattern_attempts = defaultdict(int)
    st.session_state.cool_off_counter = 0
    st.session_state.last_bet_outcomes = {'P': [], 'B': []}
    st.session_state.pattern_accuracy = defaultdict(lambda: {'success': 0, 'attempts': 0})
    st.session_state.pattern_transitions = defaultdict(lambda: defaultdict(int))
    st.session_state.last_pattern = None
    st.success("Session reset automatically.")

# --- AUTO-OPTIMIZE WEIGHTS ---
def auto_optimize_weights():
    if len(st.session_state.sequence) < 50:
        return
    recent_sequence = [x for x in st.session_state.sequence[-50:] if x in ['P', 'B']]
    if len(recent_sequence) < 10:
        return

    patterns = ['bigram', 'trigram', 'streak', 'chop', 'double']
    success_rates = {}
    for pattern in patterns:
        attempts = st.session_state.pattern_attempts[pattern]
        success = st.session_state.pattern_success[pattern]
        success_rates[pattern] = success / max(attempts, 1) if attempts > 0 else 0.5

    current_weights = st.session_state.optimized_weights.copy()
    total_weight = sum(current_weights.values())
    baseline = 0.2 / total_weight  # Adjust to maintain sum of weights

    for pattern in patterns:
        adjustment = (success_rates[pattern] - 0.5) * 0.1  # Scale adjustment based on deviation from 0.5
        current_weights[pattern] = max(0.05, min(0.5, current_weights[pattern] + adjustment))
        if current_weights[pattern] < 0.05:
            current_weights[pattern] = 0.05  # Minimum weight to ensure all patterns are considered

    # Normalize weights
    total_new_weight = sum(current_weights.values())
    for pattern in patterns:
        current_weights[pattern] = current_weights[pattern] / total_new_weight

    st.session_state.optimized_weights = current_weights
    st.session_state.optimization_status.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'weights': current_weights.copy(),
        'success_rates': success_rates.copy()
    })
    if len(st.session_state.optimization_status) > 10:
        st.session_state.optimization_status = st.session_state.optimization_status[-10:]

# --- PREDICTION FUNCTION ---
def predict_next(sequence, pattern_volatility, weights, prediction_accuracy, consecutive_losses, pattern_accuracy, pattern_transitions, last_pattern):
    # Include ties in the sequence for analysis
    cleaned_sequence = [x for x in sequence if x in ['P', 'B']]
    if len(cleaned_sequence) < 3:
        return 'B', 45.86, {}

    window_size = 50
    recent_sequence = cleaned_sequence[-window_size:]
    full_sequence = sequence[-window_size:]  # Include ties for tie influence

    bigram_transitions = defaultdict(lambda: defaultdict(int))
    trigram_transitions = defaultdict(lambda: defaultdict(int))
    pattern_transitions = pattern_transitions
    streak_count = 0
    current_streak = None
    chop_count = 0
    double_count = 0
    insights = {}
    pattern_changes = 0
    last_pattern = last_pattern
    current_pattern = None

    # Pattern detection
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
                # Update transition matrix
                pattern_transitions[last_pattern][current_pattern] += 1
            last_pattern = current_pattern

    pattern_volatility = pattern_changes / max(len(recent_sequence) - 2, 1)

    prior_p = 44.62 / 100
    prior_b = 45.86 / 100

    total_w = sum(weights.values())
    for k in weights:
        weights[k] = max(weights[k] / total_w, 0.05)

    prob_p = 0
    prob_b = 0
    total_weight = 0

    # Bigram pattern
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

    # Trigram pattern
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

    # Streak pattern
    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.7 if streak_count > 4 else 1.0)
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"

    # Chop pattern
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

    # Double pattern
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
        prob_p = 44.62
        prob_b = 45.86

    if abs(prob_p - prob_b) < 2:
        prob_p += 0.5
        prob_b -= 0.5

    # Contextual matchup analysis (hotness boost)
    recent_wins = defaultdict(int)
    for result in cleaned_sequence[-10:]:
        if result in ['P', 'B']:
            recent_wins[result] += 1
    if len(cleaned_sequence[-10:]) >= 5:
        if recent_wins['P'] / 10 >= 0.7 and pattern_volatility < 0.5:
            prob_p = min(75.0, prob_p + 5.0)
            insights['Hotness Boost'] = "Player (+5%)"
        elif recent_wins['B'] / 10 >= 0.7 and pattern_volatility < 0.5:
            prob_b = min(75.0, prob_b + 5.0)
            insights['Hotness Boost'] = "Banker (+5%)"

    # Trend momentum adjustment
    if streak_count >= 3:
        momentum_boost = min(15.0, (streak_count - 2) * 3.0)  # 3% per streak length beyond 2, capped at 15%
        if current_streak == 'P':
            prob_p = min(80.0, prob_p + momentum_boost)
            prob_b = max(20.0, prob_b - momentum_boost / 2)
        else:
            prob_b = min(80.0, prob_b + momentum_boost)
            prob_p = max(20.0, prob_p - momentum_boost / 2)
        insights['Momentum Boost'] = f"{current_streak} (+{momentum_boost:.1f}%)"

    # Pattern transition probability
    if last_pattern and current_pattern:
        total_trans = sum(pattern_transitions[last_pattern].values())
        if total_trans > 0:
            transition_probs = {next_pattern: pattern_transitions[last_pattern][next_pattern] / total_trans for next_pattern in pattern_transitions[last_pattern]}
            if current_pattern == 'streak' and 'chop' in transition_probs and transition_probs['chop'] >= 0.7:
                next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
                if next_pred == 'P':
                    prob_p += 10.0
                    prob_b -= 5.0
                else:
                    prob_b += 10.0
                    prob_p -= 5.0
                insights['Transition Adjustment'] = f"Streak to Chop (+10% for {next_pred})"
            elif current_pattern == 'chop' and 'streak' in transition_probs and transition_probs['streak'] >= 0.7:
                if recent_sequence[-1] == 'P':
                    prob_p += 10.0
                    prob_b -= 5.0
                else:
                    prob_b += 10.0
                    prob_p -= 5.0
                insights['Transition Adjustment'] = f"Chop to Streak (+10% for {recent_sequence[-1]})"

    # Tie influence factor
    if len(full_sequence) >= 2 and full_sequence[-1] == 'T':
        if streak_count >= 3:
            # Tie after a streak may signal a break
            if current_streak == 'P':
                prob_p = max(20.0, prob_p - 5.0)
                prob_b += 2.5
            else:
                prob_b = max(20.0, prob_b - 5.0)
                prob_p += 2.5
            insights['Tie Influence'] = "Streak Break (-5%)"
        elif chop_count >= 2:
            # Tie after a chop may reinforce alternation
            next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
            if next_pred == 'P':
                prob_p += 3.0
                prob_b -= 1.5
            else:
                prob_b += 3.0
                prob_p -= 1.5
            insights['Tie Influence'] = f"Chop Continuation (+3% for {next_pred})"

    # Pattern confidence score with conflict penalty
    pattern_conflicts = abs((prob_p - prob_b) / max(prob_p, prob_b)) if max(prob_p, prob_b) > 0 else 0
    if pattern_conflicts > 0.2:  # Significant conflict
        confidence_penalty = min(5.0, pattern_conflicts * 10)
        prob_p = max(50.0, prob_p - confidence_penalty)
        prob_b = max(50.0, prob_b - confidence_penalty)
        insights['Conflict Penalty'] = f"-{confidence_penalty:.1f}%"

    # Dynamic confidence threshold based on pattern strength
    recent_accuracy = (prediction_accuracy['P'] + prediction_accuracy['B']) / max(prediction_accuracy['total'], 1)
    threshold = st.session_state.dynamic_threshold + (consecutive_losses * 1.0) - (recent_accuracy * 1.5)
    threshold = min(max(threshold, 50.0), 65.0)
    dominant_pattern = max(weights, key=lambda k: weights[k] if k in insights else 0)
    if pattern_accuracy[dominant_pattern]['attempts'] > 10:
        pattern_acc = pattern_accuracy[dominant_pattern]['success'] / pattern_accuracy[dominant_pattern]['attempts']
        if pattern_acc >= 0.75 and pattern_volatility < 0.5:
            threshold = max(50.0, threshold - 2.0)
            insights['Threshold Adjustment'] = f"Lowered to {threshold:.1f}% (Strong {dominant_pattern})"

    # Final prediction
    if prob_p > prob_b and prob_p >= threshold:
        return 'P', prob_p, insights, last_pattern
    elif prob_b >= threshold:
        return 'B', prob_b, insights, last_pattern
    else:
        return None, max(prob_p, prob_b), insights, last_pattern

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

    # Recovery mode is already disabled
    st.session_state.recovery_mode = False

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
        "cool_off_counter": st.session_state.cool_off_counter,
        "last_bet_outcomes": st.session_state.last_bet_outcomes.copy(),
        "pattern_accuracy": st.session_state.pattern_accuracy.copy(),
        "pattern_transitions": st.session_state.pattern_transitions.copy(),
        "last_pattern": st.session_state.last_pattern
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
            st.session_state.last_bet_outcomes[selection] = [1] + st.session_state.last_bet_outcomes[selection][:2]
            for pattern in ['bigram', 'trigram', 'streak', 'chop', 'double']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_success[pattern] += 1
                    st.session_state.pattern_attempts[pattern] += 1
                    st.session_state.pattern_accuracy[pattern]['success'] += 1
                    st.session_state.pattern_accuracy[pattern]['attempts'] += 1
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
            st.session_state.last_bet_outcomes[selection] = [0] + st.session_state.last_bet_outcomes[selection][:2]
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
                    st.session_state.pattern_accuracy[pattern]['attempts'] += 1
        st.session_state.prediction_accuracy['total'] += 1
        st.session_state.pending_bet = None

        # Manage profit/loss
        session_profit = st.session_state.bankroll - previous_bankroll
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

        # Win/loss limit checks are already removed

        if st.session_state.total_bankroll < st.session_state.initial_total_bankroll * 0.8:
            new_session_bankroll = max(st.session_state.total_bankroll / 10, 10.0)
            st.session_state.session_bankroll = new_session_bankroll
            st.session_state.base_bet = max(new_session_bankroll * 0.02, 1.0)
            st.session_state.initial_base_bet = st.session_state.base_bet
            st.warning(f"Bankroll low: ${st.session_state.total_bankroll:.2f}. Adjusted session bankroll to ${new_session_bankroll:.2f}, base bet to ${st.session_state.base_bet:.2f}")

    # Append to sequence
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    # Auto-optimize weights for bet selection
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

    # Compute prob_variance for volatility check
    prob_history = []
    temp_sequence = st.session_state.sequence.copy()
    temp_last_pattern = st.session_state.last_pattern
    for _ in range(min(20, len(temp_sequence))):
        pred, conf, insights, temp_last_pattern = predict_next(temp_sequence, st.session_state.pattern_volatility, st.session_state.optimized_weights,
                                                              st.session_state.prediction_accuracy, st.session_state.consecutive_losses,
                                                              st.session_state.pattern_accuracy, st.session_state.pattern_transitions, temp_last_pattern)
        if pred:
            prob_history.append(conf)
        if temp_sequence:
            temp_sequence.pop()  # Simulate past predictions
    prob_variance = np.std(prob_history) if prob_history else 0

    # Update pattern volatility
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    pattern_changes = 0
    last_pattern = None
    for i in range(len(sequence) - 1):
        if i < len(sequence) - 2:
            current_pattern = 'streak' if (sequence[i] == sequence[i-1] and sequence[i-1] == sequence[i-2]) else 'chop' if (sequence[i] != sequence[i-1] and sequence[i-1] != sequence[i-2]) else 'other'
            if last_pattern and last_pattern != current_pattern:
                pattern_changes += 1
            last_pattern = current_pattern
    st.session_state.pattern_volatility = pattern_changes / max(len(sequence) - 2, 1)

    # Calculate next bet
    pred, conf, insights, st.session_state.last_pattern = predict_next(st.session_state.sequence, st.session_state.pattern_volatility,
                                                                      st.session_state.optimized_weights, st.session_state.prediction_accuracy,
                                                                      st.session_state.consecutive_losses, st.session_state.pattern_accuracy,
                                                                      st.session_state.pattern_transitions, st.session_state.last_pattern)
    if prob_variance > 15.0:
        insights['Volatility (Variance)'] = f"High (Adjustment: +3% threshold)"
    if manual_selection in ['P', 'B']:
        pred = manual_selection
        conf = max(conf, 50.0)
        st.session_state.advice = f"Manual Bet: {pred} (User override)"

    # Bet sizing (recovery mode scaling already removed)
    bet_scaling = 1.0
    if st.session_state.consecutive_losses >= 2:
        bet_scaling *= 0.8
    if conf < 55.0:
        bet_scaling *= 0.9

    # Secondary validation check with pattern shift reset
    if pred:
        # Check for pattern shift to reset loss streak penalty
        recent_sequence = [x for x in st.session_state.sequence[-10:] if x in ['P', 'B']]
        streak_count = 0
        chop_count = 0
        for i in range(len(recent_sequence) - 1):
            if recent_sequence[i] == recent_sequence[i-1]:
                streak_count += 1
            else:
                if i > 1 and recent_sequence[i] != recent_sequence[i-2]:
                    chop_count += 1
        current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'other'
        if current_pattern != st.session_state.last_pattern and st.session_state.last_pattern:
            st.session_state.last_bet_outcomes[pred] = []  # Reset loss streak on pattern shift
            insights['Loss Streak Reset'] = f"Pattern shift to {current_pattern}"

        if len(st.session_state.last_bet_outcomes[pred]) >= 3 and sum(st.session_state.last_bet_outcomes[pred][-3:]) == 0 and conf < 70.0:
            st.session_state.pending_bet = None
            st.session_state.advice = f"No bet: {pred} lost last 3 times (Confidence: {conf:.1f}% < 70%)"
            st.session_state.insights = insights
            return

    # Stricter loss streak pause
    if st.session_state.consecutive_losses >= 2 and conf < 65.0:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet: Paused after {st.session_state.consecutive_losses} losses (Confidence: {conf:.1f}% < 65%)"
        st.session_state.insights = insights
        return

    # Stricter volatility check
    if st.session_state.pattern_volatility > 0.6 or prob_variance > 15.0:
        st.session_state.advice = f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f} or variance {prob_variance:.1f})"
        st.session_state.pending_bet = None
        st.session_state.insights = insights
        return

    if pred is None or conf < 53.0:
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

        # Bankroll-aware filtering (unchanged)
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

# --- TRACK USER SESSION ---
def track_user_session_file():
    session_file = "session_count.txt"
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            count = int(f.read().strip())
    else:
        count = 0
    count += 1
    with open(session_file, "w") as f:
        f.write(str(count))
    return count

# --- MAIN APP ---
st.title("Baccarat Bet Predictor")

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
    # Removed: recovery_threshold, recovery_bet_scale, win_limit_percent, loss_limit_percent
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0 if st.session_state.target_mode == "Profit %" else 1, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
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
        st.session_state.pattern_volatility = 0.0
        st.session_state.pattern_success = defaultdict(int)
        st.session_state.pattern_attempts = defaultdict(int)
        st.session_state.saved_profits = 0.0
        st.session_state.session_profit = 0.0
        st.session_state.session_active = True
        st.session_state.optimized_weights = {'bigram': 0.4, 'trigram': 0.3, 'streak': 0.2, 'chop': 0.05, 'double': 0.05}
        st.session_state.optimization_status = []
        st.session_state.dynamic_threshold = 53.0
        st.session_state.cool_off_counter = 0
        st.session_state.last_bet_outcomes = {'P': [], 'B': []}
        st.session_state.pattern_accuracy = defaultdict(lambda: {'success': 0, 'attempts': 0})
        st.session_state.pattern_transitions = defaultdict(lambda: defaultdict(int))
        st.session_state.last_pattern = None
        st.success(f"Session started with {betting_strategy} strategy!")

# --- INPUT RESULT ---
st.subheader("Input Result")
if st.session_state.session_active:
    result = st.selectbox("Select Result", ["P", "B", "T"], index=0 if not st.session_state.last_was_tie else 2)
    manual_selection = st.selectbox("Manual Bet Override", ["None", "P", "B"], index=0)
    if st.button("Submit Result"):
        place_result(result, manual_selection if manual_selection != "None" else None)
        st.experimental_rerun()

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Total Bankroll**: ${st.session_state.total_bankroll:.2f}")
st.markdown(f"**Session Bankroll**: ${st.session_state.session_bankroll:.2f}")
st.markdown(f"**Current Session Balance**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Saved Profits**: ${st.session_state.saved_profits:.2f}")
st.markdown(f"**Session Profit/Loss**: ${st.session_state.session_profit:.2f}")
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

# --- PREDICTION INSIGHTS ---
st.subheader("Prediction Insights")
if st.session_state.insights:
    for factor, contribution in st.session_state.insights.items():
        st.markdown(f"**{factor}**: {contribution}")
if st.session_state.pattern_volatility > 0.6:
    st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Betting paused)")

# --- HISTORY ---
st.subheader("History (Last 10 Hands)")
history_display = st.session_state.history[-10:][::-1] if st.session_state.history else []
for entry in history_display:
    bet = entry["Bet"] or "None"
    result = entry["Result"]
    amount = f"${entry['Amount']:.0f}" if entry["Amount"] else "N/A"
    win = "✅" if entry["Win"] else "❌"
    st.markdown(f"**Bet**: {bet} | **Result**: {result} | **Amount**: {amount} | **Win**: {win} | **Profit**: ${entry['Session_Profit']:.2f}")

# --- LOSS LOG ---
st.subheader("Loss Log (Last 5 Losses)")
loss_display = st.session_state.loss_log[-5:] if st.session_state.loss_log else []
for loss in loss_display:
    st.json({
        "Sequence": ''.join(loss['sequence']),
        "Prediction": loss['prediction'],
        "Result": loss['result'],
        "Confidence": f"{loss['confidence']}%",
        "Insights": loss['insights']
    })

# --- OPTIMIZATION STATUS ---
st.subheader("Optimization Status (Last 10 Adjustments)")
for status in st.session_state.optimization_status[::-1]:
    st.markdown(f"**Time**: {status['time']} | **Weights**: {status['weights']} | **Success Rates**: {status['success_rates']}")
