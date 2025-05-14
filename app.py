import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import tempfile
import logging
import traceback

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
PARLAY_TABLE = [1, 1, 1, 2, 3, 4, 6, 8, 12, 16, 22, 30, 40, 52, 70, 95]  # Parlay16 sequence
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
APP_VERSION = "2025-05-14-fix-v11"

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# --- Session Tracking ---
def track_user_session() -> int:
    """Track active user sessions with fallback for file errors."""
    logging.debug("Entering track_user_session")
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
                        logging.warning(f"Invalid session file line: {line}")
                        continue
    except (PermissionError, OSError) as e:
        logging.error(f"Session file read error: {str(e)}")
        st.warning("Session tracking unavailable. Using fallback.")
        return 1

    sessions[st.session_state.session_id] = current_time

    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except (PermissionError, OSError) as e:
        logging.error(f"Session file write error: {str(e)}")
        st.warning("Session tracking may be inaccurate.")
        return len(sessions)

    logging.debug(f"track_user_session: {len(sessions)} active sessions")
    return len(sessions)

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state with default values."""
    logging.debug("Entering initialize_session_state")
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
        't3_peak_level': 1,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'z1003_loss_count': 0,
        'z1003_bet_factor': 1.0,
        'z1003_continue': False,
        'z1003_level_changes': 0,
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
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'last_win_confidence': 0.0,
        'recent_pattern_accuracy': defaultdict(float),
        'consecutive_wins': 0,
    }
    defaults['pattern_success']['fourgram'] = 0
    defaults['pattern_attempts']['fourgram'] = 0
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'T3'
    logging.debug("initialize_session_state completed")

def reset_session():
    """Reset session state to initial values."""
    logging.debug("Entering reset_session")
    for key in list(st.session_state.keys()):
        if key != 'session_id':
            del st.session_state[key]
    initialize_session_state()
    st.session_state.update({
        'bankroll': 0.0,
        'sequence': [],
        'pending_bet': None,
        't3_level': 1,
        't3_results': [],
        't3_level_changes': 0,
        't3_peak_level': 1,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'z1003_loss_count': 0,
        'z1003_bet_factor': 1.0,
        'z1003_continue': False,
        'z1003_level_changes': 0,
        'advice': "Session reset.",
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
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'last_win_confidence': 0.0,
        'consecutive_wins': 0,
    })
    logging.debug("reset_session completed")

# --- Prediction Logic ---
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, Dict, int, int, int, float, float, Dict]:
    """Analyze sequence patterns with streak and chop metrics."""
    logging.debug("Entering analyze_patterns")
    try:
        bigram_transitions = defaultdict(lambda: defaultdict(int))
        trigram_transitions = defaultdict(lambda: defaultdict(int))
        fourgram_transitions = defaultdict(lambda: defaultdict(int))
        pattern_transitions = defaultdict(lambda: defaultdict(int))
        streak_count = chop_count = double_count = pattern_changes = 0
        current_streak = last_pattern = None
        player_count = banker_count = 0
        streak_lengths = []
        chop_lengths = []

        filtered_sequence = [x for x in sequence if x in ['P', 'B']]
        for i in range(len(sequence) - 1):
            if sequence[i] == 'P':
                player_count += 1
            elif sequence[i] == 'B':
                banker_count += 1

            if i < len(sequence) - 2:
                bigram = tuple(sequence[i:i+2])
                trigram = tuple(sequence[i:i+3])
                next_outcome = sequence[i+2]
                bigram_transitions[bigram][next_outcome] += 1
                if i < len(sequence) - 3:
                    trigram_transitions[trigram][next_outcome] += 1
                    if i < len(sequence) - 4:
                        fourgram = tuple(sequence[i:i+4])
                        fourgram_transitions[fourgram][next_outcome] += 1

        current_streak_length = 0
        current_chop_length = 0
        for i in range(1, len(filtered_sequence)):
            if filtered_sequence[i] == filtered_sequence[i-1]:
                if current_streak == filtered_sequence[i]:
                    current_streak_length += 1
                else:
                    if current_streak_length > 1:
                        streak_lengths.append(current_streak_length)
                    current_streak = filtered_sequence[i]
                    current_streak_length = 1
                if i > 1 and filtered_sequence[i-1] == filtered_sequence[i-2]:
                    double_count += 1
                if current_chop_length > 1:
                    chop_lengths.append(current_chop_length)
                    current_chop_length = 0
            else:
                current_streak = None
                if current_streak_length > 1:
                    streak_lengths.append(current_streak_length)
                current_streak_length = 0
                if i > 1 and filtered_sequence[i] != filtered_sequence[i-2]:
                    current_chop_length += 1
                    chop_count += 1
                else:
                    if current_chop_length > 1:
                        chop_lengths.append(current_chop_length)
                    current_chop_length = 0

            if i < len(filtered_sequence) - 1:
                current_pattern = (
                    'streak' if current_streak_length >= 2 else
                    'chop' if chop_count >= 2 else
                    'double' if double_count >= 1 else 'other'
                )
                if last_pattern and last_pattern != current_pattern:
                    pattern_changes += 1
                last_pattern = current_pattern
                next_outcome = filtered_sequence[i+1]
                pattern_transitions[current_pattern][next_outcome] += 1

        if current_streak_length > 1:
            streak_lengths.append(current_streak_length)
        if current_chop_length > 1:
            chop_lengths.append(current_chop_length)

        volatility = pattern_changes / max(len(filtered_sequence) - 2, 1)
        total_outcomes = max(player_count + banker_count, 1)
        shoe_bias = player_count / total_outcomes if player_count > banker_count else -banker_count / total_outcomes

        extra_metrics = {
            'avg_streak_length': sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0,
            'avg_chop_length': sum(chop_lengths) / len(chop_lengths) if chop_lengths else 0,
            'streak_frequency': len(streak_lengths) / max(len(filtered_sequence), 1),
            'chop_frequency': len(chop_lengths) / max(len(filtered_sequence), 1)
        }

        logging.debug("analyze_patterns completed")
        return (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
                streak_count, chop_count, double_count, volatility, shoe_bias, extra_metrics)
    except Exception as e:
        logging.error(f"analyze_patterns error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error analyzing patterns. Try resetting the session.")
        return ({}, {}, {}, {}, 0, 0, 0, 0.0, 0.0, {})

def calculate_weights(streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> Dict[str, float]:
    """Calculate adaptive weights with error handling."""
    logging.debug("Entering calculate_weights")
    try:
        total_bets = max(st.session_state.pattern_attempts.get('fourgram', 1), 1)
        success_ratios = {
            'bigram': st.session_state.pattern_success.get('bigram', 0) / total_bets,
            'trigram': st.session_state.pattern_success.get('trigram', 0) / total_bets,
            'fourgram': st.session_state.pattern_success.get('fourgram', 0) / total_bets,
            'streak': 0.6 if streak_count >= 2 else 0.3,
            'chop': 0.4 if chop_count >= 2 else 0.2,
            'double': 0.4 if double_count >= 1 else 0.2
        }

        recent_bets = st.session_state.history[-10:]
        recent_success = defaultdict(int)
        recent_attempts = defaultdict(int)
        for h in recent_bets:
            if h['Bet_Placed'] and h['Bet'] in ['P', 'B']:
                for pattern in h.get('Previous_State', {}).get('insights', {}):
                    recent_attempts[pattern] += 1
                    if h['Win']:
                        recent_success[pattern] += 1
        for pattern in success_ratios:
            if recent_attempts[pattern] > 0:
                recent_ratio = recent_success[pattern] / recent_attempts[pattern]
                if recent_ratio > 0.7:
                    success_ratios[pattern] *= 1.5
                elif recent_ratio < 0.3:
                    success_ratios[pattern] *= 0.6

        if success_ratios['fourgram'] > 0.6:
            success_ratios['fourgram'] *= 1.3

        weights = {k: np.exp(v) / (1 + np.exp(v)) for k, v in success_ratios.items()}
        if shoe_bias > 0.1:
            weights['bigram'] *= 1.1
            weights['trigram'] *= 1.1
            weights['fourgram'] *= 1.15
        elif shoe_bias < -0.1:
            weights['bigram'] *= 0.9
            weights['trigram'] *= 0.9
            weights['fourgram'] *= 0.85

        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}
            total_weight = sum(weights.values())

        normalized_weights = {k: max(v / total_weight, 0.05) for k, v in weights.items()}

        dominant_pattern = max(normalized_weights, key=normalized_weights.get)
        st.session_state.insights['Dominant Pattern'] = {
            'pattern': dominant_pattern,
            'weight': normalized_weights[dominant_pattern] * 100
        }

        logging.debug("calculate_weights completed")
        return normalized_weights
    except NameError as e:
        logging.error(f"NameError in calculate_weights: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Variable error in weight calculation: {str(e)}. Try resetting the session.")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}
    except Exception as e:
        logging.error(f"calculate_weights error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error calculating weights. Try resetting the session.")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}

def predict_next() -> Tuple[Optional[str], float, Dict]:
    """Predict the next outcome with error handling."""
    logging.debug("Entering predict_next")
    try:
        sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
        shadow_sequence = [x for x in sequence if x in ['P', 'B']]
        if len(shadow_sequence) < 4:
            return 'B', 45.86, {'Initial': 'Default to Banker (insufficient data)'}

        recent_sequence = shadow_sequence[-WINDOW_SIZE:]
        (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
         streak_count, chop_count, double_count, volatility, shoe_bias, extra_metrics) = analyze_patterns(recent_sequence)
        st.session_state.pattern_volatility = volatility

        prior_p, prior_b = 44.62 / 100, 45.86 / 100
        weights = calculate_weights(streak_count, chop_count, double_count, shoe_bias)
        prob_p = prob_b = total_weight = 0
        insights = {}
        pattern_reliability = {}
        recent_performance = {}

        recent_bets = st.session_state.history[-10:]
        for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
            success = sum(1 for h in recent_bets if h['Bet_Placed'] and h['Win'] and pattern in h.get('Previous_State', {}).get('insights', {}))
            attempts = sum(1 for h in recent_bets if h['Bet_Placed'] and pattern in h.get('Previous_State', {}).get('insights', {}))
            recent_performance[pattern] = success / max(attempts, 1) if attempts > 0 else 0.0

        if len(recent_sequence) >= 2:
            bigram = tuple(recent_sequence[-2:])
            total = sum(bigram_transitions[bigram].values())
            if total > 0:
                p_prob = bigram_transitions[bigram]['P'] / total
                b_prob = bigram_transitions[bigram]['B'] / total
                prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['bigram']
                reliability = min(total / 5, 1.0)
                pattern_reliability['Bigram'] = reliability
                insights['Bigram'] = {
                    'weight': weights['bigram'] * 100,
                    'p_prob': p_prob * 100,
                    'b_prob': b_prob * 100,
                    'reliability': reliability * 100,
                    'recent_performance': recent_performance['bigram'] * 100
                }

        if len(recent_sequence) >= 3:
            trigram = tuple(recent_sequence[-3:])
            total = sum(trigram_transitions[trigram].values())
            if total > 0:
                p_prob = trigram_transitions[trigram]['P'] / total
                b_prob = trigram_transitions[trigram]['B'] / total
                prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['trigram']
                reliability = min(total / 3, 1.0)
                pattern_reliability['Trigram'] = reliability
 ACIÓN COMPLETADA
