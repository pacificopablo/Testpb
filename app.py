import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import tempfile
import logging

# Constants
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
APP_VERSION = "2025-05-14-opt-v8"

# Logging Setup
logging.basicConfig(filename='app.log', level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s')

# Session Tracking
def track_user_session() -> int:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())
    
    sessions = {}
    current_time = datetime.now()
    
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    session_id, timestamp = line.strip().split(',')
                    last_seen = datetime.fromisoformat(timestamp)
                    if current_time - last_seen <= timedelta(seconds=30):
                        sessions[session_id] = last_seen
    except Exception as e:
        logging.warning(f"Session file error: {str(e)}")
        return 1
    
    sessions[st.session_state.session_id] = current_time
    
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except Exception as e:
        logging.warning(f"Session file write error: {str(e)}")
    
    return len(sessions)

# Session State Management
def reset_session():
    defaults = {
        'bankroll': 0.0, 'base_bet': 0.0, 'initial_base_bet': 0.0, 'sequence': [], 'pending_bet': None,
        'strategy': 'T3', 't3_level': 1, 't3_results': [], 't3_level_changes': 0, 't3_peak_level': 1,
        'parlay_step': 1, 'parlay_wins': 0, 'parlay_using_base': True, 'parlay_step_changes': 0,
        'parlay_peak_step': 1, 'z1003_loss_count': 0, 'z1003_bet_factor': 1.0, 'z1003_continue': False,
        'z1003_level_changes': 0, 'advice': "", 'history': [], 'wins': 0, 'losses': 0,
        'target_mode': 'Profit %', 'target_value': 10.0, 'initial_bankroll': 0.0, 'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0}, 'consecutive_losses': 0, 'loss_log': [],
        'last_was_tie': False, 'insights': {}, 'pattern_volatility': 0.0, 'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int), 'safety_net_percentage': 10.0, 'last_win_confidence': 0.0,
        'consecutive_wins': 0
    }
    defaults['pattern_success']['fourgram'] = 0
    defaults['pattern_attempts']['fourgram'] = 0
    for key in list(st.session_state.keys()):
        if key != 'session_id':
            del st.session_state[key]
    st.session_state.update(defaults)

# Prediction Logic
@st.cache_data
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, Dict, int, int, int, float, float, Dict]:
    try:
        bigram_transitions = defaultdict(lambda: defaultdict(int))
        trigram_transitions = defaultdict(lambda: defaultdict(int))
        fourgram_transitions = defaultdict(lambda: defaultdict(int))
        pattern_transitions = defaultdict(lambda: defaultdict(int))
        streak_count = chop_count = double_count = pattern_changes = 0
        current_streak = None
        player_count = banker_count = 0
        streak_lengths = []
        chop_lengths = []
        filtered_sequence = [x for x in sequence if x in ['P', 'B']]
        
        for i, (curr, next_out) in enumerate(zip(sequence[:-1], sequence[1:])):
            if curr == 'P':
                player_count += 1
            elif curr == 'B':
                banker_count += 1
                
            if i < len(sequence) - 2:
                bigram = tuple(sequence[i:i+2])
                bigram_transitions[bigram][sequence[i+2]] += 1
                if i < len(sequence) - 3:
                    trigram = tuple(sequence[i:i+3])
                    trigram_transitions[trigram][sequence[i+3]] += 1
                    if i < len(sequence) - 4:
                        fourgram = tuple(sequence[i:i+4])
                        fourgram_transitions[fourgram][sequence[i+4]] += 1
        
        current_streak_length = current_chop_length = 0
        for i in range(1, len(filtered_sequence)):
            if filtered_sequence[i] == filtered_sequence[i-1]:
                current_streak = filtered_sequence[i]
                current_streak_length += 1
                if i > 1 and filtered_sequence[i-1] == filtered_sequence[i-2]:
                    double_count += 1
                if current_chop_length > 1:
                    chop_lengths.append(current_chop_length)
                    current_chop_length = 0
            else:
                if current_streak_length > 1:
                    streak_lengths.append(current_streak_length)
                    streak_count += 1
                current_streak = None
                current_streak_length = 0
                if i > 1 and filtered_sequence[i] != filtered_sequence[i-2]:
                    current_chop_length += 1
                    chop_count += 1
                else:
                    if current_chop_length > 1:
                        chop_lengths.append(current_chop_length)
                    current_chop_length = 0
        
        if current_streak_length > 1:
            streak_lengths.append(current_streak_length)
            streak_count += 1
        if current_chop_length > 1:
            chop_lengths.append(current_chop_length)
        
        volatility = len(streak_lengths + chop_lengths) / max(len(filtered_sequence) - 2, 1)
        total_outcomes = max(player_count + banker_count, 1)
        shoe_bias = player_count / total_outcomes if player_count > banker_count else -banker_count / total_outcomes
        
        return (
            bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
            streak_count, chop_count, double_count, volatility, shoe_bias,
            {
                'avg_streak_length': np.mean(streak_lengths) if streak_lengths else 0,
                'avg_chop_length': np.mean(chop_lengths) if chop_lengths else 0,
                'streak_frequency': len(streak_lengths) / max(len(filtered_sequence), 1),
                'chop_frequency': len(chop_lengths) / max(len(filtered_sequence), 1)
            }
        )
    except Exception as e:
        logging.error(f"analyze_patterns error: {str(e)}")
        st.error("Error analyzing patterns.")
        return ({}, {}, {}, {}, 0, 0, 0, 0.0, 0.0, {})

@st.cache_data
def calculate_weights(streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> Dict[str, float]:
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
        
        weights = {k: min(max(np.exp(v) / (1 + np.exp(v)), 0.05), 0.95) for k, v in success_ratios.items()}
        total_weight = sum(weights.values()) or 1
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        st.session_state.insights['Dominant Pattern'] = {
            'pattern': max(normalized_weights, key=normalized_weights.get),
            'weight': normalized_weights[max(normalized_weights, key=normalized_weights.get)] * 100
        }
        
        return normalized_weights
    except Exception colazione as e:
        logging.error(f"calculate_weights error: {str(e)}")
        st.error("Error calculating weights.")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}

def predict_next() -> Tuple[Optional[str], float, Dict]:
    try:
        sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
        shadow_sequence = [x for x in sequence if x in ['P', 'B']]
        if len(shadow_sequence) < 4:
            return 'B', 45.86, {'Initial': 'Default to Banker'}
        
        recent_sequence = shadow_sequence[-WINDOW_SIZE:]
        patterns_data = analyze_patterns(recent_sequence)
        st.session_state.pattern_volatility = patterns_data[7]
        weights = calculate_weights(*patterns_data[4:7], patterns_data[8])
        
        prob_p = prob_b = total_weight = 0
        insights = {}
        prior_p, prior_b = 0.4462, 0.4586
        
        if len(recent_sequence) >= 2:
            bigram = tuple(recent_sequence[-2:])
            total = sum(patterns_data[0][bigram].values())
            if total > 0:
                p_prob = patterns_data[0][bigram]['P'] / total
                b_prob = patterns_data[0][bigram]['B'] / total
                prob_p += weights['bigram'] * (prior_p + p_prob)
                prob_b += weights['bigram'] * (prior_b + b_prob)
                total_weight += weights['bigram']
                insights['Bigram'] = {'weight': weights['bigram'] * 100, 'p_prob': p_prob * 100, 'b_prob': b_prob * 100}
        
        if len(recent_sequence) >= 3:
            trigram = tuple(recent_sequence[-3:])
            total = sum(patterns_data[1][trigram].values())
            if total > 0:
                p_prob = patterns_data[1][trigram]['P'] / total
                b_prob = patterns_data[1][trigram]['B'] / total
                prob_p += weights['trigram'] * (prior_p + p_prob)
                prob_b += weights['trigram'] * (prior_b + b_prob)
                total_weight += weights['trigram']
                insights['Trigram'] = {'weight': weights['trigram'] * 100, 'p_prob': p_prob * 100, 'b_prob': b_prob * 100}
        
        if len(recent_sequence) >= 4:
            fourgram = tuple(recent_sequence[-4:])
            total = sum(patterns_data[2][fourgram].values())
            if total > 0:
                p_prob = patterns_data[2][fourgram]['P'] / total
                b_prob = patterns_data[2][fourgram]['B'] / total
                prob_p += weights['fourgram'] * (prior_p + p_prob)
                prob_b += weights['fourgram'] * (prior_b + b_prob)
                total_weight += weights['fourgram']
                insights['Fourgram'] = {'weight': weights['fourgram'] * 100, 'p_prob': p_prob * 100, 'b_prob': b_prob * 100}
        
        prob_p = (prob_p / total_weight * 100) if total_weight > 0 else 44.62
        prob_b = (prob_b / total_weight * 100) if total_weight > 0 else 45.86
        
        threshold = min(max(32.0 + st.session_state.consecutive_losses * 2.0, 32.0), 48.0)
        prediction = 'P' if prob_p > prob_b and prob_p >= threshold else 'B' if prob_b >= threshold else None
        confidence = max(prob_p, prob_b)
        insights['Threshold'] = {'value': f'{threshold:.1f}%'}
        
        if prediction is None:
            insights['No Bet'] = {'reason': f'Confidence below threshold ({confidence:.1f}% < {threshold:.1f}%)'}
        
        return prediction, confidence, insights
    except Exception as e:
        logging.error(f"predict_next error: {str(e)}")
        st.error("Error predicting outcome.")
        return None, 0.0, {}

# Betting Logic
def check_target_hit() -> bool:
    try:
        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
        return (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet >= st.session_state.target_value
    except Exception as e:
        logging.error(f"check_target_hit error: {str(e)}")
        st.error("Error checking target.")
        return False

def update_t3_level():
    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        old_level = st.session_state.t3_level
        st.session_state.t3_level = max(1, st.session_state.t3_level + (2 if wins == 0 else 1 if wins == 1 else -1 if wins == 2 else -2))
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
        st.session_state.t3_results = []

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], str]:
    try:
        if st.session_state.consecutive_losses >= 3 and conf < 45.0 or st.session_state.pattern_volatility > 0.6 or pred is None:
            return None, "No bet: Risk too high"
        
        if st.session_state.strategy == 'Z1003.1' and st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            return None, "No bet: Z1003.1 stop rule"
        elif st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
        else:
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
        
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if bet_amount > st.session_state.bankroll or st.session_state.bankroll - bet_amount < safe_bankroll * 0.5:
            st.session_state.t3_level = st.session_state.parlay_step = 1
            st.session_state.z1003_loss_count = 0
            return None, "No bet: Bankroll too low"
        
        return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"
    except Exception as e:
        logging.error(f"calculate_bet_amount error: {str(e)}")
        st.error("Error calculating bet.")
        return None, "No bet: Calculation error"

def place_result(result: str):
    try:
        if st.session_state.target_hit:
            reset_session()
            return
        
        previous_state = {
            'bankroll': st.session_state.bankroll, 'wins': st.session_state.wins, 'losses': st.session_state.losses,
            'consecutive_wins': st.session_state.consecutive_wins, 'consecutive_losses': st.session_state.consecutive_losses,
            't3_level': st.session_state.t3_level, 't3_results': st.session_state.t3_results.copy(),
            'parlay_step': st.session_state.parlay_step, 'parlay_wins': st.session_state.parlay_wins,
            'parlay_using_base': st.session_state.parlay_using_base, 'z1003_loss_count': st.session_state.z1003_loss_count,
            'z1003_continue': st.session_state.z1003_continue, 'pending_bet': st.session_state.pending_bet,
            'prediction_accuracy': st.session_state.prediction_accuracy.copy()
        }
        
        bet_placed = False
        win = False
        bet_amount = selection = None
        
        if st.session_state.pending_bet and result != 'T':
            bet_amount, selection = st.session_state.pending_bet
            win = result == selection
            bet_placed = True
            if win:
                st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                st.session_state.last_win_confidence = predict_next()[1]
                st.session_state.wins += 1
                if st.session_state.consecutive_wins >= 3:
                    st.session_state.base_bet = round(st.session_state.base_bet * 1.05, 2)
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
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                    else:
                        st.session_state.parlay_using_base = False
                elif st.session_state.strategy == 'Z1003.1':
                    _, conf, _ = predict_next()
                    st.session_state.z1003_continue = conf > 50.0 and st.session_state.pattern_volatility < 0.4
                    if not st.session_state.z1003_continue:
                        st.session_state.z1003_loss_count = 0
                st.session_state.prediction_accuracy[selection] += 1
                for pattern in st.session_state.insights:
                    st.session_state.pattern_success[pattern] += 1
                    st.session_state.pattern_attempts[pattern] += 1
            else:
                st.session_state.bankroll -= bet_amount
                st.session_state.consecutive_wins = 0
                st.session_state.consecutive_losses += 1
                st.session_state.losses += 1
                _, conf, _ = predict_next()
                st.session_state.loss_log.append({
                    'sequence': st.session_state.sequence[-10:], 'prediction': selection,
                    'result': result, 'confidence': f"{conf:.1f}", 'insights': st.session_state.insights.copy()
                })
                if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                    st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
                for pattern in st.session_state.insights:
                    st.session_state.pattern_attempts[pattern] += 1
            st.session_state.prediction_accuracy['total'] += 1
            st.session_state.pending_bet = None
        
        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) > SEQUENCE_LIMIT:
            st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LIMIT:]
        
        st.session_state.history.append({
            "Bet": selection, "Result": result, "Amount": bet_amount, "Win": win,
            "T3_Level": st.session_state.t3_level, "Parlay_Step": st.session_state.parlay_step,
            "Z1003_Loss_Count": st.session_state.z1003_loss_count, "Previous_State": previous_state,
            "Bet_Placed": bet_placed, "Consecutive_Wins": st.session_state.consecutive_wins
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
        
        if check_target_hit():
            st.session_state.target_hit = True
            return
        
        pred, conf, insights = predict_next()
        bet_amount, advice = calculate_bet_amount(pred, conf)
        st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
        st.session_state.advice = advice
        st.session_state.insights = insights
        
        if st.session_state.strategy == 'T3':
            update_t3_level()
    except Exception as e:
        logging.error(f"place_result error: {str(e)}")
        st.error("Error processing result.")

# Simulation Logic
def simulate_shoe(num_hands: int = 80) -> Dict:
    try:
        outcomes = np.random.choice(['P', 'B', 'T'], size=num_hands, p=[0.4462, 0.4586, 0.0952])
        sequence = []
        correct = total = 0
        
        for outcome in outcomes:
            sequence.append(outcome)
            st.session_state.sequence = sequence[-SEQUENCE_LIMIT:]
            pred, _, _ = predict_next()
            if pred and outcome in ['P', 'B']:
                total += 1
                if pred == outcome:
                    correct += 1
            st.session_state.prediction_accuracy['total'] += 1
            if outcome in ['P', 'B']:
                st.session_state.prediction_accuracy[outcome] += 1 if pred == outcome else 0
        
        accuracy = (correct / total * 100) if total > 0 else 0
        result = {'accuracy': accuracy, 'correct': correct, 'total': total, 'sequence': sequence}
        
        try:
            with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()}: Accuracy={accuracy:.1f}%, Correct={correct}/{total}\n")
        except Exception as e:
            logging.warning(f"Simulation log error: {str(e)}")
        
        return result
    except Exception as e:
        logging.error(f"simulate_shoe error: {str(e)}")
        st.error("Error running simulation.")
        return {'accuracy': 0, 'correct': 0, 'total': 0, 'sequence': []}

# UI Components
def render_setup_form():
    st.subheader("Setup")
    with st.form("setup_form"):
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 10.0, step=0.01, format="%.2f")
        base_bet = st.number_input("Base Bet ($)", min_value=0.01, value=st.session_state.base_bet or 0.20, step=0.01, format="%.2f")
        strategy = st.selectbox("Betting Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy))
        target_mode = st.radio("Target Type", ["Profit %", "Units"], horizontal=True)
        target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
        safety_net = st.number_input("Safety Net (%)", min_value=0.0, max_value=50.0, value=st.session_state.safety_net_percentage, step=5.0)
        if st.form_submit_button("Start Session"):
            if bankroll <= 0 or base_bet < 0.01 or base_bet > bankroll:
                st.error("Invalid bankroll or base bet.")
            else:
                st.session_state.update({
                    'bankroll': bankroll, 'base_bet': base_bet, 'initial_base_bet': base_bet, 'strategy': strategy,
                    'target_mode': target_mode, 'target_value': target_value, 'initial_bankroll': bankroll,
                    'safety_net_percentage': safety_net, 'sequence': [], 'history': [], 'wins': 0, 'losses': 0,
                    'pending_bet': None, 't3_level': 1, 't3_results': [], 'parlay_step': 1, 'parlay_wins': 0,
                    'parlay_using_base': True, 'z1003_loss_count': 0, 'z1003_continue': False
                })
                st.success(f"Session started with {strategy}!")

def render_result_input():
    st.subheader("Enter Result")
    st.markdown("""
    <style>
    .stButton > button {width: 90px; height: 35px; font-size: 14px; font-weight: bold; border-radius: 6px; border: 1px solid; cursor: pointer;}
    .stButton > button:hover {transform: scale(1.05);}
    .stButton > button[kind="player_btn"] {background: #007bff; border-color: #0056b3; color: white;}
    .stButton > button[kind="banker_btn"] {background: #dc3545; border-color: #a71d2a; color: white;}
    .stButton > button[kind="tie_btn"] {background: #28a745; border-color: #1e7e34; color: white;}
    .stButton > button[kind="undo_btn"] {background: #6c757d; border-color: #545b62; color: white;}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player", key="player_btn"):
            place_result("P")
    with col2:
        if st.button("Banker", key="banker_btn"):
            place_result("B")
    with col3:
        if st.button("Tie", key="tie_btn"):
            place_result("T")
    with col4:
        if st.button("Undo Last", key="undo_btn"):
            if st.session_state.sequence:
                if st.session_state.history:
                    last = st.session_state.history.pop()
                    st.session_state.update(last['Previous_State'])
                    st.session_state.sequence.pop()
                    if last['Bet_Placed'] and not last['Win'] and st.session_state.loss_log:
                        if st.session_state.loss_log[-1]['result'] == last['Result']:
                            st.session_state.loss_log.pop()
                    st.session_state.advice = f"Next Bet: ${st.session_state.pending_bet[0]:.2f} on {st.session_state.pending_bet[1]}" if st.session_state.pending_bet else "No bet pending."
                    st.success("Undone last action.")
                    st.rerun()
                else:
                    st.session_state.sequence.pop()
                    st.session_state.pending_bet = None
                    st.session_state.advice = "No bet pending."
                    st.success("Undone last result.")
                    st.rerun()
            else:
                st.warning("No results to undo.")

def render_bead_plate():
    st.subheader("Sequence (Bead Plate)")
    sequence = st.session_state.sequence[-90:]
    grid = [sequence[i*6:(i+1)*6] + [''] * (6 - len(sequence[i*6:(i+1)*6])) for i in range(15)]
    html = "<div style='display: flex; gap: 5px; overflow-x: auto;'>"
    for col in grid:
        html += "<div style='display: flex; flex-direction: column; gap: 5px;'>"
        for result in col:
            style = "width: 20px; height: 20px; border: 1px solid #ddd; border-radius: 50%;" if not result else f"width: 20px; height: 20px; background: {'blue' if result == 'P' else 'red' if result == 'B' else 'green'}; border-radius: 50%;"
            html += f"<div style='{style}'></div>"
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_prediction():
    if st.session_state.pending_bet:
        amount, side = st.session_state.pending_bet
        if amount:
            st.markdown(f"<h4 style='color: {'blue' if side == 'P' else 'red'};'>Prediction: {side} | Bet: ${amount:.2f}</h4>", unsafe_allow_html=True)
        else:
            st.info("No bet placed.")
    elif not st.session_state.target_hit:
        st.info(st.session_state.advice)

def render_insights():
    st.subheader("Prediction Insights")
    if not st.session_state.insights:
        st.info("No insights yet.")
        return
    
    extra_metrics = analyze_patterns(st.session_state.sequence[-WINDOW_SIZE:])[-1]
    pattern_insights = {k: v for k, v in st.session_state.insights.items() if k in ['Bigram', 'Trigram', 'Fourgram']}
    meta_insights = {k: v for k, v in st.session_state.insights.items() if k in ['Threshold', 'No Bet', 'Dominant Pattern']}
    
    if pattern_insights:
        st.markdown("**Patterns**:")
        for pattern, data in sorted(pattern_insights.items(), key=lambda x: x[1].get('weight', 0), reverse=True):
            with st.expander(f"{pattern} ({data.get('weight', 0):.1f}%)"):
                st.markdown(f"- Player: {data.get('p_prob', 0):.1f}% | Banker: {data.get('b_prob', 0):.1f}%")
    
    if meta_insights:
        st.markdown("**Factors**:")
        if 'Dominant Pattern' in meta_insights:
            st.markdown(f"- Dominant: {meta_insights['Dominant Pattern'].get('pattern', 'N/A')} ({meta_insights['Dominant Pattern'].get('weight', 0):.1f}%)")
        if 'Threshold' in meta_insights:
            st.markdown(f"- Threshold: {meta_insights['Threshold'].get('value', 'N/A')}")
        if 'No Bet' in meta_insights:
            st.info(f"- No Bet: {meta_insights['No Bet'].get('reason', 'N/A')}")
    
    st.markdown(f"**Trends**: Streak Length: {extra_metrics.get('avg_streak_length', 0):.1f}, Chop Length: {extra_metrics.get('avg_chop_length', 0):.1f}")

def render_status():
    st.subheader("Status")
    st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f} | **Base Bet**: ${st.session_state.base_bet:.2f}")
    st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
    st.markdown(f"**Strategy**: {st.session_state.strategy}")
    if st.session_state.initial_base_bet > 0:
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        st.markdown(f"**Profit**: {profit / st.session_state.initial_base_bet:.2f} units (${profit:.2f})")

def render_accuracy():
    st.subheader("Accuracy")
    total = st.session_state.prediction_accuracy['total']
    if total > 0:
        p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
        b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
        st.markdown(f"Player: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
        st.markdown(f"Banker: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

def render_history():
    if st.session_state.history:
        st.subheader("History")
        n = st.slider("Show last N bets", 5, 50, 10)
        st.dataframe([
            {
                "Bet": h["Bet"] or "-", "Result": h["Result"], "Amount": f"${h['Amount']:.2f}" if h["Bet_Placed"] else "-",
                "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-"
            } for h in st.session_state.history[-n:]
        ])

def render_simulation():
    st.subheader("Simulation")
    num_hands = st.number_input("Hands to Simulate", min_value=10, max_value=200, value=80, step=10)
    if st.button("Run Simulation"):
        result = simulate_shoe(num_hands)
        st.write(f"Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

# Main App
st.set_page_config(page_title=f"Baccarat Predictor {APP_VERSION}", layout="wide")
st.title(f"Baccarat Predictor {APP_VERSION}")
reset_session()
render_setup_form()
if st.session_state.initial_bankroll > 0:
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        render_result_input()
        render_prediction()
        render_bead_plate()
        render_history()
    with col2:
        render_status()
        render_accuracy()
        render_insights()
        render_simulation()
if st.button("Reset Session State"):
    reset_session()
    st.success("Session reset.")
