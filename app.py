# Version: 2025-05-14-fix-v14
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
APP_VERSION = "2025-05-14-fix-v14"

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

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
        't3_accuracy': {'correct': 0, 'total': 0},  # New: Track T3-specific accuracy
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

# [Other unchanged functions: track_user_session, reset_session, analyze_patterns, calculate_weights, predict_next, check_target_hit, render_setup_form, render_result_input, render_bead_plate, render_prediction, render_insights, render_status, simulate_shoe, main]

def update_t3_level():
    """Update T3 betting level based on new fixed 3-result rule logic with level cap."""
    logging.debug("Entering update_t3_level")
    try:
        if len(st.session_state.t3_results) >= 3:
            recent = st.session_state.t3_results[-3:]
            pattern = "".join(recent)
            old_level = st.session_state.t3_level

            if pattern == 'WWW':
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif pattern == 'LLL':
                st.session_state.t3_level += 2
            elif pattern in ('WWL', 'WLW', 'LWW'):
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif pattern in ('LLW', 'LWL', 'WLL'):
                st.session_state.t3_level += 1

            # Cap T3 level at 10
            st.session_state.t3_level = min(st.session_state.t3_level, 10)

            if old_level != st.session_state.t3_level:
                st.session_state.t3_level_changes += 1
            st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)

            # Keep only the last 3 results
            st.session_state.t3_results = st.session_state.t3_results[-3:]
        logging.debug("update_t3_level completed")
    except Exception as e:
        logging.error(f"update_t3_level error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error updating T3 level. Try resetting the session.")

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate the next bet amount with dynamic base bet adjustment."""
    logging.debug("Entering calculate_bet_amount")
    try:
        if st.session_state.pattern_volatility > 0.7:  # Relaxed from 0.5 to 0.7
            return None, f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        if pred is None or conf < 40.0:
            return None, f"No bet: Confidence too low ({conf:.1f}%)"
        if st.session_state.last_win_confidence < 40.0 and st.session_state.consecutive_wins > 0:
            return None, f"No bet: Low-confidence win ({st.session_state.last_win_confidence:.1f}%)"

        # Dynamic base bet adjustment
        adjusted_base_bet = st.session_state.base_bet
        if st.session_state.bankroll < st.session_state.initial_bankroll * 0.5:
            adjusted_base_bet *= 0.5  # Reduce base bet by 50% if bankroll is low
            logging.debug(f"Base bet reduced to {adjusted_base_bet:.2f} due to low bankroll")

        if st.session_state.strategy == 'Z1003.1':
            if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
                return None, "No bet: Stopped after three losses (Z1003.1 rule)"
            bet_amount = adjusted_base_bet + (st.session_state.z1003_loss_count * 0.10)
        elif st.session_state.strategy == 'Flatbet':
            bet_amount = adjusted_base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = adjusted_base_bet * st.session_state.t3_level
            logging.debug(f"T3 bet: base_bet={adjusted_base_bet:.2f}, t3_level={st.session_state.t3_level}, bet_amount={bet_amount:.2f}")
        else:  # Parlay16
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[min(st.session_state.parlay_step - 1, len(PARLAY_TABLE) - 1)]
            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)

        if bet_amount > st.session_state.bankroll:
            st.session_state.t3_level = 1
            st.session_state.parlay_step = 1
            st.session_state.z1003_loss_count = 0
            return None, "No bet: Bet exceeds bankroll, levels reset"
        if st.session_state.safety_net_enabled:
            safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
            if st.session_state.bankroll - bet_amount < safe_bankroll * 0.5:
                st.session_state.t3_level = 1
                st.session_state.parlay_step = 1
                st.session_state.z1003_loss_count = 0
                return None, "No bet: Below safety net, levels reset"

        logging.debug("calculate_bet_amount completed")
        return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"
    except Exception as e:
        logging.error(f"calculate_bet_amount error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error calculating bet amount. Try resetting the session.")
        return None, "No bet: Calculation error"

def place_result(result: str):
    """Process a game result with T3 accuracy tracking."""
    logging.debug("Entering place_result")
    try:
        if st.session_state.target_hit:
            reset_session()
            return

        st.session_state.last_was_tie = (result == 'T')
        bet_amount = 0
        bet_placed = False
        selection = None
        win = False

        previous_state = {
            "bankroll": st.session_state.bankroll,
            "t3_level": st.session_state.t3_level,
            "t3_results": st.session_state.t3_results.copy(),
            "parlay_step": st.session_state.parlay_step,
            "parlay_wins": st.session_state.parlay_wins,
            "z1003_loss_count": st.session_state.z1003_loss_count,
            "z1003_bet_factor": st.session_state.z1003_bet_factor,
            "z1003_continue": st.session_state.z1003_continue,
            "z1003_level_changes": st.session_state.z1003_level_changes,
            "pending_bet": st.session_state.pending_bet,
            "wins": st.session_state.wins,
            "losses": st.session_state.losses,
            "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
            "consecutive_losses": st.session_state.consecutive_losses,
            "t3_level_changes": st.session_state.t3_level_changes,
            "parlay_step_changes": st.session_state.parlay_step_changes,
            "pattern_volatility": st.session_state.pattern_volatility,
            "pattern_success": st.session_state.pattern_success.copy(),
            "pattern_attempts": st.session_state.pattern_attempts.copy(),
            "safety_net_percentage": st.session_state.safety_net_percentage,
            "safety_net_enabled": st.session_state.safety_net_enabled,
            "consecutive_wins": st.session_state.consecutive_wins,
            "last_win_confidence": st.session_state.last_win_confidence,
            "insights": st.session_state.insights.copy(),
        }

        if st.session_state.pending_bet and result != 'T':
            bet_amount, selection = st.session_state.pending_bet
            win = result == selection
            bet_placed = True
            if win:
                st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                st.session_state.wins += 1
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                st.session_state.last_win_confidence = predict_next()[1]
                logging.debug(f"Win recorded: Total wins={st.session_state.wins}, Consecutive wins={st.session_state.consecutive_wins}")
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('W')
                    st.session_state.t3_accuracy['correct'] += 1  # Track T3 win
                    st.session_state.t3_accuracy['total'] += 1
                elif st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins += 1
                    if st.session_state.parlay_wins >= 2:
                        old_step = st.session_state.parlay_step
                        st.session_state.parlay_step = 1
                        st.session_state.parlay_wins = 0
                        if old_step != st.session_state.parlay_step:
                            st.session_state.parlay_step_changes += 1
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                elif st.session_state.strategy == 'Z1003.1':
                    _, conf, _ = predict_next()
                    if conf > 50.0 and st.session_state.pattern_volatility < 0.4:
                        st.session_state.z1003_continue = True
                    else:
                        st.session_state.z1003_loss_count = 0
                        st.session_state.z1003_continue = False
                st.session_state.prediction_accuracy[selection] += 1
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_success[pattern] += 1
                        st.session_state.pattern_attempts[pattern] += 1
            else:
                st.session_state.bankroll -= bet_amount
                st.session_state.losses += 1
                st.session_state.consecutive_wins = 0
                st.session_state.consecutive_losses += 1
                logging.debug(f"Loss recorded: Total losses={st.session_state.losses}, Consecutive losses={st.session_state.consecutive_losses}")
                _, conf, _ = predict_next()
                st.session_state.loss_log.append({
                    'sequence': st.session_state.sequence[-10:],
                    'prediction': selection,
                    'result': result,
                    'confidence': f"{conf:.1f}",
                    'insights': st.session_state.insights.copy()
                })
                if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                    st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_attempts[pattern] += 1
                if st.session_state.strategy == 'Parlay16':
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = min(st.session_state.parlay_step + 1, len(PARLAY_TABLE))
                    st.session_state.parlay_wins = 0
                    if old_step != st.session_state.parlay_step:
                        st.session_state.parlay_step_changes += 1
                    st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)
                elif st.session_state.strategy == 'Z1003.1':
                    st.session_state.z1003_loss_count += 1
                    old_factor = st.session_state.z1003_bet_factor
                    st.session_state.z1003_bet_factor = min(st.session_state.z1003_bet_factor + 0.1, 2.0)
                    if old_factor != st.session_state.z1003_bet_factor:
                        st.session_state.z1003_level_changes += 1
                elif st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('L')
                    st.session_state.t3_accuracy['total'] += 1  # Track T3 loss
            st.session_state.prediction_accuracy['total'] += 1
            st.session_state.pending_bet = None

        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) > SEQUENCE_LIMIT:
            st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LIMIT:]

        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level,
            "Parlay_Step": st.session_state.parlay_step,
            "Z1003_Loss_Count": st.session_state.z1003_loss_count,
            "Previous_State": previous_state,
            "Bet_Placed": bet_placed,
            "Consecutive_Wins": st.session_state.consecutive_wins,
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]

        if check_target_hit():
            st.session_state.target_hit = True
            return

        pred, conf, insights = predict_next()
        if st.session_state.strategy == 'Z1003.1' and st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            bet_amount, advice = None, "No bet: Stopped after three losses (Z1003.1 rule)"
        else:
            bet_amount, advice = calculate_bet_amount(pred, conf)
        st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
        st.session_state.advice = advice
        st.session_state.insights = insights

        if st.session_state.strategy == 'T3':
            update_t3_level()
        elif st.session_state.strategy == 'Parlay16' and bet_placed and win and st.session_state.parlay_wins < 2:
            old_step = st.session_state.parlay_step
            st.session_state.parlay_step = min(st.session_state.parlay_step + 1, len(PARLAY_TABLE))
            if old_step != st.session_state.parlay_step:
                st.session_state.parlay_step_changes += 1
            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)

        if st.session_state.wins < 0 or st.session_state.losses < 0:
            logging.error(f"""Invalid win/loss counts: wins={st.session_state.wins}, losses={st.session_state.losses}. The T3 strategy adjusts bet sizes based on periods of three rounds (win or loss) at any level, using the following rules:
- WWW: Decrease level by 2
- LLL: Increase level by 2
- WWL, WLW, LWW: Decrease level by 1
- LLW, LWL, WLL: Increase level by 1""")
            st.session_state.wins = max(0, st.session_state.wins)
            st.session_state.losses = max(0, st.session_state.losses)

        logging.debug("place_result completed")
    except Exception as e:
        logging.error(f"place_result error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error processing result. Try resetting the session.")

# [Rest of the code remains unchanged]
if __name__ == "__main__":
    main()
