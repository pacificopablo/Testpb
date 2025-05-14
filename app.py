# Version: 2025-05-14-fix-v16
import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
try:
    import numpy as np
except ImportError as e:
    raise ImportError("NumPy is not installed. Please run `pip install numpy`.") from e
from typing import Tuple, Dict, Optional, List
import tempfile
import logging
import traceback

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
PARLAY_TABLE = [1, 1, 1, 2, 3, 4, 6, 8, 12, 16, 22, 30, 40, 52, 70, 95]
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
APP_VERSION = "2025-05-14-fix-v16"

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def initialize_session_state():
    """Initialize session state with default values and error handling."""
    logging.debug("Entering initialize_session_state")
    try:
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
            't3_accuracy': {'correct': 0, 'total': 0},
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
            'no_bet_count': 0,  # New: Track consecutive "no bets" for T3 recovery
        }
        defaults['pattern_success']['fourgram'] = 0
        defaults['pattern_attempts']['fourgram'] = 0
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        if st.session_state.strategy not in STRATEGIES:
            st.session_state.strategy = 'T3'
        logging.debug("initialize_session_state completed")
    except Exception as e:
        logging.error(f"initialize_session_state error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error initializing session state: {str(e)}")
        raise

def calculate_weights(streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> Dict[str, float]:
    """Calculate adaptive weights with enhanced fourgram priority."""
    logging.debug("Entering calculate_weights")
    try:
        total_bets = max(st.session_state.pattern_attempts.get('fourgram', 1), 1)
        success_ratios = {
            'bigram': st.session_state.pattern_success.get('bigram', 0) / total_bets,
            'trigram': st.session_state.pattern_success.get('trigram', 0) / total_bets,
            'fourgram': st.session_state.pattern_success.get('fourgram', 0) / total_bets * 1.5,  # Boost fourgram
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
        st.error(f"Error calculating weights: {str(e)}")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate the next bet amount with robust checks and T3 recovery."""
    logging.debug("Entering calculate_bet_amount")
    try:
        if st.session_state.bankroll <= 0:
            st.session_state.t3_level = 1
            return None, "No bet: Bankroll is zero"

        if st.session_state.pattern_volatility > 0.5:  # Reverted to 0.5
            st.session_state.no_bet_count += 1
            if st.session_state.no_bet_count >= 5 and st.session_state.strategy == 'T3':
                st.session_state.t3_level = 1
                st.session_state.no_bet_count = 0
                logging.debug("T3 level reset to 1 due to 5 consecutive no bets")
            return None, f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.no_bet_count = 0

        if pred is None or conf < 40.0:
            return None, f"No bet: Confidence too low ({conf:.1f}%)"
        if st.session_state.last_win_confidence < 40.0 and st.session_state.consecutive_wins > 0:
            return None, f"No bet: Low-confidence win ({st.session_state.last_win_confidence:.1f}%)"

        adjusted_base_bet = st.session_state.base_bet
        if st.session_state.bankroll < st.session_state.initial_bankroll * 0.5:
            adjusted_base_bet = max(st.session_state.base_bet * 0.5, 0.01)  # Ensure non-zero
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
        st.error(f"Error calculating bet amount: {str(e)}")
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
                    st.session_state.t3_accuracy['correct'] += 1
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
                    st.session_state.t3_accuracy['total'] += 1
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
        st.error(f"Error processing result: {str(e)}")

def render_status():
    """Render the session status section with T3 accuracy."""
    logging.debug("Entering render_status")
    try:
        st.subheader("Session Status")
        st.markdown("""
        <style>
        .status-box {
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .status-label {
            font-weight: bold;
            color: #343a40;
        }
        .status-value {
            color: #007bff;
        }
        .status-negative {
            color: #dc3545;
        }
        .status-neutral {
            color: #6c757d;
        }
        @media (max-width: 600px) {
            .status-box {
                padding: 10px;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        profit_loss = st.session_state.bankroll - st.session_state.initial_bankroll
        profit_loss_pct = (profit_loss / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0
        total_bets = st.session_state.wins + st.session_state.losses
        win_rate = (st.session_state.wins / total_bets * 100) if total_bets > 0 else 0.0

        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            progress = (profit_loss / target_profit * 100) if target_profit > 0 else 0.0
            target_text = f"{st.session_state.target_value}% Profit (${target_profit:.2f})"
        else:
            target_units = st.session_state.target_value
            units_earned = profit_loss / st.session_state.initial_base_bet if st.session_state.initial_base_bet > 0 else 0.0
            progress = (units_earned / target_units * 100) if target_units > 0 else 0.0
            target_text = f"{target_units} Units"

        status_html = "<div class='status-box'>"
        status_html += f"<p><span class='status-label'>Bankroll:</span> <span class='status-value'>${st.session_state.bankroll:.2f}</span></p>"
        status_html += f"<p><span class='status-label'>Profit/Loss:</span> <span class={'status-value' if profit_loss >= 0 else 'status-negative'}>${profit_loss:.2f} ({profit_loss_pct:.1f}%)</span></p>"
        status_html += f"<p><span class='status-label'>Wins/Losses:</span> <span class='status-value'>{st.session_state.wins}/{st.session_state.losses}</span> (Win Rate: {win_rate:.1f}%)</p>"
        status_html += f"<p><span class='status-label'>Strategy:</span> <span class='status-value'>{st.session_state.strategy}</span></p>"

        if st.session_state.strategy == 'T3':
            t3_accuracy = (st.session_state.t3_accuracy['correct'] / st.session_state.t3_accuracy['total'] * 100) if st.session_state.t3_accuracy['total'] > 0 else 0.0
            status_html += f"<p><span class='status-label'>T3 Level:</span> <span class='status-value'>{st.session_state.t3_level}</span> (Peak: {st.session_state.t3_peak_level}, Changes: {st.session_state.t3_level_changes})</p>"
            status_html += f"<p><span class='status-label'>T3 Accuracy:</span> <span class='status-value'>{t3_accuracy:.1f}%</span> ({st.session_state.t3_accuracy['correct']}/{st.session_state.t3_accuracy['total']})</p>"
        elif st.session_state.strategy == 'Parlay16':
            status_html += f"<p><span class='status-label'>Parlay Step:</span> <span class='status-value'>{st.session_state.parlay_step}</span> (Peak: {st.session_state.parlay_peak_step}, Wins: {st.session_state.parlay_wins})</p>"
        elif st.session_state.strategy == 'Z1003.1':
            status_html += f"<p><span class='status-label'>Z1003 Loss Count:</span> <span class='status-value'>{st.session_state.z1003_loss_count}</span> (Bet Factor: {st.session_state.z1003_bet_factor:.2f})</p>"

        status_html += f"<p><span class='status-label'>Target:</span> <span class='status-value'>{target_text}</span> (Progress: {progress:.1f}%)</p>"
        status_html += f"<p><span class='status-label'>Safety Net:</span> <span class='status-value'>{'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}</span> ({st.session_state.safety_net_percentage}%)</p>"
        status_html += f"<p><span class='status-label'>Consecutive Wins/Losses:</span> <span class='status-value'>{st.session_state.consecutive_wins}/{st.session_state.consecutive_losses}</span></p>"
        status_html += f"<p><span class='status-label'>Pattern Volatility:</span> <span class='status-value'>{st.session_state.pattern_volatility:.2f}</span></p>"
        status_html += "</div>"

        st.markdown(status_html, unsafe_allow_html=True)
        logging.debug("render_status completed")
    except Exception as e:
        logging.error(f"render_status error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error rendering status section: {str(e)}")

def main():
    """Main application logic with enhanced error handling."""
    logging.debug("Entering main")
    try:
        # Verify dependencies
        if not hasattr(np, 'exp'):
            raise ImportError("NumPy is not properly installed or corrupted.")

        st.set_page_config(page_title="Baccarat Predictor", layout="wide")
        st.title(f"Baccarat Predictor v{APP_VERSION}")
        initialize_session_state()

        num_users = track_user_session()
        st.sidebar.write(f"Active Users: {num_users}")

        if st.sidebar.button("Reset Session"):
            reset_session()
            st.success("Session reset successfully!")
            st.rerun()

        if st.session_state.target_hit:
            st.success("Target hit! Session ended. Reset to start a new session.")
            return

        render_setup_form()
        if st.session_state.bankroll > 0:
            st.sidebar.subheader("Session Stats")
            st.sidebar.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
            st.sidebar.write(f"Wins: {st.session_state.wins}")
            st.sidebar.write(f"Losses: {st.session_state.losses}")
            st.sidebar.write(f"Strategy: {st.session_state.strategy}")
            if st.session_state.strategy == 'T3':
                t3_accuracy = (st.session_state.t3_accuracy['correct'] / st.session_state.t3_accuracy['total'] * 100) if st.session_state.t3_accuracy['total'] > 0 else 0.0
                st.sidebar.write(f"T3 Level: {st.session_state.t3_level}")
                st.sidebar.write(f"T3 Peak Level: {st.session_state.t3_peak_level}")
                st.sidebar.write(f"T3 Level Changes: {st.session_state.t3_level_changes}")
                st.sidebar.write(f"T3 Accuracy: {t3_accuracy:.1f}%")
            elif st.session_state.strategy == 'Parlay16':
                st.sidebar.write(f"Parlay Step: {st.session_state.parlay_step}")
                st.sidebar.write(f"Parlay Peak Step: {st.session_state.parlay_peak_step}")
                st.sidebar.write(f"Parlay Step Changes: {st.session_state.parlay_step_changes}")
                st.sidebar.write(f"Parlay Wins: {st.session_state.parlay_wins}")
            elif st.session_state.strategy == 'Z1003.1':
                st.sidebar.write(f"Z1003 Loss Count: {st.session_state.z1003_loss_count}")
                st.sidebar.write(f"Z1003 Bet Factor: {st.session_state.z1003_bet_factor:.2f}")
                st.sidebar.write(f"Z1003 Level Changes: {st.session_state.z1003_level_changes}")
            st.sidebar.write(f"Consecutive Wins: {st.session_state.consecutive_wins}")
            st.sidebar.write(f"Consecutive Losses: {st.session_state.consecutive_losses}")
            st.sidebar.write(f"Pattern Volatility: {st.session_state.pattern_volatility:.2f}")

            col1, col2 = st.columns([2, 1])
            with col1:
                render_status()
                render_result_input()
                render_prediction()
                render_bead_plate()
            with col2:
                render_insights()

            with st.expander("Loss Log", expanded=False):
                if st.session_state.loss_log:
                    for i, log in enumerate(st.session_state.loss_log):
                        st.write(f"**Loss {i+1}**")
                        st.write(f"- Sequence: {''.join(log['sequence'])}")
                        st.write(f"- Prediction: {log['prediction']}")
                        st.write(f"- Result: {log['result']}")
                        st.write(f"- Confidence: {log['confidence']}%")
                        st.write("- Insights:")
                        for key, value in log['insights'].items():
                            st.write(f"  - {key}: {value}")
                else:
                    st.info("No losses recorded yet.")

            with st.expander("Run Simulation", expanded=False):
                if st.button("Simulate Shoe (80 hands)"):
                    result = simulate_shoe()
                    st.write(f"**Simulation Results**")
                    st.write(f"- Accuracy: {result['accuracy']:.1f}%")
                    st.write(f"- Correct Predictions: {result['correct']}/{result['total']}")
                    st.write(f"- Sequence: {''.join(result['sequence'])}")
                    st.write("- Pattern Performance:")
                    for pattern in result['pattern_success']:
                        st.write(f"  - {pattern}: {result['pattern_success'][pattern]}/{result['pattern_attempts'][pattern]}")

        logging.debug("main completed")
    except ImportError as e:
        logging.error(f"Dependency error in main: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Dependency error: {str(e)}. Ensure NumPy is installed (`pip install numpy`).")
    except NameError as e:
        logging.error(f"NameError in main: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Variable error: {str(e)}. Try resetting the session.")
    except Exception as e:
        logging.error(f"main error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error running application: {str(e)}. Try resetting the session or check logs.")

if __name__ == "__main__":
    main()
