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
APP_VERSION = "2025-05-14-fix-v8"  # Updated version for T3 and Parlay16 fixes

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# --- Modified Functions ---
def update_t3_level():
    """Update T3 betting level based on recent results."""
    logging.debug("Entering update_t3_level")
    try:
        if len(st.session_state.t3_results) >= 3:  # Handle edge cases
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
                logging.debug(f"T3 level changed from {old_level} to {st.session_state.t3_level}")
            st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
            st.session_state.t3_results = []
            logging.debug(f"T3 results reset: {st.session_state.t3_results}")
        logging.debug("update_t3_level completed")
    except Exception as e:
        logging.error(f"update_t3_level error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error updating T3 level. Try resetting the session.")

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate the next bet amount with relaxed constraints."""
    logging.debug("Entering calculate_bet_amount")
    try:
        # Relaxed conditions to allow more bets
        if st.session_state.consecutive_losses >= 3 and conf < 40.0:
            return None, f"No bet: Paused after {st.session_state.consecutive_losses} losses (low confidence)"
        if st.session_state.pattern_volatility > 0.8:
            return None, f"No bet: High pattern volatility"
        if pred is None or conf < 20.0:
            return None, f"No bet: Confidence too low"
        if st.session_state.last_win_confidence < 35.0 and st.session_state.consecutive_wins > 0:
            return None, f"No bet: Low-confidence win ({st.session_state.last_win_confidence:.1f}%)"

        if st.session_state.strategy == 'Z1003.1':
            if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
                return None, "No bet: Stopped after three losses (Z1003.1 rule)"
            bet_amount = st.session_state.base_bet + (st.session_state.z1003_loss_count * 0.10)
        elif st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
        else:  # Parlay16
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
            logging.debug(f"Parlay16 bet: Step={st.session_state.parlay_step}, Using Base={st.session_state.parlay_using_base}, Amount=${bet_amount:.2f}")
            st্ট

        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if bet_amount > st.session_state.bankroll:
            logging.warning(f"Bet {bet_amount} exceeds bankroll {st.session_state.bankroll}. Warning issued.")
            return None, "No bet: Bet exceeds bankroll"
        if st.session_state.bankroll - bet_amount < safe_bankroll * 0.5:
            logging.warning(f"Bet {bet_amount} would leave bankroll below safety net {safe_bankroll * 0.5}. Warning issued.")
            return None, "No bet: Below safety net"

        logging.debug(f"Bet amount calculated: ${bet_amount:.2f} on {pred}")
        return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"
    except Exception as e:
        logging.error(f"calculate_bet_amount error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error calculating bet amount. Try resetting the session.")
        return None, "No bet: Calculation error"

def place_result(result: str):
    """Process a game result with enhanced T3 and Parlay16 handling."""
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
            "parlay_using_base": st.session_state.parlay_using_base,
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
                if st.session_state.consecutive_wins >= 3:
                    st.session_state.base_bet *= 1.05
                    st.session_state.base_bet = round(st.session_state.base_bet, 2)
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('W')
                    logging.debug(f"T3 results after win: {st.session_state.t3_results}")
                elif st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins += 1
                    old_step = st.session_state.parlay_step
                    if st.session_state.parlay_wins == 1:
                        st.session_state.parlay_using_base = False  # Switch to parlay bet
                        st.session_state.parlay_step = min(old_step + 1, 16)  # Advance step
                    elif st.session_state.parlay_wins == 2:
                        st.session_state.parlay_step = 1
                        st.session_state.parlay_wins = 0
                        st.session_state.parlay_using_base = True  # Reset to base
                        if old_step != st.session_state.parlay_step:
                            st.session_state.parlay_step_changes += 1
                    logging.debug(f"Parlay16 after win: Step={st.session_state.parlay_step}, Wins={st.session_state.parlay_wins}, Using Base={st.session_state.parlay_using_base}")
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
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('L')
                    logging.debug(f"T3 results after loss: {st.session_state.t3_results}")
                elif st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins = 0
                    st.session_state.parlay_using_base = True  # Return to base bet after loss
                    logging.debug(f"Parlay16 after loss: Step={st.session_state.parlay_step}, Wins={st.session_state.parlay_wins}, Using Base={st.session_state.parlay_using_base}")
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_attempts[pattern] += 1
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

        # Validate win/loss counts
        if st.session_state.wins < 0 or st.session_state.losses < 0:
            logging.error(f"Invalid win/loss counts: wins={st.session_state.wins}, losses={st.session_state.losses}")
            st.session_state.wins = max(0, st.session_state.wins)
            st.session_state.losses = max(0, st.session_state.losses)

        logging.debug("place_result completed")
    except Exception as e:
        logging.error(f"place_result error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error processing result. Try resetting the session.")

def render_status():
    """Render session status with T3 and Parlay16 debug info."""
    logging.debug("Entering render_status")
    try:
        st.subheader("Status")
        st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
        st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
        st.markdown(f"**Safety Net Percentage**: {st.session_state.safety_net_percentage}%")
        st.markdown(f"**Online Users**: {track_user_session()}")
        strategy_status = f"**Betting Strategy**: {st.session_state.strategy}"
        if st.session_state.strategy == 'T3':
            strategy_status += f" | Level: {st.session_state.t3_level} | Peak Level: {st.session_state.t3_peak_level} | Level Changes: {st.session_state.t3_level_changes}"
            st.markdown(f"**T3 Results**: {st.session_state.t3_results} (W=Win, L=Loss)")
        elif st.session_state.strategy == 'Parlay16':
            strategy_status += f" | Step: {st.session_state.parlay_step}/16 | Peak Step: {st.session_state.parlay_peak_step} | Step Changes: {st.session_state.parlay_step_changes} | Consecutive Wins: {st.session_state.parlay_wins}"
            st.markdown(f"**Parlay16 Status**: Wins={st.session_state.parlay_wins}, Using Base Bet={st.session_state.parlay_using_base}")
        elif st.session_state.strategy == 'Z1003.1':
            strategy_status += f" | Loss Count: {st.session_state.z1003_loss_count} | Level Changes: {st.session_state.z1003_level_changes} | Continue: {st.session_state.z1003_continue}"
        st.markdown(strategy_status)
        st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
        st.markdown(f"**Consecutive Wins**: {st.session_state.consecutive_wins}")

        if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
            profit = st.session_state.bankroll - st.session_state.initial_bankroll
            units_profit = profit / st.session_state.initial_base_bet
            st.markdown(f"**Units Profit**: {units_profit:.2f} units (${profit:.2f})")
        else:
            st.markdown("**Units Profit**: 0.00 units ($0.00)")
        logging.debug("render_status completed")
    except Exception as e:
        logging.error(f"render_status error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering status. Try resetting the session.")
