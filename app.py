# Version: 2025-05-14-fix-v17
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
APP_VERSION = "2025-05-14-fix-v17"

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def track_user_session() -> int:
    """Track active users by managing session file."""
    logging.debug("Entering track_user_session")
    try:
        # Use session ID to track unique users
        session_id = st.session_state.get('session_id', str(hash(datetime.now().timestamp())))
        st.session_state['session_id'] = session_id
        current_time = datetime.now()
        timeout = timedelta(minutes=10)  # Sessions expire after 10 minutes

        # Read existing sessions
        sessions = {}
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r') as f:
                for line in f:
                    sid, timestamp = line.strip().split(',')
                    sessions[sid] = datetime.fromisoformat(timestamp)

        # Update current session
        sessions[session_id] = current_time

        # Remove expired sessions
        sessions = {sid: ts for sid, ts in sessions.items() if current_time - ts < timeout}

        # Write updated sessions
        with open(SESSION_FILE, 'w') as f:
            for sid, ts in sessions.items():
                f.write(f"{sid},{ts.isoformat()}\n")

        num_users = len(sessions)
        logging.debug(f"track_user_session completed: {num_users} active users")
        return num_users
    except Exception as e:
        logging.error(f"track_user_session error: {str(e)}\n{traceback.format_exc()}")
        return 1  # Default to 1 user to avoid UI crashes

# [Unchanged functions: initialize_session_state, calculate_weights, calculate_bet_amount, place_result, render_status, update_t3_level, predict_next, render_setup_form, render_result_input, render_bead_plate, render_prediction, render_insights, simulate_shoe, reset_session, analyze_patterns, check_target_hit]

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

        try:
            num_users = track_user_session()
        except Exception as e:
            logging.error(f"Failed to track user session: {str(e)}\n{traceback.format_exc()}")
            num_users = 1  # Fallback to avoid UI issues
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
                        st.sidebar.write(f"- Result: {log['result']}")
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
