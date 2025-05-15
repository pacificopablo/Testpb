import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import uuid

# --- Set Page Config (Must be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="MANG BACCARAT GROUP")

# --- Constants ---
SESSION_FILE = "online_users.txt"
SIMULATION_LOG = "simulation_log.txt"
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

# --- Custom CSS for Elegant Design ---
st.markdown("""
<style>
/* General Styling */
body {
    font-family: 'Roboto', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.stApp {
    max-width: 1280px;
    margin: 0 auto;
    padding: 2rem;
}

/* Card Styling */
.card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
}

/* Headers */
h1 {
    color: #1e40af;
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 2.5rem;
    letter-spacing: 0.5px;
}
h2 {
    color: #1e293b;
    font-weight: 600;
    font-size: 1.75rem;
    margin-bottom: 1rem;
}
h3 {
    color: #475569;
    font-weight: 500;
    font-size: 1.25rem;
}

/* Buttons */
.stButton > button {
    display: block;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    font-size: 0.95rem;
    min-width: 100px;
    width: 100%;
    text-align: center;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin: 0.2rem 0;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    filter: brightness(1.05);
}
.stButton > button:active {
    transform: scale(1);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Specific Button Colors */
button[kind="player_btn"] {
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
}
button[kind="banker_btn"] {
    background: linear-gradient(135deg, #dc2626, #991b1b);
}
button[kind="tie_btn"] {
    background: linear-gradient(135deg, #059669, #064e3b);
}
button[kind="undo_btn"] {
    background: linear-gradient(135deg, #6b7280, #374151);
}

/* Ensure Columns are Visible */
.st-emotion-cache-1r4s1t7 {
    display: flex;
    flex-wrap: nowrap;
    gap: 0.5rem;
    justify-content: center;
    width: 100%;
}

/* Form Inputs */
.stNumberInput, .stSelectbox, .stRadio, .stCheckbox {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 0.5rem;
    border: 1px solid #e2e8f0;
}
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    border: none;
    background: transparent;
}

/* Expander */
.stExpander {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    padding: 0.5rem;
}

/* Bead Plate */
.bead-plate {
    background: #ffffff;
    padding: 1rem;
    border-radius: 12px;
    box/**************_space_start_**************st.session_state.sequence[-90:]
        grid = [[] for _ in range(15)]
        for i, result in enumerate(sequence):
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
        "Z1003_Bet_Factor": None,
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed
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

# --- Simulation Logic ---
def simulate_shoe(num_hands: int = 80) -> Dict:
    """Simulate a Baccarat shoe and log results."""
    outcomes = np.random.choice(
        ['P', 'B', 'T'],
        size=num_hands,
        p=[0.4462, 0.4586, 0.0952]
    )
    sequence = []
    correct = total = 0
    pattern_success = defaultdict(int)
    pattern_attempts = defaultdict(int)

    for outcome in outcomes:
        sequence.append(outcome)
        pred, conf, insights = predict_next()
        if pred and outcome in ['P', 'B']:
            total += 1
            if pred == outcome:
                correct += 1
                for pattern in insights:
                    pattern_success[pattern] += 1
                    pattern_attempts[pattern] += 1
            else:
                for pattern in insights:
                    pattern_attempts[pattern] += 1
        st.session_state.sequence = sequence.copy()
        st.session_state.prediction_accuracy['total'] += 1
        if outcome in ['P', 'B']:
            st.session_state.prediction_accuracy[outcome] += 1 if pred == outcome else 0

    accuracy = (correct / total * 100) if total > 0 else 0
    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'pattern_success': dict(pattern_success),
        'pattern_attempts': dict(pattern_attempts),
        'sequence': sequence
    }

    try:
        with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                    f"Fourgram={result['pattern_success'].get('fourgram', 0)}/{result['pattern_attempts'].get('fourgram', 0)}\n")
    except PermissionError:
        st.error("Unable to write to simulation log.")

    return result

# --- UI Components ---
def render_setup_form():
    """Render the setup form for session configuration."""
    with st.container():
        st.markdown('<div class="card"><h2>Setup Session</h2>', unsafe_allow_html=True)
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10), step=0.10)
            with col2:
                betting_strategy = st.selectbox(
                    "Betting Strategy", STRATEGIES,
                    index=STRATEGIES.index(st.session_state.strategy),
                    help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression. Z1003.1: Resets after first win, stops after three losses."
                )
                target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
            target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
            
            safety_net_enabled = st.checkbox(
                "Enable Safety Net",
                value=st.session_state.safety_net_enabled,
                help="Ensures a percentage of the initial bankroll is preserved after each bet."
            )
            
            safety_net_percentage = st.session_state.safety_net_percentage
            if safety_net_enabled:
                safety_net_percentage = st.number_input(
                    "Safety Net Percentage (%)",
                    min_value=0.0, max_value=50.0, value=st.session_state.safety_net_percentage, step=5.0,
                    help="Percentage of initial bankroll to keep as a safety net."
                )
            
            if st.form_submit_button("Start Session"):
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.10:
                    st.error("Base bet must be at least $0.10.")
                elif base_bet > bankroll:
                    st.error("Base bet cannot exceed bankroll.")
                else:
                    st.session_state.update({
                        'bankroll': bankroll,
                        'base_bet': base_bet,
                        'initial_base_bet': base_bet,
                        'strategy': betting_strategy,
                        'sequence': [],
                        'pending_bet': None,
                        't3_level': 1,
                        't3_results': [],
                        't3_level_changes': 0,
                        't3_peak_level': 1,
                        'parlay_step': 1,
                        'parlay_wins': 0,
                        'parlay_using_base': True,
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
                        'target_mode': target_mode,
                        'target_value': target_value,
                        'initial_bankroll': bankroll,
                        'target_hit': False,
                        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
                        'consecutive_losses': 0,
                        'loss_log': [],
                        'last_was_tie': False,
                        'insights': {},
                        'pattern_volatility': 0.0,
                        'pattern_success': defaultdict(int),
                        'pattern_attempts': defaultdict(int),
                        'safety_net_percentage': safety_net_percentage,
                        'safety_net_enabled': safety_net_enabled
                    })
                    st.session_state.pattern_success['fourgram'] = 0
                    st.session_state.pattern_attempts['fourgram'] = 0
                    st.success(f"Session started with {betting_strategy} strategy!")
        st.markdown('</div>', unsafe_allow_html=True)

def render_result_input():
    """Render the result input buttons."""
    with st.container():
        st.markdown('<div class="card"><h2>Enter Result</h2>', unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            if st.button("Player", key="player_btn"):
                place_result("P")
        with cols[1]:
            if st.button("Banker", key="banker_btn"):
                place_result("B")
        with cols[2]:
            if st.button("Tie", key="tie_btn"):
                place_result("T")
        with cols[3]:
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
                                st.session_state.advice = f"Next Bet: ${amount:.2f} on {pred}"
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
        st.markdown('</div>', unsafe_allow_html=True)

def render_bead_plate():
    """Render the current sequence as a bead plate."""
    with st.container():
        st.markdown('<div class="card bead-plate"><h2>Current Sequence (Bead Plate)</h2>', unsafe_allow_html=True)
        sequence = st.session_state.sequence[-90:]
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
                style = (
                    "width: 24px; height: 24px; border: 1px solid #e2e8f0; border-radius: 50%; background: #f1f5f9;" if result == '' else
                    f"width: 24px; height: 24px; background-color: {'#2563eb' if result == 'P' else '#dc2626' if result == 'B' else '#059669'}; border-radius: 50%;"
                )
                col_html += f"<div style='{style}'></div>"
            col_html += "</div>"
            bead_plate_html += col_html
        bead_plate_html += "</div>"
        st.markdown(bead_plate_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_prediction():
    """Render the current prediction and advice."""
    with st.container():
        st.markdown('<div class="card"><h2>Prediction</h2>', unsafe_allow_html=True)
        if st.session_state.pending_bet:
            amount, side = st.session_state.pending_bet
            color = '#2563eb' if side == 'P' else '#dc2626'
            st.markdown(f"<h4 style='color:{color}; margin: 0;'>Bet: ${amount:.2f} on {side}</h4>", unsafe_allow_html=True)
        elif not st.session_state.target_hit:
            st.info(st.session_state.advice)
        st.markdown('</div>', unsafe_allow_html=True)

def render_insights():
    """Render prediction insights and volatility warnings."""
    with st.container():
        st.markdown('<div class="card"><h2>Prediction Insights</h2>', unsafe_allow_html=True)
        with st.expander("View Details", expanded=True):
            if st.session_state.insights:
                for factor, contribution in st.session_state.insights.items():
                    st.markdown(f"**{factor}**: {contribution}")
            else:
                st.write("No insights available yet.")
            if st.session_state.pattern_volatility > 0.5:
                st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Betting paused)")
        st.markdown('</div>', unsafe_allow_html=True)

def render_status():
    """Render session status information."""
    with st.container():
        st.markdown('<div class="card"><h2>Session Status</h2>', unsafe_allow_html=True)
        st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
        st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
        st.markdown(f"**Safety Net**: {'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}"
                    f"{' | Percentage: ' + str(st.session_state.safety_net_percentage) + '%' if st.session_state.safety_net_enabled else ''}")
        strategy_status = f"**Betting Strategy**: {st.session_state.strategy}"
        if st.session_state.strategy == 'T3':
            strategy_status += f" | Level: {st.session_state.t3_level} | Peak Level: {st.session_state.t3_peak_level} | Level Changes: {st.session_state.t3_level_changes}"
        elif st.session_state.strategy == 'Parlay16':
            strategy_status += f" | Steps: {st.session_state.parlay_step}/16 | Peak Steps: {st.session_state.parlay_peak_step} | Step Changes: {st.session_state.parlay_step_changes} | Consecutive Wins: {st.session_state.parlay_wins}"
        elif st.session_state.strategy == 'Z1003.1':
            strategy_status += f" | Loss Count: {st.session_state.z1003_loss_count} | Level Changes: {st.session_state.z1003_level_changes} | Continue: {st.session_state.z1003_continue}"
        st.markdown(strategy_status)
        st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
        st.markdown(f"**Online Users**: {track_user_session()}")

        if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
            profit = st.session_state.bankroll - st.session_state.initial_bankroll
            units_profit = profit / st.session_state.initial_base_bet
            st.markdown(f"**Units Profit**: {units_profit:.2f} units (${profit:.2f})")
        else:
            st.markdown("**Units Profit**: 0.00 units ($0.00)")
        st.markdown('</div>', unsafe_allow_html=True)

def render_accuracy():
    """Render prediction accuracy metrics and trend chart."""
    with st.container():
        st.markdown('<div class="card"><h2>Prediction Accuracy</h2>', unsafe_allow_html=True)
        total = st.session_state.prediction_accuracy['total']
        if total > 0:
            p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
            b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
            st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
            st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

        st.markdown('<h3>Accuracy Trend</h3>', unsafe_allow_html=True)
        if st.session_state.history:
            accuracy_data = []
            correct = total = 0
            for h in st.session_state.history[-50:]:
                if h['Bet_Placed'] and h['Bet'] in ['P', 'B']:
                    total += 1
                    if h['Win']:
                        correct += 1
                    accuracy_data.append(correct / max(total, 1) * 100)
            if accuracy_data:
                st.line_chart(accuracy_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_loss_log():
    """Render recent loss log."""
    if st.session_state.loss_log:
        with st.container():
            st.markdown('<div class="card"><h2>Recent Losses</h2>', unsafe_allow_html=True)
            st.dataframe([
                {
                    "Sequence": ", ".join(log['sequence']),
                    "Prediction": log['prediction'],
                    "Result": log['result'],
                    "Confidence": f"{log['confidence']}%",
                    "Insights": "; ".join([f"{k}: {v}" for k, v in log['insights'].items()])
                }
                for log in st.session_state.loss_log[-5:]
            ])
            st.markdown('</div>', unsafe_allow_html=True)

def render_history():
    """Render betting history table."""
    if st.session_state.history:
        with st.container():
            st.markdown('<div class="card"><h2>Bet History</h2>', unsafe_allow_html=True)
            n = st.slider("Show last N bets", 5, 50, 10)
            st.dataframe([
                {
                    "Bet": h["Bet"] if h["Bet"] else "-",
                    "Result": h["Result"],
                    "Amount": f"${h['Amount']:.2f}" if h["Bet_Placed"] else "-",
                    "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-",
                    "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-",
                    "Parlay_Step": h["Parlay_Step"] if st.session_state.strategy == 'Parlay16' else "-",
                    "Z1003_Loss_Count": h["Z1003_Loss_Count"] if st.session_state.strategy == 'Z1003.1' else "-",
                }
                for h in st.session_state.history[-n:]
            ])
            st.markdown('</div>', unsafe_allow_html=True)

def render_export():
    """Render session data export option."""
    with st.container():
        st.markdown('<div class="card"><h2>Export Session</h2>', unsafe_allow_html=True)
        if st.button("Download Session Data"):
            csv_data = "Bet,Result,Amount,Win,T3_Level,Parlay_Step,Z1003_Loss_Count\n"
            for h in st.session_state.history:
                csv_data += f"{h['Bet'] or '-'},{h['Result']},${h['Amount']:.2f},{h['Win']},{h['T3_Level']},{h['Parlay_Step']},{h['Z1003_Loss_Count']}\n"
            st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

def render_simulation():
    """Render simulation controls and results."""
    with st.container():
        st.markdown('<div class="card"><h2>Run Simulation</h2>', unsafe_allow_html=True)
        num_hands = st.number_input("Number of Hands to Simulate", min_value=10, max_value=200, value=80, step=10)
        if st.button("Run Simulation"):
            result = simulate_shoe(num_hands)
            st.markdown("**Simulation Results**")
            st.markdown(f"Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']} correct)")
            st.markdown("**Pattern Performance:**")
            for pattern in result['pattern_success']:
                success = result['pattern_success'][pattern]
                attempts = result['pattern_attempts'][pattern]
                st.markdown(f"{pattern}: {success}/{attempts} ({success/attempts*100:.1f}%)" if attempts > 0 else f"{pattern}: 0/0 (0%)")
            st.markdown("Results logged to simulation_log.txt")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Application ---
def main():
    """Main application function."""
    st.title("MANG BACCARAT GROUP")
    initialize_session_state()

    # Split layout into two columns for better organization
    col1, col2 = st.columns([2, 1])

    with col1:
        render_setup_form()
        render_result_input()
        render_bead_plate()
        render_prediction()
        render_insights()
        render_history()

    with col2:
        render_status()
        render_accuracy()
        render_loss_log()
        render_export()
        render_simulation()

if __name__ == "__main__":
    main()
