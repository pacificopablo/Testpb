import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Constants
SHOE_SIZE = 100
HISTORY_LIMIT = 100
OUTCOME_MAP = {'P': 0, 'B': 1, 'T': 2}
REVERSE_OUTCOME_MAP = {0: 'P', 1: 'B', 2: 'T'}

# Session Management
def track_user_session():
    return 1  # Simplified for Streamlit Cloud

def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = track_user_session()
        st.session_state.sequence = []
        st.session_state.bet_history = []
        st.session_state.bankroll = 0.0
        st.session_state.base_bet = 0.0
        st.session_state.bets_placed = 0
        st.session_state.bets_won = 0
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.pending_bet = None
        st.session_state.ai_auto_play = False
        st.session_state.simulation_running = False

# HP Johnson Prediction (Golden Secret Strategy)
def predict_hp_johnson(sequence):
    if not sequence:
        return None, 0

    last_outcome = sequence[-1]
    sequence_length = len(sequence)

    # Check for streak (3 or more identical outcomes)
    streak_length = 0
    streak_outcome = last_outcome
    for i in range(sequence_length - 1, -1, -1):
        if sequence[i] == streak_outcome:
            streak_length += 1
        else:
            break

    # Mode 2: If streak of 3 or more just broke
    if sequence_length >= 4 and streak_length == 1:
        prev_outcome = sequence[-2]
        streak_check_length = 0
        streak_check_outcome = prev_outcome
        for i in range(sequence_length - 2, -1, -1):
            if sequence[i] == streak_check_outcome:
                streak_check_length += 1
            else:
                break
        if streak_check_length >= 3:
            # Streak broke, bet opposite of the streak's outcome
            prediction = 'P' if prev_outcome == 'B' else 'B'
            confidence = 80  # Higher confidence for Mode 2
            return prediction, confidence

    # Mode 1: Bet opposite to the last outcome
    prediction = 'P' if last_outcome == 'B' else 'B'
    confidence = 60  # Base confidence for Mode 1
    return prediction, confidence

# AI Model Training
def train_ml_model(sequence):
    if len(sequence) < 5:
        return None, None
    X, y = [], []
    for i in range(len(sequence) - 4):
        window = sequence[i:i+4]
        next_outcome = sequence[i+4]
        if 'T' not in window and next_outcome != 'T':
            features = [
                OUTCOME_MAP[window[-1]],
                sum(1 for j, outcome in enumerate(sequence[:i+4]) if outcome == 'P'),
                sum(1 for j, outcome in enumerate(sequence[:i+4]) if outcome == 'B'),
                sum(1 for j in range(1, i+4) if sequence[j-1] != sequence[j]),
                sum(1 for j in range(1, i+4) if sequence[j-1] == sequence[j]),
                sequence[:i+4].count('P') / (i+4),
                sequence[:i+4].count('B') / (i+4)
            ]
            X.append(features)
            y.append(OUTCOME_MAP[next_outcome])
    if not X or not y:
        return None, None
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Modified Prediction with Voting
def predict_next_outcome(sequence, model, scaler):
    if len(sequence) < 4:
        return None, 0, {'AI': (None, 0), 'TimeBeforeLast': (None, 0), 'HPJohnson': (None, 0)}

    # AI Prediction
    ai_pred, ai_conf = None, 0
    if model and scaler:
        features = [
            OUTCOME_MAP[sequence[-1]],
            sum(1 for outcome in sequence if outcome == 'P'),
            sum(1 for outcome in sequence if outcome == 'B'),
            sum(1 for j in range(1, len(sequence)) if sequence[j-1] != sequence[j]),
            sum(1 for j in range(1, len(sequence)) if sequence[j-1] == sequence[j]),
            sequence.count('P') / len(sequence),
            sequence.count('B') / len(sequence)
        ]
        X = np.array([features])
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        max_prob_idx = np.argmax(probs)
        ai_pred = REVERSE_OUTCOME_MAP[max_prob_idx]
        ai_conf = probs[max_prob_idx] * 100

    # TimeBeforeLast Prediction
    tbl_pred, tbl_conf = None, 0
    last_positions = defaultdict(list)
    for i, outcome in enumerate(sequence):
        last_positions[outcome].append(i)
    if last_positions['P'] and last_positions['B']:
        time_since_p = len(sequence) - max(last_positions['P'])
        time_since_b = len(sequence) - max(last_positions['B'])
        if time_since_p > time_since_b:
            tbl_pred = 'P'
            tbl_conf = min(80, 50 + 5 * (time_since_p - time_since_b))
        elif time_since_b > time_since_p:
            tbl_pred = 'B'
            tbl_conf = min(80, 50 + 5 * (time_since_b - time_since_p))

    # HP Johnson Prediction
    hp_pred, hp_conf = predict_hp_johnson(sequence)

    # Voting System
    predictions = [ai_pred, tbl_pred, hp_pred]
    confidences = [ai_conf * 0.5, tbl_conf * 0.25, hp_conf * 0.25]
    valid_preds = [(p, c) for p, c in zip(predictions, confidences) if p is not None]
    if not valid_preds:
        return None, 0, {'AI': (ai_pred, ai_conf), 'TimeBeforeLast': (tbl_pred, tbl_conf), 'HPJohnson': (hp_pred, hp_conf)}
    
    # Weighted voting
    vote_counts = Counter(p for p, c in valid_preds if c > 50)
    if not vote_counts:
        return None, 0, {'AI': (ai_pred, ai_conf), 'TimeBeforeLast': (tbl_pred, tbl_conf), 'HPJohnson': (hp_pred, hp_conf)}
    
    final_pred = max(vote_counts, key=lambda p: sum(c for pred, c in valid_preds if pred == p))
    final_conf = sum(c for pred, c in valid_preds if pred == final_pred) / sum(1 for pred, c in valid_preds if pred == final_pred)
    
    return final_pred, final_conf, {'AI': (ai_pred, ai_conf), 'TimeBeforeLast': (tbl_pred, tbl_conf), 'HPJohnson': (hp_pred, hp_conf)}

# UI and Logic (Simplified)
def main():
    st.set_page_config(page_title="Baccarat Predictor", layout="wide")
    initialize_session()

    st.title("Baccarat Predictor")
    
    # Setup Form
    with st.form("setup_form"):
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, step=10.0)
        base_bet = st.number_input("Base Bet ($)", min_value=0.0, step=1.0)
        strategy = st.selectbox("Money Management", ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "Grid", "OscarGrind", "1222"])
        stop_loss = st.number_input("Stop Loss ($)", min_value=0.0, step=10.0)
        win_limit = st.number_input("Win Limit ($)", min_value=0.0, step=10.0)
        submit = st.form_submit_button("Start Session")
        
        if submit:
            if bankroll <= 0 or base_bet <= 0:
                st.error("Bankroll and bet must be positive.")
            elif base_bet > bankroll * 0.05:
                st.error("Bet cannot exceed 5% of bankroll.")
            else:
                st.session_state.bankroll = bankroll
                st.session_state.base_bet = base_bet
                st.session_state.strategy = strategy
                st.session_state.stop_loss = stop_loss
                st.session_state.win_limit = win_limit
                st.session_state.sequence = []
                st.session_state.bet_history = []
                st.session_state.bets_placed = 0
                st.session_state.bets_won = 0
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.model, st.session_state.scaler = None, None
                st.session_state.pending_bet = None
                st.success(f"Session started: Bankroll ${bankroll:.0f}, Bet ${base_bet:.0f}, Strategy: {strategy}")

    # Result Input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player"):
            place_result('P')
    with col2:
        if st.button("Banker"):
            place_result('B')
    with col3:
        if st.button("Tie"):
            place_result('T')
    with col4:
        if st.button("Undo"):
            undo()

    # Bead Plate
    if st.session_state.sequence:
        st.subheader("Bead Plate")
        rows = 6
        cols = 14
        grid = [['' for _ in range(cols)] for _ in range(rows)]
        for i, outcome in enumerate(st.session_state.sequence[-rows*cols:]):
            row = i % rows
            col = i // rows
            grid[row][col] = outcome
        st.write(pd.DataFrame(grid, columns=[f"Col {i+1}" for i in range(cols)]))

    # Prediction
    if st.session_state.sequence:
        model, scaler = st.session_state.model, st.session_state.scaler
        if len(st.session_state.sequence) >= 5:
            model, scaler = train_ml_model(st.session_state.sequence)
            st.session_state.model, st.session_state.scaler = model, scaler
        prediction, confidence, details = predict_next_outcome(st.session_state.sequence, model, scaler)
        st.subheader("Prediction")
        st.write(f"**Final Prediction**: {prediction if prediction else 'None'} ({confidence:.0f}%)")
        st.write(f"AI: {details['AI'][0]} ({details['AI'][1]:.0f}%)")
        st.write(f"TimeBeforeLast: {details['TimeBeforeLast'][0]} ({details['TimeBeforeLast'][1]:.0f}%)")
        st.write(f"HP Johnson: {details['HPJohnson'][0]} ({details['HPJohnson'][1]:.0f}%)")
        if prediction and confidence >= 60:
            bet_amount = calculate_bet_amount()
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, prediction)
                st.write(f"**Advice**: Bet ${bet_amount:.0f} on {prediction}")
            else:
                st.session_state.pending_bet = None
                st.write(f"**Advice**: Skip (bet ${bet_amount:.0f} > bankroll)")
        else:
            st.session_state.pending_bet = None
            st.write(f"**Advice**: Skip (confidence: {confidence:.0f}%)")

    # Status
    st.subheader("Status")
    st.write(f"Bankroll: ${st.session_state.bankroll:.0f}")
    st.write(f"Profit: ${st.session_state.bankroll - st.session_state.initial_bankroll:.0f}")
    st.write(f"Bets Placed: {st.session_state.bets_placed}")
    st.write(f"Bets Won: {st.session_state.bets_won}")
    st.write(f"Win Rate: {st.session_state.bets_won / st.session_state.bets_placed * 100:.1f}%" if st.session_state.bets_placed > 0 else "Win Rate: N/A")
    st.write(f"Sequence: {', '.join(st.session_state.sequence[-10:])}")

    # History
    if st.session_state.bet_history:
        st.subheader("Bet History")
        history_df = pd.DataFrame(
            [(h['Result'], h['Bet Amount'], h['Bet Selection'], h['Outcome'], h['Previous State']['bankroll']) for h in st.session_state.bet_history],
            columns=['Result', 'Bet Amount', 'Bet Selection', 'Outcome', 'Bankroll Before']
        )
        st.dataframe(history_df.tail(10))

def calculate_bet_amount():
    # Simplified; assumes existing logic for strategies like T3, Flatbet, etc.
    if st.session_state.strategy == 'Flatbet':
        return st.session_state.base_bet
    elif st.session_state.strategy == 'T3':
        return st.session_state.base_bet * st.session_state.t3_level
    # Add other strategies as needed
    return st.session_state.base_bet

def place_result(result):
    if st.session_state.bankroll <= 0:
        st.error("Bankroll depleted. Reset session.")
        return
    previous_state = {
        'bankroll': st.session_state.bankroll,
        't3_level': st.session_state.t3_level,
        't3_results': st.session_state.t3_results[:]
    }
    if st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        st.session_state.bets_placed += 1
        if result == bet_selection:
            st.session_state.bets_won += 1
            st.session_state.bankroll += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            if st.session_state.strategy == 'T3':
                if not st.session_state.t3_results:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                st.session_state.t3_results.append('W')
        else:
            st.session_state.bankroll -= bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('L')
        if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []
        st.session_state.bet_history.append({
            'Result': result,
            'Bet Amount': bet_amount,
            'Bet Selection': bet_selection,
            'Outcome': 'win' if result == bet_selection else 'loss',
            'Previous State': previous_state
        })
        st.session_state.pending_bet = None
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > HISTORY_LIMIT:
        st.session_state.sequence.pop(0)
    st.experimental_rerun()

def undo():
    if not st.session_state.sequence:
        return
    st.session_state.sequence.pop()
    if st.session_state.bet_history:
        last_bet = st.session_state.bet_history.pop()
        st.session_state.bankroll = last_bet['Previous State']['bankroll']
        if last_bet['Bet Amount'] > 0:
            st.session_state.bets_placed -= 1
            if last_bet['Outcome'] == 'win':
                st.session_state.bets_won -= 1
        if last_bet['Previous State']['t3_level']:
            st.session_state.t3_level = last_bet['Previous State']['t3_level']
            st.session_state.t3_results = last_bet['Previous State']['t3_results']
    st.session_state.pending_bet = None
    st.experimental_rerun()

if __name__ == "__main__":
    main()
