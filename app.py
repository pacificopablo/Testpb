import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

# Constants
SHOE_SIZE = 100
HISTORY_LIMIT = 100
OUTCOME_MAP = {'P': 0, 'B': 1, 'T': 2}
REVERSE_OUTCOME_MAP = {0: 'P', 1: 'B', 2: 'T'}
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "Grid", "OscarGrind", "1222"]
PARLAY_TABLE = {i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
    (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
    (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
], 1)}
FOUR_TIER_TABLE = {1: {'step1': 1, 'step2': 3}, 2: {'step1': 7, 'step2': 21},
                   3: {'step1': 50, 'step2': 150}, 4: {'step1': 350, 'step2': 1050}}
FOUR_TIER_MIN_BANKROLL = sum(FOUR_TIER_TABLE[tier][step] for tier in FOUR_TIER_TABLE for step in FOUR_TIER_TABLE[tier])
FLATBET_LEVELUP_TABLE = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
FLATBET_LEVELUP_MIN_BANKROLL = sum(FLATBET_LEVELUP_TABLE[level] * 5 for level in FLATBET_LEVELUP_TABLE)
FLATBET_LEVELUP_THRESHOLDS = {1: -5.0, 2: -10.0, 3: -20.0, 4: -40.0, 5: -40.0}
GRID = [
    [0, 1, 2, 3, 4, 4, 3, 2, 1], [1, 0, 1, 3, 4, 4, 4, 3, 2], [2, 1, 0, 2, 3, 4, 5, 4, 3],
    [3, 3, 2, 0, 2, 4, 5, 6, 5], [4, 4, 3, 2, 0, 2, 5, 7, 7], [4, 4, 4, 4, 2, 0, 3, 7, 9],
    [3, 4, 5, 5, 5, 3, 0, 5, 9], [2, 3, 4, 6, 7, 7, 5, 0, 8], [1, 2, 3, 5, 7, 9, 9, 8, 0],
    [1, 1, 2, 3, 5, 8, 11, 15, 15], [0, 0, 1, 2, 4, 8, 15, 15, 30]
]
GRID_MIN_BANKROLL = max(max(row) for row in GRID) * 5
_1222_MIN_BANKROLL = 10

# Session Management
def track_user_session():
    return 1  # Simplified for Streamlit Cloud

def initialize_session():
    defaults = {
        'session_id': track_user_session(),
        'sequence': [],
        'bet_history': [],
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_bankroll': 0.0,
        'peak_bankroll': 0.0,
        'bets_placed': 0,
        'bets_won': 0,
        't3_level': 1,
        't3_results': [],
        'model': None,
        'scaler': None,
        'pending_bet': None,
        'ai_auto_play': False,
        'simulation_running': False,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'moon_level': 1,
        'moon_level_changes': 0,
        'moon_peak_level': 1,
        'four_tier_level': 1,
        'four_tier_step': 1,
        'four_tier_losses': 0,
        'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0.0,
        'grid_pos': [0, 0],
        'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1,
        'current_streak': 0,
        'current_streak_type': None,
        'longest_streak': 0,
        'longest_streak_type': None,
        'current_chop_count': 0,
        'longest_chop': 0,
        'level_1222': 1,
        'next_bet_multiplier_1222': 1,
        'rounds_1222': 0,
        'level_start_bankroll_1222': 0.0,
        'last_positions': {'P': [], 'B': [], 'T': []},
        'time_before_last': {'P': 0, 'B': 0, 'T': 0},
        'shoe_completed': False,
        'safety_net_enabled': True,
        'safety_net_percentage': 0.02,
        'stop_loss_percentage': 1.0,
        'win_limit': 1.5,
        'target_profit_option': 'None',
        'target_profit_percentage': 0.0,
        'target_profit_units': 0.0,
        'strategy': 'T3'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
            prediction = 'P' if prev_outcome == 'B' else 'B'
            confidence = 80
            return prediction, confidence

    # Mode 1: Bet opposite to the last outcome
    prediction = 'P' if last_outcome == 'B' else 'B'
    confidence = 60
    return prediction, confidence

# AI Model Training
def train_ml_model(sequence):
    if len(sequence) < 5:
        return None, None
    X, y = [], []
    for i in range(len(sequence) - 4):
        window = sequence[i:i+4]
        next_outcome = sequence[i+4]
        features = [OUTCOME_MAP[window[j]] for j in range(4)] + [
            st.session_state.time_before_last.get(k, len(sequence) + 1) / (len(sequence) + 1)
            for k in ['P', 'B']
        ] + [st.session_state.current_streak / 10.0, st.session_state.current_chop_count / 10.0,
             st.session_state.bets_won / max(st.session_state.bets_placed, 1)]
        X.append(features)
        y.append(OUTCOME_MAP[next_outcome])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Modified Prediction with Voting
def predict_next_outcome(sequence, model, scaler):
    if len(sequence) < 4:
        return None, 0, {'AI': (None, 0), 'TimeBeforeLast': (None, 0), 'HPJohnson': (None, 0)}

    # AI Prediction
    ai_pred, ai_conf = None, 0
    if model and scaler:
        features = [OUTCOME_MAP[sequence[-1]]] + [
            sum(1 for outcome in sequence if outcome == 'P'),
            sum(1 for outcome in sequence if outcome == 'B'),
            sum(1 for j in range(1, len(sequence)) if sequence[j-1] != sequence[j]),
            sum(1 for j in range(1, len(sequence)) if sequence[j-1] == sequence[j]),
            sequence.count('P') / len(sequence),
            sequence.count('B') / len(sequence)
        ]
        X_scaled = scaler.transform([features])
        probs = model.predict_proba(X_scaled)[0]
        max_prob_idx = np.argmax(probs)
        ai_pred = REVERSE_OUTCOME_MAP[max_prob_idx]
        ai_conf = probs[max_prob_idx] * 100

    # TimeBeforeLast Prediction
    tbl_pred, tbl_conf = None, 0
    tbl_values = {k: st.session_state.time_before_last.get(k, len(sequence) + 1) for k in ['P', 'B', 'T']}
    max_tbl = max(tbl_values.values(), default=1)
    tbl_weights = {k: (max_tbl - v + 1) / max_tbl if v <= len(sequence) else 0.0 for k, v in tbl_values.items()}
    tbl_pred = min(tbl_values, key=tbl_values.get)
    tbl_conf = tbl_weights[tbl_pred] * 100

    # HP Johnson Prediction
    hp_pred, hp_conf = predict_hp_johnson([x for x in sequence if x in ['P', 'B']])

    # Voting System
    predictions = [(ai_pred, ai_conf * 0.5), (tbl_pred, tbl_conf * 0.25), (hp_pred, hp_conf * 0.25)]
    valid_preds = [(p, c) for p, c in predictions if p in ['P', 'B'] and c > 50]
    if not valid_preds:
        return None, 0, {'AI': (ai_pred, ai_conf), 'TimeBeforeLast': (tbl_pred, tbl_conf), 'HPJohnson': (hp_pred, hp_conf)}
    
    vote_counts = Counter(p for p, c in valid_preds)
    final_pred = max(vote_counts, key=lambda p: sum(c for pred, c in valid_preds if pred == p))
    final_conf = sum(c for pred, c in valid_preds if pred == final_pred) / len([p for p, c in valid_preds if p == final_pred])
    
    return final_pred, final_conf, {'AI': (ai_pred, ai_conf), 'TimeBeforeLast': (tbl_pred, tbl_conf), 'HPJohnson': (hp_pred, hp_conf)}

# Betting Logic
def calculate_bet_amount(bet_selection):
    try:
        if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
            return st.session_state.base_bet
        if st.session_state.strategy == 'Flatbet':
            return st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            return st.session_state.base_bet * st.session_state.t3_level
        elif st.session_state.strategy == 'Parlay16':
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            return st.session_state.base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
        elif st.session_state.strategy == 'Moon':
            return st.session_state.base_bet * st.session_state.moon_level
        elif st.session_state.strategy == 'FourTier':
            step_key = 'step1' if st.session_state.four_tier_step == 1 else 'step2'
            return st.session_state.base_bet * FOUR_TIER_TABLE[st.session_state.four_tier_level][step_key]
        elif st.session_state.strategy == 'FlatbetLevelUp':
            return st.session_state.base_bet * FLATBET_LEVELUP_TABLE[st.session_state.flatbet_levelup_level]
        elif st.session_state.strategy == 'Grid':
            return st.session_state.base_bet * GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]]
        elif st.session_state.strategy == 'OscarGrind':
            return st.session_state.base_bet * st.session_state.oscar_current_bet_level
        elif st.session_state.strategy == '1222':
            return st.session_state.base_bet * st.session_state.level_1222 * st.session_state.next_bet_multiplier_1222
        return 0.0
    except:
        return 0.0

def simulate_shoe_result():
    return random.choices(['P', 'B', 'T'], weights=[0.4586, 0.4460, 0.0954], k=1)[0]

def place_result(result):
    if st.session_state.bankroll <= 0:
        st.error("Bankroll depleted. Reset session.")
        return

    # Check limits
    if st.session_state.stop_loss_percentage > 0 and st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage:
        reset_session()
        st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}%. Game reset.")
        return
    if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
        reset_session()
        st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}%. Game reset.")
        return
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0 and profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
        reset_session()
        st.success(f"Target profit reached: ${profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}%). Game reset.")
        return
    if st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0 and profit >= st.session_state.target_profit_units:
        reset_session()
        st.success(f"Target profit reached: ${profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset.")
        return

    # Save state for undo
    previous_state = {
        'bankroll': st.session_state.bankroll,
        't3_level': st.session_state.t3_level,
        't3_results': st.session_state.t3_results[:],
        'parlay_step': st.session_state.parlay_step,
        'parlay_wins': st.session_state.parlay_wins,
        'parlay_using_base': st.session_state.parlay_using_base,
        'parlay_step_changes': st.session_state.parlay_step_changes,
        'parlay_peak_step': st.session_state.parlay_peak_step,
        'moon_level': st.session_state.moon_level,
        'moon_level_changes': st.session_state.moon_level_changes,
        'moon_peak_level': st.session_state.moon_peak_level,
        'four_tier_level': st.session_state.four_tier_level,
        'four_tier_step': st.session_state.four_tier_step,
        'four_tier_losses': st.session_state.four_tier_losses,
        'flatbet_levelup_level': st.session_state.flatbet_levelup_level,
        'flatbet_levelup_net_loss': st.session_state.flatbet_levelup_net_loss,
        'grid_pos': st.session_state.grid_pos[:],
        'oscar_cycle_profit': st.session_state.oscar_cycle_profit,
        'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
        'level_1222': st.session_state.level_1222,
        'next_bet_multiplier_1222': st.session_state.next_bet_multiplier_1222,
        'rounds_1222': st.session_state.rounds_1222,
        'level_start_bankroll_1222': st.session_state.level_start_bankroll_1222,
        'current_streak': st.session_state.current_streak,
        'current_streak_type': st.session_state.current_streak_type,
        'longest_streak': st.session_state.longest_streak,
        'longest_streak_type': st.session_state.longest_streak_type,
        'current_chop_count': st.session_state.current_chop_count,
        'longest_chop': st.session_state.longest_chop,
        'last_positions': st.session_state.last_positions.copy(),
        'time_before_last': st.session_state.time_before_last.copy()
    }

    # Update streak/chop
    if result in ['P', 'B']:
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']] + [result]
        if len(valid_sequence) == 1 or st.session_state.current_streak_type != result:
            st.session_state.current_streak = 1
            st.session_state.current_streak_type = result
        else:
            st.session_state.current_streak += 1
        if st.session_state.current_streak > st.session_state.longest_streak:
            st.session_state.longest_streak = st.session_state.current_streak
            st.session_state.longest_streak_type = result
        if len(valid_sequence) > 1 and valid_sequence[-2] != result:
            st.session_state.current_chop_count += 1
        else:
            st.session_state.current_chop_count = 0
        if st.session_state.current_chop_count > st.session_state.longest_chop:
            st.session_state.longest_chop = st.session_state.current_chop_count
    else:
        st.session_state.current_streak = 0
        st.session_state.current_streak_type = None
        if st.session_state.current_chop_count > st.session_state.longest_chop:
            st.session_state.longest_chop = st.session_state.current_chop_count
        st.session_state.current_chop_count = 0

    # Resolve bet
    bet_amount = 0
    bet_selection = None
    bet_outcome = None
    if st.session_state.pending_bet and result in ['P', 'B']:
        bet_amount, bet_selection = st.session_state.pending_bet
        st.session_state.bets_placed += 1
        if result == bet_selection:
            winnings = bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            st.session_state.bankroll += winnings
            st.session_state.bets_won += 1
            bet_outcome = 'win'
            if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                if st.session_state.strategy == 'T3':
                    if not st.session_state.t3_results:
                        st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
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
                elif st.session_state.strategy == 'Moon':
                    st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                elif st.session_state.strategy == 'FourTier':
                    st.session_state.four_tier_level = 1
                    st.session_state.four_tier_step = 1
                    st.session_state.four_tier_losses = 0
                    st.session_state.shoe_completed = True
                elif st.session_state.strategy == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                elif st.session_state.strategy == 'Grid':
                    st.session_state.grid_pos[1] += 1
                    if st.session_state.grid_pos[1] >= len(GRID[0]):
                        st.session_state.grid_pos[1] = 0
                        if st.session_state.grid_pos[0] < len(GRID) - 1:
                            st.session_state.grid_pos[0] += 1
                    if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                        st.session_state.grid_pos = [0, 0]
                elif st.session_state.strategy == 'OscarGrind':
                    st.session_state.oscar_cycle_profit += winnings
                    if st.session_state.oscar_cycle_profit >= st.session_state.base_bet:
                        st.session_state.oscar_current_bet_level = 1
                        st.session_state.oscar_cycle_profit = 0.0
                    else:
                        next_bet_level = st.session_state.oscar_current_bet_level + 1
                        potential_winnings = st.session_state.base_bet * next_bet_level * (0.95 if bet_selection == 'B' else 1.0)
                        if st.session_state.oscar_cycle_profit + potential_winnings > st.session_state.base_bet:
                            next_bet_level = max(1, int((st.session_state.base_bet - st.session_state.oscar_cycle_profit) / (st.session_state.base_bet * (0.95 if bet_selection == 'B' else 1.0)) + 0.99))
                        st.session_state.oscar_current_bet_level = next_bet_level
                elif st.session_state.strategy == '1222':
                    st.session_state.next_bet_multiplier_1222 = 2
        else:
            st.session_state.bankroll -= bet_amount
            bet_outcome = 'loss'
            if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('L')
                elif st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins = 0
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                    st.session_state.parlay_using_base = True
                    if old_step != st.session_state.parlay_step:
                        st.session_state.parlay_step_changes += 1
                    st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                elif st.session_state.strategy == 'Moon':
                    old_level = st.session_state.moon_level
                    st.session_state.moon_level += 1
                    if old_level != st.session_state.moon_level:
                        st.session_state.moon_level_changes += 1
                    st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                elif st.session_state.strategy == 'FourTier':
                    st.session_state.four_tier_losses += 1
                    if st.session_state.four_tier_losses == 1:
                        st.session_state.four_tier_step = 2
                    elif st.session_state.four_tier_losses >= 2:
                        st.session_state.four_tier_level = min(st.session_state.four_tier_level + 1, 4)
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                elif st.session_state.strategy == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
                    current_level = st.session_state.flatbet_levelup_level
                    if current_level < 5 and st.session_state.flatbet_levelup_net_loss <= FLATBET_LEVELUP_THRESHOLDS[current_level]:
                        st.session_state.flatbet_levelup_level = min(st.session_state.flatbet_levelup_level + 1, 5)
                        st.session_state.flatbet_levelup_net_loss = 0.0
                elif st.session_state.strategy == 'Grid':
                    st.session_state.grid_pos[0] += 1
                    if st.session_state.grid_pos[0] >= len(GRID):
                        st.session_state.grid_pos = [0, 0]
                    if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                        st.session_state.grid_pos = [0, 0]
                elif st.session_state.strategy == '1222':
                    st.session_state.next_bet_multiplier_1222 = 1
        if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1 if wins > losses else st.session_state.t3_level + 1 if losses > wins else st.session_state.t3_level)
            st.session_state.t3_results = []
        if st.session_state.strategy == '1222' and bet_amount > 0:
            st.session_state.rounds_1222 += 1
            if st.session_state.rounds_1222 >= 5:
                if st.session_state.bankroll >= st.session_state.peak_bankroll:
                    st.session_state.level_1222 = 1
                    st.session_state.next_bet_multiplier_1222 = 1
                    st.session_state.rounds_1222 = 0
                    st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                elif st.session_state.bankroll > st.session_state.level_start_bankroll_1222:
                    st.session_state.level_1222 = max(1, st.session_state.level_1222 - 1)
                    st.session_state.next_bet_multiplier_1222 = 1
                    st.session_state.rounds_1222 = 0
                    st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                else:
                    st.session_state.level_1222 += 1
                    st.session_state.next_bet_multiplier_1222 = 1
                    st.session_state.rounds_1222 = 0
                    st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
        st.session_state.peak_bankroll = max(st.session_state.peak_bankroll, st.session_state.bankroll)
        st.session_state.pending_bet = None

    # Add result
    if result in ['P', 'B', 'T']:
        st.session_state.sequence.append(result)
        current_position = len(st.session_state.sequence)
        st.session_state.last_positions[result].append(current_position)
        if len(st.session_state.last_positions[result]) > 2:
            st.session_state.last_positions[result].pop(0)
        for outcome in ['P', 'B', 'T']:
            if len(st.session_state.last_positions[outcome]) >= 2:
                st.session_state.time_before_last[outcome] = current_position - st.session_state.last_positions[outcome][-2]
            else:
                st.session_state.time_before_last[outcome] = current_position + 1

    # Train model
    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
    if len(valid_sequence) >= 5:
        st.session_state.model, st.session_state.scaler = train_ml_model(valid_sequence)

    # Log history
    st.session_state.bet_history.append({
        'Result': result,
        'Bet Amount': bet_amount,
        'Bet Selection': bet_selection,
        'Outcome': bet_outcome,
        'Previous State': previous_state
    })
    if len(st.session_state.bet_history) > HISTORY_LIMIT:
        st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

    # Predict next
    if len(valid_sequence) < 4:
        st.session_state.pending_bet = None
        st.session_state.advice = "Need 4 more Player or Banker results"
    else:
        prediction, confidence, details = predict_next_outcome(valid_sequence, st.session_state.model, st.session_state.scaler)
        strategy_used = []
        if details['AI'][0] == prediction:
            strategy_used.append('AI')
        if details['TimeBeforeLast'][0] == prediction:
            strategy_used.append('TimeBeforeLast')
        if details['HPJohnson'][0] == prediction:
            strategy_used.append('HPJohnson')
        strategy_used = '+'.join(strategy_used)
        if prediction in ['P', 'B'] and confidence >= 60:
            bet_amount = calculate_bet_amount(prediction)
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, prediction)
                strategy_info = f"{st.session_state.strategy}"
                if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                    strategy_info = "Safety Net (Flatbet)"
                elif st.session_state.strategy == 'T3':
                    strategy_info += f" Level {st.session_state.t3_level}"
                elif st.session_state.strategy == 'Parlay16':
                    strategy_info += f" Step {st.session_state.parlay_step}/16"
                elif st.session_state.strategy == 'Moon':
                    strategy_info += f" Level {st.session_state.moon_level}"
                elif st.session_state.strategy == 'FourTier':
                    strategy_info += f" Level {st.session_state.four_tier_level} Step {st.session_state.four_tier_step}"
                elif st.session_state.strategy == 'FlatbetLevelUp':
                    strategy_info += f" Level {st.session_state.flatbet_levelup_level}"
                elif st.session_state.strategy == 'Grid':
                    strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                elif st.session_state.strategy == 'OscarGrind':
                    strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level}"
                elif st.session_state.strategy == '1222':
                    strategy_info += f" Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222}, Bet: {st.session_state.next_bet_multiplier_1222 * st.session_state.level_1222}u"
                st.session_state.advice = f"Bet ${bet_amount:.2f} on {prediction} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
        else:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% or Tie)"

def reset_session():
    setup_values = {
        'bankroll': st.session_state.bankroll,
        'base_bet': st.session_state.base_bet,
        'initial_bankroll': st.session_state.initial_bankroll,
        'peak_bankroll': st.session_state.bankroll,
        'strategy': st.session_state.strategy,
        'stop_loss_percentage': st.session_state.stop_loss_percentage,
        'win_limit': st.session_state.win_limit,
        'safety_net_enabled': st.session_state.safety_net_enabled,
        'safety_net_percentage': st.session_state.safety_net_percentage,
        'target_profit_option': st.session_state.target_profit_option,
        'target_profit_percentage': st.session_state.target_profit_percentage,
        'target_profit_units': st.session_state.target_profit_units,
        'ai_auto_play': st.session_state.ai_auto_play
    }
    initialize_session()
    st.session_state.update(setup_values)
    st.session_state.update({
        'sequence': [],
        'bet_history': [],
        'pending_bet': None,
        'bets_placed': 0,
        'bets_won': 0,
        't3_level': 1,
        't3_results': [],
        'shoe_completed': False,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'moon_level': 1,
        'moon_level_changes': 0,
        'moon_peak_level': 1,
        'four_tier_level': 1,
        'four_tier_step': 1,
        'four_tier_losses': 0,
        'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0.0,
        'grid_pos': [0, 0],
        'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1,
        'current_streak': 0,
        'current_streak_type': None,
        'longest_streak': 0,
        'longest_streak_type': None,
        'current_chop_count': 0,
        'longest_chop': 0,
        'level_1222': 1,
        'next_bet_multiplier_1222': 1,
        'rounds_1222': 0,
        'level_start_bankroll_1222': setup_values.get('bankroll', 0.0),
        'last_positions': {'P': [], 'B': [], 'T': []},
        'time_before_last': {'P': 0, 'B': 0, 'T': 0}
    })

def undo():
    if not st.session_state.sequence:
        st.warning("No results to undo.")
        return
    last_bet = st.session_state.bet_history.pop() if st.session_state.bet_history else None
    st.session_state.sequence.pop()
    if last_bet:
        for key, value in last_bet["Previous State"].items():
            st.session_state[key] = value
        if last_bet["Bet Amount"] > 0:
            st.session_state.bets_placed -= 1
            if last_bet["Outcome"] == 'win':
                st.session_state.bets_won -= 1
    current_position = len(st.session_state.sequence)
    last_result = last_bet["Result"] if last_bet else None
    if last_result and last_result in st.session_state.last_positions and st.session_state.last_positions[last_result]:
        st.session_state.last_positions[last_result].pop()
    for outcome in ['P', 'B', 'T']:
        if len(st.session_state.last_positions[outcome]) >= 2:
            st.session_state.time_before_last[outcome] = current_position - st.session_state.last_positions[outcome][-2]
        else:
            st.session_state.time_before_last[outcome] = current_position + 1
    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
    if len(valid_sequence) < 4:
        st.session_state.pending_bet = None
        st.session_state.advice = "Need 4 more Player or Banker results"
    else:
        if len(valid_sequence) >= 5:
            st.session_state.model, st.session_state.scaler = train_ml_model(valid_sequence)
        prediction, confidence, details = predict_next_outcome(valid_sequence, st.session_state.model, st.session_state.scaler)
        strategy_used = []
        if details['AI'][0] == prediction:
            strategy_used.append('AI')
        if details['TimeBeforeLast'][0] == prediction:
            strategy_used.append('TimeBeforeLast')
        if details['HPJohnson'][0] == prediction:
            strategy_used.append('HPJohnson')
        strategy_used = '+'.join(strategy_used)
        if prediction in ['P', 'B'] and confidence >= 60:
            bet_amount = calculate_bet_amount(prediction)
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, prediction)
                strategy_info = f"{st.session_state.strategy}"
                if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                    strategy_info = "Safety Net (Flatbet)"
                elif st.session_state.strategy == 'T3':
                    strategy_info += f" Level {st.session_state.t3_level}"
                elif st.session_state.strategy == 'Parlay16':
                    strategy_info += f" Step {st.session_state.parlay_step}/16"
                elif st.session_state.strategy == 'Moon':
                    strategy_info += f" Level {st.session_state.moon_level}"
                elif st.session_state.strategy == 'FourTier':
                    strategy_info += f" Level {st.session_state.four_tier_level} Step {st.session_state.four_tier_step}"
                elif st.session_state.strategy == 'FlatbetLevelUp':
                    strategy_info += f" Level {st.session_state.flatbet_levelup_level}"
                elif st.session_state.strategy == 'Grid':
                    strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                elif st.session_state.strategy == 'OscarGrind':
                    strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level}"
                elif st.session_state.strategy == '1222':
                    strategy_info += f" Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222}, Bet: {st.session_state.next_bet_multiplier_1222 * st.session_state.level_1222}u"
                st.session_state.advice = f"Bet ${bet_amount:.2f} on {prediction} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
        else:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% or Tie)"
    st.success("Undone last action.")

# UI
def main():
    st.set_page_config(page_title="Baccarat Predictor", layout="wide")
    initialize_session()

    st.title("MANG BACCARAT GROUP")

    # CSS Styling
    st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    h1 { color: #1a3c6e; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem; }
    .stButton > button { background: #1a3c6e; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    # Setup Form
    with st.form("setup_form"):
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, step=10.0, value=1000.0)
        base_bet = st.number_input("Base Bet ($)", min_value=0.0, step=1.0, value=10.0)
        strategy = st.selectbox("Money Management", STRATEGIES, index=STRATEGIES.index('T3'))
        stop_loss_percentage = st.number_input("Stop Loss (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0) / 100
        win_limit = st.number_input("Win Limit (Multiple of Bankroll)", min_value=1.0, value=1.5, step=0.5)
        safety_net_enabled = st.checkbox("Enable Safety Net", value=True)
        safety_net_percentage = st.number_input("Safety Net (%)", min_value=0.0, max_value=100.0, value=2.0, step=1.0) / 100
        target_mode = st.selectbox("Target Profit Mode", ["None", "Profit %", "Units"])
        target_value = 0.0
        if target_mode == "Profit %":
            target_value = st.number_input("Target Profit (%)", min_value=0.0, value=10.0, step=5.0) / 100
        elif target_mode == "Units":
            target_value = st.number_input("Target Profit (Units)", min_value=0.0, value=100.0, step=10.0)
        ai_auto_play = st.checkbox("Enable AI Auto-Play", value=False)
        submit = st.form_submit_button("Start Session")

        if submit:
            min_bankroll = {
                "T3": base_bet * 3,
                "Flatbet": base_bet * 5,
                "Parlay16": base_bet * 190,
                "Moon": base_bet * 10,
                "FourTier": base_bet * FOUR_TIER_MIN_BANKROLL,
                "FlatbetLevelUp": base_bet * FLATBET_LEVELUP_MIN_BANKROLL,
                "Grid": base_bet * GRID_MIN_BANKROLL,
                "OscarGrind": base_bet * 10,
                "1222": base_bet * _1222_MIN_BANKROLL
            }
            if bankroll < min_bankroll[strategy]:
                st.error(f"Bankroll must be at least ${min_bankroll[strategy]:.2f} for {strategy}.")
            elif base_bet <= 0:
                st.error("Base bet must be greater than 0.")
            else:
                reset_session()
                st.session_state.update({
                    'bankroll': bankroll,
                    'base_bet': base_bet,
                    'initial_bankroll': bankroll,
                    'peak_bankroll': bankroll,
                    'strategy': strategy,
                    'stop_loss_percentage': stop_loss_percentage,
                    'win_limit': win_limit,
                    'safety_net_enabled': safety_net_enabled,
                    'safety_net_percentage': safety_net_percentage,
                    'target_profit_option': target_mode,
                    'target_profit_percentage': target_value if target_mode == "Profit %" else 0.0,
                    'target_profit_units': target_value if target_mode == "Units" else 0.0,
                    'ai_auto_play': ai_auto_play,
                    'level_start_bankroll_1222': bankroll
                })
                st.success(f"Session started: Bankroll ${bankroll:.0f}, Bet ${base_bet:.0f}, Strategy: {strategy}")
                if ai_auto_play:
                    for _ in range(SHOE_SIZE):
                        if st.session_state.shoe_completed:
                            break
                        result = simulate_shoe_result()
                        place_result(result)
                    st.session_state.shoe_completed = True
                    st.rerun()

    # Result Input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player", disabled=st.session_state.bankroll <= 0 or st.session_state.ai_auto_play or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled)):
            place_result('P')
            st.rerun()
    with col2:
        if st.button("Banker", disabled=st.session_state.bankroll <= 0 or st.session_state.ai_auto_play or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled)):
            place_result('B')
            st.rerun()
    with col3:
        if st.button("Tie", disabled=st.session_state.bankroll <= 0 or st.session_state.ai_auto_play or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled)):
            place_result('T')
            st.rerun()
    with col4:
        if st.button("Undo", disabled=not st.session_state.sequence or st.session_state.ai_auto_play or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled)):
            undo()
            st.rerun()
    if st.session_state.shoe_completed and st.button("Reset and Start New Shoe"):
        reset_session()
        st.session_state.shoe_completed = False
        st.rerun()

    # Bead Plate
    if st.session_state.sequence:
        st.subheader("Bead Plate")
        sequence = st.session_state.sequence[-84:]
        grid = [['' for _ in range(14)] for _ in range(6)]
        for i, result in enumerate(sequence):
            col = i // 6
            row = i % 6
            if col < 14:
                color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)

    # Prediction
    if st.session_state.bankroll > 0:
        st.subheader("Prediction")
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
        if len(valid_sequence) < 4:
            st.write("Need 4 more Player or Banker results")
        else:
            prediction, confidence, details = predict_next_outcome(valid_sequence, st.session_state.model, st.session_state.scaler)
            st.markdown(f"**Final Prediction**: {prediction if prediction else 'None'} ({confidence:.0f}%)")
            st.write(f"AI: {details['AI'][0] if details['AI'][0] else 'None'} ({details['AI'][1]:.0f}%)")
            st.write(f"TimeBeforeLast: {details['TimeBeforeLast'][0] if details['TimeBeforeLast'][0] else 'None'} ({details['TimeBeforeLast'][1]:.0f}%)")
            st.write(f"HP Johnson: {details['HPJohnson'][0] if details['HPJohnson'][0] else 'None'} ({details['HPJohnson'][1]:.0f}%)")
            st.write(f"**Advice**: {st.session_state.advice}")

    # Status
    st.subheader("Status")
    if st.session_state.bankroll == 0:
        st.write("Bankroll: $0.00")
        st.write("Profit: N/A")
    else:
        st.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
        st.write(f"Profit: ${st.session_state.bankroll - st.session_state.initial_bankroll:.2f}")
    st.write(f"Bets Placed: {st.session_state.bets_placed}")
    st.write(f"Bets Won: {st.session_state.bets_won}")
    st.write(f"Win Rate: {st.session_state.bets_won / st.session_state.bets_placed * 100:.1f}%" if st.session_state.bets_placed > 0 else "Win Rate: N/A")
    st.write(f"Sequence: {', '.join(st.session_state.sequence[-10:])}")
    st.write(f"Strategy: {st.session_state.strategy}")
    tbl_display = {k: f"{v}" if v <= len(st.session_state.sequence) else "N/A" for k, v in st.session_state.time_before_last.items()}
    st.markdown(
        f"**Time Before Last**:<br>P: {tbl_display['P']} hands<br>B: {tbl_display['B']} hands<br>T: {tbl_display['T']} hands",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**Streak**: {st.session_state.current_streak} ({st.session_state.current_streak_type or 'None'})<br>"
        f"**Longest Streak**: {st.session_state.longest_streak} ({st.session_state.longest_streak_type or 'None'})<br>"
        f"**Chop**: {st.session_state.current_chop_count}<br>**Longest Chop**: {st.session_state.longest_chop}",
        unsafe_allow_html=True
    )

    # History
    if st.session_state.bet_history:
        st.subheader("Bet History")
        n = st.slider("Show last N bets", 5, 50, 10)
        history_df = pd.DataFrame(
            [(h['Result'], f"${h['Bet Amount']:.2f}", h['Bet Selection'], h['Outcome'], f"${h['Previous State']['bankroll']:.2f}") for h in st.session_state.bet_history[-n:]],
            columns=['Result', 'Bet Amount', 'Bet Selection', 'Outcome', 'Bankroll Before']
        )
        st.dataframe(history_df)

if __name__ == "__main__":
    main()
