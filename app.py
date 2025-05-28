
import streamlit as st
import numpy as np
import time

# Normalize input tokens
def normalize_result(s):
    s = s.strip().lower()
    return {'banker': 'Banker', 'b': 'Banker', 'player': 'Player', 'p': 'Player', 'tie': 'Tie', 't': 'Tie'}.get(s, None)

# NumPy-based roadmap updates
def update_big_road(history, state):
    max_rows, max_cols = 6, 15  # Reduced from 20
    if 'big_road_grid' not in state:
        state.big_road_grid = np.full((max_rows, max_cols), '', dtype='U1')
        state.big_road_cols = 0
        state.big_road_row = 0
        state.big_road_last_outcome = None
        state.big_road_last_length = 0

    grid, col, row, last_outcome = state.big_road_grid, state.big_road_cols, state.big_road_row, state.big_road_last_outcome
    start_idx = state.big_road_last_length
    if start_idx > len(history):  # Undo
        grid = np.full((max_rows, max_cols), '', dtype='U1')
        col, row, last_outcome = 0, 0, None
        start_idx = 0

    for i in range(start_idx, len(history)):
        result = history[i]
        mapped = 'P' if result == 'Player' else 'B' if result == 'Banker' else 'T'
        if mapped == 'T':
            if col < max_cols and row < max_rows and grid[row, col] == '':
                grid[row, col] = 'T'
            continue
        if col >= max_cols:
            break
        if last_outcome is None or (mapped == last_outcome and row < max_rows - 1):
            grid[row, col] = mapped
            row += 1
        else:
            col += 1
            row = 0
            if col < max_cols:
                grid[row, col] = mapped
                row += 1
        last_outcome = mapped if mapped != 'T' else last_outcome

    state.big_road_grid, state.big_road_cols, state.big_road_row, state.big_road_last_outcome = grid, col + 1, row, last_outcome
    state.big_road_last_length = len(history)
    return grid, col + 1

def update_derived_road(big_road_grid, num_cols, state, road_type):
    max_rows, max_cols = 6, 15
    key, cols_key, last_length_key = f'{road_type}_grid', f'{road_type}_cols', f'{road_type}_last_length'
    if key not in state:
        state[key] = np.full((max_rows, max_cols), '', dtype='U1')
        state[cols_key] = 0
        state[last_length_key] = 0

    grid, col = state[key], state[cols_key]
    start_col = state[last_length_key]
    if start_col > num_cols:  # Undo
        grid = np.full((max_rows, max_cols), '', dtype='U1')
        col = 0
        start_col = 0

    offset = {'big_eye_boy': 3, 'small_road': 4, 'cockroach_pig': 5}[road_type]
    row = 0
    for c in range(max(start_col + offset, offset), num_cols):
        if col >= max_cols:
            break
        last_col = big_road_grid[:, c - 1]
        compare_col = big_road_grid[:, c - offset]
        last_non_empty = np.where(last_col != '')[0]
        compare_non_empty = np.where(compare_col != '')[0]
        if last_non_empty.size and compare_non_empty.size:
            if last_col[last_non_empty[0]] == compare_col[compare_non_empty[0]]:
                grid[row, col] = 'R'
            else:
                grid[row, col] = 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            break

    state[key], state[cols_key], state[last_length_key] = grid, col + 1 if row > 0 else col, num_cols
    return grid, col + 1 if row > 0 else col

def compute_patterns(recent, history_length, state):
    if 'patterns' not in state or state.patterns_last_length != len(recent):
        freq = {'Banker': 0, 'Player': 0, 'Tie': 0}
        for r in recent:
            freq[r] += 1
        total = len(recent)

        # Streak
        streak_value, streak_length = None, 0
        if recent:
            last = recent[-1]
            streak_length = 1
            for r in recent[-2::-1]:
                if r == last:
                    streak_length += 1
                else:
                    break
            streak_value = last

        # Chop
        chop_4 = len(recent) >= 4 and all(recent[i] != recent[i + 1] for i in range(-4, -1))
        chop_5 = len(recent) >= 5 and all(recent[i] != recent[i + 1] for i in range(-5, -1))

        # Trend
        window = 6 if history_length < 10 else 10 if history_length <= 20 else 15
        recent_window = recent[-window:] if len(recent) >= window else recent
        trend_total = len(recent_window)
        banker_ratio = sum(1 for r in recent_window if r == 'Banker') / trend_total if trend_total else 0
        player_ratio = sum(1 for r in recent_window if r == 'Player') / trend_total if trend_total else 0
        trend_bet = 'Banker' if banker_ratio > player_ratio + 0.2 else 'Player' if player_ratio > banker_ratio + 0.2 else None
        trend_score = (banker_ratio if trend_bet == 'Banker' else player_ratio) * 60 if trend_bet else 0

        state.patterns = {
            'freq': freq, 'total': total,
            'streak_value': streak_value, 'streak_length': streak_length,
            'chop_4': chop_4, 'chop_5': chop_5,
            'trend_bet': trend_bet, 'trend_score': trend_score
        }
        state.patterns_last_length = len(recent)
    return state.patterns

@st.cache_data
def compute_win_loss(history_tuple, ai_mode, strategy):
    start_time = time.time()
    history = list(history_tuple)
    outcomes = ['O'] * len(history)
    for i in range(1, len(history)):
        bet, _, _, _ = advanced_bet_selection(history[:i], ai_mode, strategy)
        actual_result = history[i]
        outcomes[i] = 'N' if bet in ('Pass', None) or actual_result == 'Tie' else 'W' if bet == actual_result else 'L'
    print(f"Win/loss time: {time.time() - start_time:.3f}s")
    return outcomes

def compute_recent_accuracy(history, outcomes):
    recent_outcomes = outcomes[-10:]
    wins = recent_outcomes.count('W')
    total = sum(1 for o in recent_outcomes if o in ['W', 'L'])
    return wins / total if total else 1.0

@st.cache_data
def advanced_bet_selection(results, mode, strategy):
    start_time = time.time()
    max_results = 100  # Limit history
    recent = results[-30:] if len(results) > 30 else results
    if not recent:
        return 'Pass', 0, "No results yet.", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    history_length = min(len(results), max_results)

    # Patterns
    patterns = compute_patterns(recent, history_length, st.session_state)
    freq, total = patterns['freq'], patterns['total']
    is_flat_betting = strategy == "Flat Betting"

    # Flat betting bias
    if is_flat_betting and len(recent) >= 3:
        last_outcome = recent[-1]
        if recent[-3:] == [last_outcome] * 3:
            scores[last_outcome] += 80
            reason_parts.append("Recent 3-hand momentum.")
            pattern_insights.append(f"3-hand momentum: {last_outcome}")
            emotional_tone = "Confident"
        elif patterns['chop_5']:
            alternate_bet = 'Player' if last_outcome == 'Banker' else 'Banker'
            scores[alternate_bet] += 90
            reason_parts.append("5-hand chop detected.")
            pattern_insights.append("5-hand chop")
            emotional_tone = "Excited"

    # Streak
    if patterns['streak_length'] >= 3 and patterns['streak_value'] != 'Tie':
        streak_score = min(80 + (patterns['streak_length'] - 3) * 12, 95)
        scores[patterns['streak_value']] += streak_score
        reason_parts.append(f"Streak of {patterns['streak_length']} {patterns['streak_value']}.")
        pattern_insights.append(f"Streak: {patterns['streak_length']} {patterns['streak_value']}")
        emotional_tone = "Optimistic" if patterns['streak_length'] < 5 else "Confident"
        if patterns['streak_length'] >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if patterns['streak_value'] == 'Banker' else 'Banker'
            scores[contrarian_bet] += 35
            reason_parts.append("Long streak; possible break.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    # Chop
    if patterns['chop_4']:
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        chop_score = 85 if patterns['chop_5'] else 75
        scores[alternate_bet] += chop_score
        reason_parts.append(f"Chop in last {5 if patterns['chop_5'] else 4} hands.")
        pattern_insights.append(f"Chop: {5 if patterns['chop_5'] else 4} hands")
        emotional_tone = "Excited"

    # Trend
    if patterns['trend_bet']:
        scores[patterns['trend_bet']] += patterns['trend_score']
        reason_parts.append(f"Trend favors {patterns['trend_bet']}.")
        pattern_insights.append(f"Trend: {patterns['trend_bet']}")
        emotional_tone = "Hopeful"

    # Roadmaps
    big_road_grid, num_cols = update_big_road(recent, st.session_state)
    if num_cols > 0:
        last_col = big_road_grid[:, num_cols - 1]
        col_length = np.sum(last_col != '')
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            scores[bet_side] += 65
            reason_parts.append(f"Big Road column of {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")

    roadmap_weight = 0.8 if is_flat_betting else 1.0
    if num_cols >= 3:
        big_eye_grid, big_eye_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'big_eye_boy')
        if big_eye_cols > 0:
            last_col = big_eye_grid[:, big_eye_cols - 1]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0, num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0, num_cols - 1] == 'B' else 'Banker'
                scores[last_side if last_signal == 'R' else opposite_side] += (45 if last_signal == 'R' else 40) * roadmap_weight
                reason_parts.append(f"Big Eye Boy: {'Repeat' if last_signal == 'R' else 'Break'}.")
                pattern_insights.append(f"Big Eye Boy: {'Repeat' if last_signal == 'R' else 'Break'}")

    if num_cols >= 4 and max(scores.values()) < 80:  # Skip if high confidence
        small_road_grid, small_road_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'small_road')
        if small_road_cols > 0:
            last_col = small_road_grid[:, small_road_cols - 1]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0, num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0, num_cols - 1] == 'B' else 'Banker'
                scores[last_side if last_signal == 'R' else opposite_side] += (40 if last_signal == 'R' else 35) * roadmap_weight
                reason_parts.append(f"Small Road: {'Repeat' if last_signal == 'R' else 'Break'}.")
                pattern_insights.append(f"Small Road: {'Repeat' if last_signal == 'R' else 'Break'}")

    if num_cols >= 5 and max(scores.values()) < 80:
        cockroach_grid, cockroach_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'cockroach_pig')
        if cockroach_cols > 0:
            last_col = cockroach_grid[:, cockroach_cols - 1]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0, num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0, num_cols - 1] == 'B' else 'Banker'
                scores[last_side if last_signal == 'R' else opposite_side] += (35 if last_signal == 'R' else 30) * roadmap_weight
                reason_parts.append(f"Cockroach Pig: {'Repeat' if last_signal == 'R' else 'Break'}.")
                pattern_insights.append(f"Cockroach Pig: {'Repeat' if last_signal == 'R' else 'Break'}")

    # Frequency
    decay_factor = 0.9 ** (total // 10)
    scores['Banker'] += (freq['Banker'] / total * 0.9) * 40 * decay_factor
    scores['Player'] += (freq['Player'] / total * 1.0) * 40 * decay_factor
    scores['Tie'] += (freq['Tie'] / total * 0.5) * 40 * decay_factor
    reason_parts.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}.")
    pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    # Select bet
    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores.values())), 95)

    # Accuracy
    outcomes = compute_win_loss(tuple(results), mode, strategy)
    accuracy = compute_recent_accuracy(results, outcomes)
    if accuracy < 0.5:
        confidence = max(confidence - 10, 0)
        reason_parts.append("Low recent accuracy.")
        emotional_tone = "Cautious"

    # Thresholds
    confidence_threshold = 55 if is_flat_betting and mode == 'Conservative' else 35 if is_flat_betting and mode == 'Aggressive' else 60 if mode == 'Conservative' else 40
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}%).")
    elif confidence < 70 and mode == 'Conservative':
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence.")

    # Tie restriction
    if bet_choice == 'Tie' and confidence < (85 if is_flat_betting else 80):
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice]), 95)
        reason_parts.append("Tie too risky.")
        emotional_tone = "Cautious"

    # Conflicts
    if len(pattern_insights) > 2 and max(scores.values()) - min(scores.values()) < 20:
        confidence = max(confidence - 15, 40)
        reason_parts.append("Conflicting patterns.")
        emotional_tone = "Skeptical"

    reason = " ".join(reason_parts)
    print(f"Bet selection time: {time.time() - start_time:.3f}s")
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

def money_management(bankroll, base_bet, strategy, confidence=None, history=None):
    start_time = time.time()
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if strategy == "T3" and history and len(history) >= 3:
        recent = history[-3:]
        last_result = recent[-1]
        if all(r == last_result for r in recent) and last_result in ['Player', 'Banker']:
            calculated_bet = base_bet * (4 if len(history) >= 4 and history[-4] == last_result else 2)
        else:
            calculated_bet = base_bet
    elif strategy == "Fixed 5% of Bankroll":
        calculated_bet = bankroll * 0.05
    elif strategy == "Flat Betting":
        calculated_bet = base_bet
    elif strategy == "Confidence-Based":
        confidence_factor = (confidence or 50) / 100.0
        calculated_bet = bankroll * (0.02 + confidence_factor * 0.03)
    else:
        calculated_bet = base_bet

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    print(f"Money management time: {time.time() - start_time:.3f}s")
    return round(bet_size, 2)

def update_bankroll(history, base_bet, strategy, state):
    start_time = time.time()
    if 'bankroll_progress' not in state:
        state.bankroll_progress = []
        state.bet_sizes = []
        state.bankroll_last_length = 0
        state.current_bankroll = state.initial_bankroll

    start_idx = state.bankroll_last_length
    if start_idx > len(history):  # Undo
        start_idx = 0
        state.current_bankroll = state.initial_bankroll
        state.bankroll_progress = []
        state.bet_sizes = []

    for i in range(start_idx, len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _ = advanced_bet_selection(current_rounds[:-1], state.ai_mode, strategy) if i else ('Pass', 0, '', 'Neutral')
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            state.bankroll_progress.append(state.current_bankroll)
            state.bet_sizes.append(0.0)
            continue
        bet_size = money_management(state.current_bankroll, base_bet, strategy, confidence, current_rounds)
        state.bet_sizes.append(bet_size)
        if actual_result == bet:
            state.current_bankroll += bet_size * (0.95 if bet == 'Banker' else 1.0)
        elif actual_result != 'Tie':
            state.current_bankroll -= bet_size
        state.bankroll_progress.append(state.current_bankroll)

    state.bankroll_last_length = len(history)
    print(f"Bankroll update time: {time.time() - start_time:.3f}s")
    return state.bankroll_progress, state.bet_sizes

def render_grid(grid, num_cols, grid_type, state):
    key = f'{grid_type}_html'
    if key not in state or state.get(f'{grid_type}_last_length', 0) != num_cols:
        html_rows = []
        display_cols = min(num_cols, 10)  # Reduced from 12
        for row in range(6):
            row_display = []
            for col in range(display_cols):
                outcome = grid[row, col] if grid_type != 'bead_plate' else grid[row][col]
                if grid_type == 'bead_plate' and outcome in ['P', 'B', 'T']:
                    color = '#3182ce' if outcome == 'P' else '#e53e3e' if outcome == 'B' else '#38a169'
                    row_display.append(f'<div style="width: 22px; height: 22px; background-color: {color}; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                elif outcome == 'P':
                    row_display.append('<div style="width: 22px; height: 22px; background-color: #3182ce; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                elif outcome == 'B' and grid_type == 'big_road':
                    row_display.append('<div style="width: 22px; height: 22px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                elif outcome == 'T':
                    row_display.append('<div style="width: 22px; height: 22px; border: 2px solid #38a169; border-radius: 50%; display: inline-block;"></div>')
                elif outcome == 'R':
                    row_display.append('<div style="width: 22px; height: 22px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                elif outcome == 'B' and grid_type != 'big_road':
                    row_display.append('<div style="width: 22px; height: 22px; background-color: #3182ce; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                else:
                    row_display.append('<div style="width: 22px; height: 22px; display: inline-block;"></div>')
            html_rows.append(' '.join(row_display))
        state[key] = html_rows
        state[f'{grid_type}_last_length'] = num_cols
    return state[key]

def main():
    start_time = time.time()
    st.set_page_config(page_title="Mang Baccarat Group", page_icon="🎲", layout="centered")
    st.title("Mang Baccarat Group")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.update({
            'history': [], 'initial_bankroll': 1000.0, 'base_bet': 10.0,
            'money_management_strategy': "Fixed 5% of Bankroll", 'ai_mode': "Conservative",
            'selected_patterns': ["Bead Plate", "Big Road"]
        })

    # Game Settings
    with st.expander("Game Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
        with col2:
            base_bet = st.number_input("Base Bet", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
        with col3:
            strategy_options = ["Fixed 5% of Bankroll", "Flat Betting", "Confidence-Based", "T3"]
            strategy = st.selectbox("Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
        with col4:
            ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

        if any([
            st.session_state.initial_bankroll != initial_bankroll,
            st.session_state.base_bet != base_bet,
            st.session_state.money_management_strategy != strategy,
            st.session_state.ai_mode != ai_mode
        ]):
            st.session_state.update({
                'initial_bankroll': initial_bankroll, 'base_bet': base_bet,
                'money_management_strategy': strategy, 'ai_mode': ai_mode
            })
            st.rerun()

        st.markdown(f"**Strategy:** {strategy}")

    # Game Input Buttons
    with st.expander("Input Game Results", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Banker"):
                st.session_state.history.append("Banker")
                st.rerun()
        with col2:
            if st.button("Player"):
                st.session_state.history.append("Player")
                st.rerun()
        with col3:
            if st.button("Tie"):
                st.session_state.history.append("Tie")
                st.rerun()
        with col4:
            if st.button("Undo", disabled=len(st.session_state.history) == 0):
                st.session_state.history.pop()
                st.rerun()

    # Shoe Patterns
    patterns_container = st.expander("Shoe Patterns", expanded=False)
    with patterns_container:
        pattern_options = ["Bead Plate", "Big Road", "Big Eye Boy", "Cockroach Pig"]
        selected_patterns = st.multiselect("Select Patterns", pattern_options, default=st.session_state.selected_patterns, key="pattern_select")
        if st.session_state.selected_patterns != selected_patterns:
            st.session_state.selected_patterns = selected_patterns
            st.rerun()

        if "Bead Plate" in selected_patterns:
            st.markdown("### Bead Plate")
            sequence = st.session_state.history[-60:]  # Reduced from 72
            sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence]
            grid = [['' for _ in range(10)] for _ in range(6)]  # 6x10
            for i, result in enumerate(sequence):
                col = i // 6
                row = i % 6
                if col < 10:
                    grid[row][col] = result
            html_rows = render_grid(grid, len(sequence) // 6 + 1 if sequence else 0, 'bead_plate', st.session_state)
            for row in html_rows:
                st.markdown(row, unsafe_allow_html=True)
            if not st.session_state.history:
                st.write("_No results yet._")

            # Win/Loss Tracker
            st.markdown("#### Win/Loss Tracker")
            st.markdown("<p style='font-size: 12px; color: #666666;'>Green (●): Win, Red (●): Loss, Gray (●): Pass/Tie</p>", unsafe_allow_html=True)
            outcomes = compute_win_loss(tuple(st.session_state.history[-100:]), st.session_state.ai_mode, st.session_state.money_management_strategy)[-60:]
            if 'tracker_html' not in st.session_state or st.session_state.get('tracker_last_length', 0) != len(outcomes):
                tracker_row = [
                    '<div style="width: 22px; height: 22px; background-color: #38a169; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>' if o == 'W'
                    else '<div style="width: 22px; height: 22px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>' if o == 'L'
                    else '<div style="width: 22px; height: 22px; background-color: #666666; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>'
                    for o in outcomes
                ]
                st.session_state.tracker_html = ' '.join(tracker_row)
                st.session_state.tracker_last_length = len(outcomes)
            st.markdown(st.session_state.tracker_html, unsafe_allow_html=True)

        for pattern, road_type in [("Big Road", "big_road"), ("Big Eye Boy", "big_eye_boy"), ("Cockroach Pig", "cockroach_pig")]:
            if pattern in selected_patterns:
                st.markdown(f"### {pattern}")
                if pattern != "Big Road":
                    st.markdown("<p style='font-size: 12px; color: #666666;'>Red (●): Repeat Pattern, Blue (●): Break Pattern</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = update_big_road(st.session_state.history[-100:], st.session_state)
                grid, num_cols = (big_road_grid, num_cols) if road_type == "big_road" else update_derived_road(big_road_grid, num_cols, st.session_state, road_type)
                if num_cols > 0:
                    html_rows = render_grid(grid, num_cols, road_type, st.session_state)
                    for row in html_rows:
                        st.markdown(row, unsafe_allow_html=True)
                else:
                    st.write(f"_No {pattern} data yet._")

    # Bet Prediction
    with st.container():
        st.markdown("### Prediction for Next Bet")
        bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history[-100:], st.session_state.ai_mode, st.session_state.money_management_strategy)
        if bet == 'Pass':
            st.warning("No bet! Pattern unclear.")
        else:
            bankroll_progress, _ = update_bankroll(st.session_state.history[-100:], st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state)
            current_bankroll = bankroll_progress[-1] if bankroll_progress else st.session_state.initial_bankroll
            bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy, confidence, st.session_state.history[-100:])
            st.success(f"Bet: **{bet}**  Confidence: **{confidence}%**  Bet Size: **${bet_size:.2f}**  Emotion: **{emotional_tone}**")
        st.info(reason)
        if pattern_insights:
            st.markdown("#### Patterns")
            for insight in pattern_insights:
                st.write(f"- {insight}")

    # Bankroll Progression
    with st.expander("Bankroll and Bet Size Progression", expanded=False):
        bankroll_progress, bet_sizes = update_bankroll(st.session_state.history[-100:], st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state)
        if 'bankroll_html' not in st.session_state or st.session_state.get('bankroll_last_length', 0) != len(bankroll_progress):
            html_output = [
                f"{len(bankroll_progress) - i}: Bankroll ${val:.2f}, Bet: ${bet_size:.2f}" if bet_size > 0
                else f"{len(bankroll_progress) - i}: Bankroll ${val:.2f}, Bet: None"
                for i, (val, bet_size) in enumerate(zip(reversed(bankroll_progress), reversed(bet_sizes)))
            ]
            st.session_state.bankroll_html = html_output
            st.session_state.bankroll_last_length = len(bankroll_progress)

        st.markdown("### Progression (Newest to Oldest)")
        for line in st.session_state.bankroll_html[:50]:  # Limit display
            st.write(f"Hand {line}")
        st.markdown(f"**Current Bankroll:** ${bankroll_progress[-1]:.2f}" if bankroll_progress else f"**Current Bankroll:** ${st.session_state.initial_bankroll:.2f}")

    # Reset Game
    with st.expander("Reset Game", expanded=False):
        if st.button("Reset History and Bankroll"):
            st.session_state.clear()
            st.rerun()

    print(f"Total render time: {time.time() - start_time:.3f}s")
if __name__ == "__main__":
    main()
