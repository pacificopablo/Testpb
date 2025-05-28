import streamlit as st
import time

# Normalize input tokens
def normalize_result(s):
    s = s.strip().lower()
    if s == 'banker' or s == 'b':
        return 'Banker'
    if s == 'player' or s == 'p':
        return 'Player'
    if s == 'tie' or s == 't':
        return 'Tie'
    return None

# Incremental roadmap updates
def update_big_road(history, state):
    max_rows = 6
    max_cols = 20  # Reduced from 30
    if 'big_road_grid' not in state:
        state.big_road_grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
        state.big_road_cols = 0
        state.big_road_row = 0
        state.big_road_last_outcome = None
        state.big_road_last_length = 0

    grid, col, row, last_outcome = state.big_road_grid, state.big_road_cols, state.big_road_row, state.big_road_last_outcome
    start_idx = state.big_road_last_length
    if start_idx > len(history):  # Handle undo
        grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
        col, row, last_outcome = 0, 0, None
        start_idx = 0

    for i in range(start_idx, len(history)):
        result = history[i]
        mapped = 'P' if result == 'Player' else 'B' if result == 'Banker' else 'T'
        if mapped == 'T':
            if col < max_cols and row < max_rows and grid[row][col] == '':
                grid[row][col] = 'T'
            continue
        if col >= max_cols:
            break
        if last_outcome is None or (mapped == last_outcome and row < max_rows - 1):
            grid[row][col] = mapped
            row += 1
        else:
            col += 1
            row = 0
            if col < max_cols:
                grid[row][col] = mapped
                row += 1
        last_outcome = mapped if mapped != 'T' else last_outcome

    state.big_road_grid, state.big_road_cols, state.big_road_row, state.big_road_last_outcome = grid, col + 1, row, last_outcome
    state.big_road_last_length = len(history)
    return grid, col + 1

def update_derived_road(big_road_grid, num_cols, state, road_type):
    max_rows = 6
    max_cols = 20
    key = f'{road_type}_grid'
    cols_key = f'{road_type}_cols'
    last_length_key = f'{road_type}_last_length'

    if key not in state:
        state[key] = [['' for _ in range(max_cols)] for _ in range(max_rows)]
        state[cols_key] = 0
        state[last_length_key] = 0

    grid, col = state[key], state[cols_key]
    start_col = state[last_length_key]
    if start_col > num_cols:  # Reset on undo
        grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
        col = 0
        start_col = 0

    offset = {'big_eye_boy': 3, 'small_road': 4, 'cockroach_pig': 5}[road_type]
    row = 0
    for c in range(max(start_col + offset, offset), num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        compare_col = [big_road_grid[r][c - offset] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        compare_non_empty = next((i for i, x in enumerate(compare_col) if x in ['P', 'B']), None)
        if last_non_empty is not None and compare_non_empty is not None:
            if last_col[last_non_empty] == compare_col[compare_non_empty]:
                grid[row][col] = 'R'
            else:
                grid[row][col] = 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            break

    state[key], state[cols_key], state[last_length_key] = grid, col + 1 if row > 0 else col, num_cols
    return grid, col + 1 if row > 0 else col

def detect_streak(recent, freq):
    if not recent:
        return None, 0
    last = recent[-1]
    count = 1
    for r in recent[-2::-1]:
        if r == last:
            count += 1
        else:
            break
    return last, count

def is_alternating_pattern(recent):
    if len(recent) < 4:
        return False
    for i in range(len(recent) - 1):
        if recent[i] == recent[i + 1]:
            return False
    return True

def recent_trend_analysis(recent, history_length, freq):
    window = 6 if history_length < 10 else 10 if history_length <= 20 else 15
    recent_window = recent[-window:] if len(recent) >= window else recent
    if not recent_window:
        return None, 0
    total = len(recent_window)
    banker_ratio = sum(1 for r in recent_window if r == 'Banker') / total
    player_ratio = sum(1 for r in recent_window if r == 'Player') / total
    if banker_ratio > player_ratio + 0.2:
        return 'Banker', banker_ratio * 60
    elif player_ratio > banker_ratio + 0.2:
        return 'Player', player_ratio * 60
    return None, 0

@st.cache_data
def compute_win_loss(history_tuple, ai_mode, strategy):
    history = list(history_tuple)
    outcomes = []
    for i in range(len(history)):
        if i == 0:
            outcomes.append('N')
            continue
        prior_history = history[:i]
        bet, _, _, _, _ = advanced_bet_selection(prior_history, ai_mode, strategy)
        actual_result = history[i]
        if bet in ('Pass', None) or actual_result == 'Tie':
            outcomes.append('N')
        elif bet == actual_result:
            outcomes.append('W')
        else:
            outcomes.append('L')
    return outcomes

def compute_recent_accuracy(history, ai_mode, strategy):
    outcomes = compute_win_loss(tuple(history), ai_mode, strategy)[-10:]
    if not outcomes:
        return 1.0
    wins = outcomes.count('W')
    total = sum(1 for o in outcomes if o in ['W', 'L'])
    return wins / total if total > 0 else 1.0

@st.cache_data
def advanced_bet_selection(results, mode, money_management_strategy):
    start_time = time.time()
    max_recent_count = 30
    recent = results[-max_recent_count:] if len(results) > max_recent_count else results
    if not recent:
        return 'Pass', 0, "No results yet. Let’s wait for the shoe to develop!", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    history_length = len(results)

    # Precompute frequency
    freq = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in recent:
        if r in freq:
            freq[r] += 1
    total = len(recent)

    # Flat betting bias
    is_flat_betting = money_management_strategy == "Flat Betting"
    if is_flat_betting and len(recent) >= 3:
        recent_outcomes = recent[-3:]
        last_outcome = recent_outcomes[-1]
        if all(r == last_outcome for r in recent_outcomes):
            scores[last_outcome] += 80
            reason_parts.append("Strong recent momentum in last 3 hands.")
            pattern_insights.append(f"Recent 3-hand momentum: {last_outcome}")
            emotional_tone = "Confident"
        elif len(recent) >= 5 and is_alternating_pattern(recent[-5:]):
            alternate_bet = 'Player' if last_outcome == 'Banker' else 'Banker'
            scores[alternate_bet] += 90
            reason_parts.append("Strong 5-hand alternating pattern detected.")
            pattern_insights.append("Recent 5-hand chop")
            emotional_tone = "Excited"

    # Streak detection
    streak_value, streak_length = detect_streak(recent, freq)
    if streak_length >= 3 and streak_value != "Tie":
        streak_score = min(80 + (streak_length - 3) * 12, 95)
        scores[streak_value] += streak_score
        reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected.")
        pattern_insights.append(f"Streak: {streak_length} {streak_value}")
        emotional_tone = "Optimistic" if streak_length < 5 else "Confident"
        if streak_length >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
            scores[contrarian_bet] += 35
            reason_parts.append(f"Long streak ({streak_length}); considering a break.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    # Alternating pattern
    if len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        chop_score = 85 if len(recent) >= 5 and is_alternating_pattern(recent[-5:]) else 75
        scores[alternate_bet] += chop_score
        reason_parts.append(f"Alternating pattern (chop) in last {4 if chop_score == 75 else 5} hands.")
        pattern_insights.append(f"Chop pattern: Alternating P/B ({4 if chop_score == 75 else 5} hands)")
        emotional_tone = "Excited"

    # Trend analysis
    trend_bet, trend_score = recent_trend_analysis(recent, history_length, freq)
    if trend_bet:
        scores[trend_bet] += trend_score
        reason_parts.append(f"Recent trend favors {trend_bet} in last {6 if history_length < 10 else 10 if history_length <= 20 else 15} hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        emotional_tone = "Hopeful"

    # Roadmaps
    big_road_grid, num_cols = update_big_road(recent, st.session_state)
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            scores[bet_side] += 65
            reason_parts.append(f"Big Road column of {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")

    roadmap_weight = 0.8 if is_flat_betting else 1.0
    if num_cols >= 3:
        big_eye_grid, big_eye_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'big_eye_boy')
        if big_eye_cols > 0:
            last_col = [big_eye_grid[row][big_eye_cols - 1] for row in range(6)]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
                if last_signal == 'R':
                    scores[last_side] += 45 * roadmap_weight
                    reason_parts.append("Big Eye Boy suggests pattern repetition.")
                    pattern_insights.append("Big Eye Boy: Repeat pattern")
                else:
                    scores[opposite_side] += 40 * roadmap_weight
                    reason_parts.append("Big Eye Boy indicates a pattern break.")
                    pattern_insights.append("Big Eye Boy: Break pattern")

    if num_cols >= 4:
        small_road_grid, small_road_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'small_road')
        if small_road_cols > 0:
            last_col = [small_road_grid[row][small_road_cols - 1] for row in range(6)]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
                if last_signal == 'R':
                    scores[last_side] += 40 * roadmap_weight
                    reason_parts.append("Small Road suggests pattern repetition.")
                    pattern_insights.append("Small Road: Repeat pattern")
                else:
                    scores[opposite_side] += 35 * roadmap_weight
                    reason_parts.append("Small Road indicates a pattern break.")
                    pattern_insights.append("Small Road: Break pattern")

    if num_cols >= 5:
        cockroach_grid, cockroach_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'cockroach_pig')
        if cockroach_cols > 0:
            last_col = [cockroach_grid[row][cockroach_cols - 1] for row in range(6)]
            last_signal = next((x for x in last_col if x in ['R', 'B']), None)
            if last_signal:
                last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
                opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
                if last_signal == 'R':
                    scores[last_side] += 35 * roadmap_weight
                    reason_parts.append("Cockroach Pig suggests pattern repetition.")
                    pattern_insights.append("Cockroach Pig: Repeat pattern")
                else:
                    scores[opposite_side] += 30 * roadmap_weight
                    reason_parts.append("Cockroach Pig indicates a pattern break.")
                    pattern_insights.append("Cockroach Pig: Break pattern")

    # Frequency with decay
    decay_factor = 0.9 ** (total // 10)
    scores['Banker'] += (freq['Banker'] / total * 0.9) * 40 * decay_factor
    scores['Player'] += (freq['Player'] / total * 1.0) * 40 * decay_factor
    scores['Tie'] += (freq['Tie'] / total * 0.5) * 40 * decay_factor
    reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
    pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    # Select bet
    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores['Banker'], scores['Player'], scores['Tie'])), 95)

    # Accuracy adjustment
    accuracy = compute_recent_accuracy(results, mode, money_management_strategy)
    if accuracy < 0.5:
        confidence = max(confidence - 10, 0)
        reason_parts.append("Recent prediction accuracy low; confidence reduced.")
        emotional_tone = "Cautious"

    # Confidence thresholds
    confidence_threshold = 55 if is_flat_betting and mode == 'Conservative' else 35 if is_flat_betting and mode == 'Aggressive' else 60 if mode == 'Conservative' else 40
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif confidence < 70 and mode == 'Conservative':
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    # Tie bet restriction
    tie_threshold = 85 if is_flat_betting else 80
    if bet_choice == 'Tie' and confidence < tie_threshold:
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice]), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    # Conflicting patterns
    if len(pattern_insights) > 2 and max(scores.values()) - min(scores.values()) < 20:
        confidence = max(confidence - 15, 40)
        reason_parts.append("Multiple conflicting patterns; lowering confidence.")
        emotional_tone = "Skeptical"

    reason = " ".join(reason_parts)
    print(f"Bet selection time: {time.time() - start_time:.3f}s")  # Debug timing
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

def money_management(bankroll, base_bet, strategy, confidence=None, history=None):
    start_time = time.time()
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if strategy == "T3":
        if not history or len(history) < 3:
            calculated_bet = base_bet
        else:
            recent = history[-3:]
            last_result = recent[-1]
            streak = all(r == last_result for r in recent)
            if streak and last_result in ['Player', 'Banker']:
                if len(history) >= 4 and history[-4] == last_result:
                    calculated_bet = base_bet * 4
                else:
                    calculated_bet = base_bet * 2
            else:
                calculated_bet = base_bet
    elif strategy == "Fixed 5% of Bankroll":
        calculated_bet = bankroll * 0.05
    elif strategy == "Flat Betting":
        calculated_bet = base_bet
    elif strategy == "Confidence-Based":
        if confidence is None:
            confidence = 50
        confidence_factor = confidence / 100.0
        bet_percentage = 0.02 + (confidence_factor * 0.03)
        calculated_bet = bankroll * bet_percentage
    else:
        calculated_bet = base_bet

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    print(f"Money management time: {time.time() - start_time:.3f}s")  # Debug timing
    return round(bet_size, 2)

@st.cache_data
def calculate_bankroll(history_tuple, base_bet, strategy):
    start_time = time.time()
    history = list(history_tuple)
    bankroll = st.session_state.initial_bankroll
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode, strategy) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_size = money_management(current_bankroll, base_bet, strategy, confidence, current_rounds)
        bet_sizes.append(bet_size)
        if actual_result == bet:
            if bet == 'Banker':
                win_amount = bet_size * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += bet_size
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
        bankroll_progress.append(current_bankroll)
    print(f"Bankroll calc time: {time.time() - start_time:.3f}s")  # Debug timing
    return bankroll_progress, bet_sizes

def render_grid(grid, num_cols, grid_type, state, display_cols=12):
    key = f'{grid_type}_html'
    if key not in state or state.get(f'{grid_type}_last_length', 0) != num_cols:
        html_rows = []
        for row in range(6):
            row_display = []
            for col in range(min(num_cols, display_cols)):
                outcome = grid[row][col]
                if grid_type == 'bead_plate':
                    if outcome in ['P', 'B', 'T']:
                        color = '#3182ce' if outcome == 'P' else '#e53e3e' if outcome == 'B' else '#38a169'
                        row_display.append(f'<div style="width: 22px; height: 22px; background-color: {color}; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                    else:
                        row_display.append('<div style="width: 22px; height: 22px; display: inline-block;"></div>')
                else:
                    if outcome == 'P':
                        row_display.append('<div style="width: 22px; height: 22px; background-color: #3182ce; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                    elif outcome == 'B':
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
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.session_state.money_management_strategy = "Fixed 5% of Bankroll"
        st.session_state.ai_mode = "Conservative"
        st.session_state.selected_patterns = ["Bead Plate", "Big Road"]

    # Game Settings
    with st.expander("Game Settings", expanded=False):
        col_init, col_base, col_strategy, col_mode = st.columns(4)
        with col_init:
            initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
        with col_base:
            base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
        with col_strategy:
            strategy_options = ["Fixed 5% of Bankroll", "Flat Betting", "Confidence-Based", "T3"]
            money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
        with col_mode:
            ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

        if (st.session_state.initial_bankroll != initial_bankroll or
            st.session_state.base_bet != base_bet or
            st.session_state.money_management_strategy != money_management_strategy or
            st.session_state.ai_mode != ai_mode):
            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode
            st.rerun()

        st.markdown(f"**Selected Money Management Strategy:** {money_management_strategy}")

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
        selected_patterns = st.multiselect(
            "Select Patterns to Display",
            pattern_options,
            default=st.session_state.selected_patterns,
            key="pattern_select"
        )
        if st.session_state.selected_patterns != selected_patterns:
            st.session_state.selected_patterns = selected_patterns
            st.rerun()

        if "Bead Plate" in selected_patterns:
            st.markdown("### Bead Plate")
            sequence = [r for r in st.session_state.history][-72:]  # Reduced from 84
            sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence]
            grid = [['' for _ in range(12)] for _ in range(6)]  # 6x12
            for i, result in enumerate(sequence):
                col = i // 6
                row = i % 6
                if col < 12:
                    grid[row][col] = result
            html_rows = render_grid(grid, len(sequence) // 6 + 1 if sequence else 0, 'bead_plate', st.session_state)
            for row in html_rows:
                st.markdown(row, unsafe_allow_html=True)
            if not st.session_state.history:
                st.write("_No results yet. Click the buttons above to add results._")

            # Win/Loss Tracker
            st.markdown("#### Win/Loss Tracker")
            st.markdown("<p style='font-size: 12px; color: #666666;'>Green (●): Win, Red (●): Loss, Gray (●): Pass/Tie</p>", unsafe_allow_html=True)
            outcomes = compute_win_loss(tuple(st.session_state.history), st.session_state.ai_mode, st.session_state.money_management_strategy)[-72:]
            if 'tracker_html' not in st.session_state or st.session_state.get('tracker_last_length', 0) != len(outcomes):
                tracker_row = []
                for outcome in outcomes:
                    if outcome == 'W':
                        tracker_row.append('<div style="width: 22px; height: 22px; background-color: #38a169; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                    elif outcome == 'L':
                        tracker_row.append('<div style="width: 22px; height: 22px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                    else:
                        tracker_row.append('<div style="width: 22px; height: 22px; background-color: #666666; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                st.session_state.tracker_html = ' '.join(tracker_row)
                st.session_state.tracker_last_length = len(outcomes)
            st.markdown(st.session_state.tracker_html, unsafe_allow_html=True)

        if "Big Road" in selected_patterns:
            st.markdown("### Big Road")
            big_road_grid, num_cols = update_big_road(st.session_state.history, st.session_state)
            if num_cols > 0:
                html_rows = render_grid(big_road_grid, num_cols, 'big_road', st.session_state)
                for row in html_rows:
                    st.markdown(row, unsafe_allow_html=True)
            else:
                st.write("_No Big Road data yet._")

        if "Big Eye Boy" in selected_patterns:
            st.markdown("### Big Eye Boy")
            st.markdown("<p style='font-size: 12px; color: #666666;'>Red (●): Repeat Pattern, Blue (●): Break Pattern</p>", unsafe_allow_html=True)
            big_road_grid, num_cols = update_big_road(st.session_state.history, st.session_state)
            big_eye_grid, big_eye_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'big_eye_boy')
            if big_eye_cols > 0:
                html_rows = render_grid(big_eye_grid, big_eye_cols, 'big_eye_boy', st.session_state)
                for row in html_rows:
                    st.markdown(row, unsafe_allow_html=True)
            else:
                st.write("_No Big Eye Boy data yet._")

        if "Cockroach Pig" in selected_patterns:
            st.markdown("### Cockroach Pig")
            st.markdown("<p style='font-size: 12px; color: #666666;'>Regular (red), Strong (blue). Represents strong or regular break in pattern.</p>", unsafe_allow_html=True)
            big_road_grid, num_cols = update_big_road(st.session_state.history, st.session_state)
            cockroach_grid, cockroach_cols = update_derived_road(big_road_grid, num_cols, st.session_state, 'cockroach_pig')
            if cockroach_cols > 0:
                html_rows = render_grid(cockroach_grid, cockroach_cols, 'cockroach_pig', st.session_state)
                for row in html_rows:
                    st.markdown(row, unsafe_allow_html=True)
            else:
                st.write("_Regular_")

    # Bet Prediction
    prediction_container = st.container()
    with prediction_container:
        st.markdown("### Prediction for Next Bet")
        bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode, st.session_state.money_management_strategy)
        if bet == 'Pass':
            st.warning("I’m not betting this time! The pattern is too unclear.")
        else:
            current_bankroll = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
            recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy, confidence, st.session_state.history)
            st.success(f"Predicted Bet: **{bet}**    Confidence: **{confidence}%**    Recommended Bet Size: **${recommended_bet_size:.2f}**    Emotion: **{emotional_tone}**")
        st.info(reason)
        if pattern_insights:
            st.markdown("#### Detected Patterns")
            for insight in pattern_insights:
                st.write(f"- {insight}")

    # Bankroll Progression
    with st.expander("Bankroll and Bet Size Progression", expanded=False):
        bankroll_progress, bet_sizes = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)
        if 'bankroll_html' not in st.session_state or st.session_state.get('bankroll_last_length', 0) != len(bankroll_progress):
            html_output = []
            total_hands = len(bankroll_progress)
            for i, (val, bet_size) in enumerate(zip(reversed(bankroll_progress), reversed(bet_sizes))):
                hand_number = total_hands - i
                bet_display = f"Bet: ${bet_size:.2f}" if bet_size > 0 else "Bet: None (No prediction, Tie, or Pass)"
                html_output.append(f"{hand_number}: Bankroll ${val:.2f}, {bet_display}")
            st.session_state.bankroll_html = html_output
            st.session_state.bankroll_last_length = len(bankroll_progress)
            st.session_state.current_bankroll = bankroll_progress[-1] if bankroll_progress else st.session_state.initial_bankroll

        st.markdown("### Progression (Newest to Oldest)")
        for line in st.session_state.bankroll_html:
            st.write(f"Hand {line}")
        st.markdown(f"**Current Bankroll:** ${st.session_state.current_bankroll:.2f}")

    # Reset Game
    with st.expander("Reset Game", expanded=False):
        if st.button("Reset History and Bankroll"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    print(f"Total render time: {time.time() - start_time:.3f}s")  # Debug timing

if __name__ == "__main__":
    main()
