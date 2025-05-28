
import streamlit as st
import logging
import plotly.graph_objects as go
import json

# Set up basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Normalize input
def normalize(s):
    s = s.strip().lower()
    if s == 'banker' or s == 'b':
        return 'Banker'
    if s == 'player' or s == 'p':
        return 'Player'
    if s == 'tie' or s == 't':
        return 'Tie'
    return None

def detect_streak(s):
    if not s:
        return None, 0
    last = s[-1]
    count = 1
    for i in range(len(s) - 2, -1, -1):
        if s[i] == last:
            count += 1
        else:
            break
    return last, count

def is_alternating(s, min_length=4):
    if len(s) < min_length:
        return False
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

def is_zigzag(s):
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i + 2] and s[i] != s[i + 1]:
            return True
    return False

def recent_trend(s, window=10):
    recent = s[-window:] if len(s) >= window else s
    if not recent:
        return None, 0
    freq = frequency_count(recent)
    total = len(recent)
    if total == 0:
        return None, 0
    banker_ratio = freq['Banker'] / total
    player_ratio = freq['Player'] / total
    if banker_ratio > player_ratio + 0.2:
        return 'Banker', min(banker_ratio * 50, 80)
    elif player_ratio > banker_ratio + 0.2:
        return 'Player', min(player_ratio * 50, 80)
    return None, 0

def frequency_count(s):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in s:
        if r in count:
            count[r] += 1
    return count

def build_big_road(s):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0
    last_outcome = None

    for result in s:
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
    return grid, col + 1

def build_big_eye_boy(big_road_grid, num_cols):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0

    for c in range(3, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        third_last = [big_road_grid[r][c - 3] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        third_non_empty = next((i for i, x in enumerate(third_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and third_non_empty is not None:
            if last_col[last_non_empty] == third_last[third_non_empty]:
                grid[row][col] = 'R'  # Repeat (red)
            else:
                grid[row][col] = 'B'  # Break (blue)
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

def build_cockroach_pig(big_road_grid, num_cols):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0

    for c in range(4, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        fourth_last = [big_road_grid[r][c - 4] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        fourth_non_empty = next((i for i, x in enumerate(fourth_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and fourth_non_empty is not None:
            if last_col[last_non_empty] == fourth_last[fourth_non_empty]:
                grid[row][col] = 'R'  # Repeat (red)
            else:
                grid[row][col] = 'B'  # Break (blue)
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

def advanced_bet_selection(s, mode='Conservative'):
    max_recent_count = 30
    recent = s[-max_recent_count:]
    if not recent:
        return 'Pass', 0, "No results yet. Let’s wait for the shoe to develop!", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    confidence = 0
    pattern_count = 0

    streak_value, streak_length = detect_streak(recent)
    if streak_length >= 3 and streak_value != "Tie":
        streak_score = min(20 + (streak_length - 3) * 5, 40)
        scores[streak_value] += streak_score
        reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected.")
        pattern_insights.append(f"Streak: {streak_length} {streak_value}")
        pattern_count += 1
        emotional_tone = "Optimistic" if streak_length < 5 else "Confident"
        if streak_length >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
            scores[contrarian_bet] += 15
            reason_parts.append(f"Long streak ({streak_length}); considering a break.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    if len(recent) >= 4 and is_alternating(recent[-4:], min_length=4):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += 30
        reason_parts.append("Alternating pattern (chop) in last 4 hands.")
        pattern_insights.append("Chop pattern: Alternating P/B")
        pattern_count += 1
        emotional_tone = "Excited"

    if is_zigzag(recent[-6:]):
        last = recent[-1]
        zigzag_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[zigzag_bet] += 25
        reason_parts.append("Zigzag pattern (P-B-P or B-P-B) detected in last 6 hands.")
        pattern_insights.append("Zigzag pattern: P-B-P/B-P-B")
        pattern_count += 1
        emotional_tone = "Curious"

    trend_bet, trend_score = recent_trend(recent)
    if trend_bet:
        scores[trend_bet] += min(trend_score, 30)
        reason_parts.append(f"Recent trend favors {trend_bet} in last 10 hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        pattern_count += 1
        emotional_tone = "Hopeful"

    big_road_grid, num_cols = build_big_road(recent)
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            col_score = 20 if col_length == 3 else 30 if col_length == 4 else 40
            scores[bet_side] += col_score
            reason_parts.append(f"Big Road column of {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")
            pattern_count += 1

    big_eye_grid, big_eye_cols = build_big_eye_boy(big_road_grid, num_cols)
    if big_eye_cols > 0:
        last_col = [big_eye_grid[row][big_eye_cols - 1] for row in range(6)]
        last_signal = next((x for x in last_col if x in ['R', 'B']), None)
        if last_signal:
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            if last_signal == 'R':
                scores[last_side] += 15
                reason_parts.append("Big Eye Boy suggests pattern repetition.")
                pattern_insights.append("Big Eye Boy: Repeat pattern")
            else:
                scores[opposite_side] += 10
                reason_parts.append("Big Eye Boy indicates a pattern break.")
                pattern_insights.append("Big Eye Boy: Break pattern")
            pattern_count += 1

    cockroach_grid, cockroach_cols = build_cockroach_pig(big_road_grid, num_cols)
    if cockroach_cols > 0:
        last_col = [cockroach_grid[row][cockroach_cols - 1] for row in range(6)]
        last_signal = next((x for x in last_col if x in ['R', 'B']), None)
        if last_signal:
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            if last_signal == 'R':
                scores[last_side] += 10
                reason_parts.append("Cockroach Pig suggests pattern repetition.")
                pattern_insights.append("Cockroach Pig: Repeat pattern")
            else:
                scores[opposite_side] += 8
                reason_parts.append("Cockroach Pig indicates a pattern break.")
                pattern_insights.append("Cockroach Pig: Break pattern")
            pattern_count += 1

    recent_wins = recent[-5:] if len(recent) >= 5 else recent
    freq = frequency_count(recent_wins)
    total = len(recent_wins)
    if total > 0:
        scores['Banker'] += (freq['Banker'] / total) * 20
        scores['Player'] += (freq['Player'] / total) * 20
        reason_parts.append(f"Recent momentum: Banker {freq['Banker']}, Player {freq['Player']} in last {total} hands.")
        pattern_insights.append(f"Momentum: B:{freq['Banker']}, P:{freq['Player']}")

    freq = frequency_count(recent)
    total = len(recent)
    if total > 0:
        scores['Banker'] += (freq['Banker'] / total * 0.9) * 20
        scores['Player'] += (freq['Player'] / total * 1.0) * 20
        scores['Tie'] += (freq['Tie'] / total * 0.5) * 20 if freq['Tie'] / total > 0.3 else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    if pattern_count >= 3:
        max_score = max(scores['Banker'], scores['Player'])
        if max_score > 0:
            coherence_bonus = 10 if pattern_count == 3 else 15
            max_bet = 'Banker' if scores['Banker'] > scores['Player'] else 'Player'
            scores[max_bet] += coherence_bonus
            reason_parts.append(f"Multiple patterns align on {max_bet} (+{coherence_bonus} bonus).")
            pattern_insights.append(f"Coherence: {pattern_count} patterns align")

    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores.values(), default=0) * 1.5), 95)

    confidence_threshold = 60 if mode == 'Conservative' else 40
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif mode == 'Conservative' and confidence < 70:
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    if bet_choice == 'Tie' and confidence < 85:
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice] * 1.5), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    if pattern_count > 3 and max(scores.values(), default=0) - min([scores['Banker'], scores['Player']], default=0) < 15:
        confidence = max(confidence - 10, 40)
        reason_parts.append("Multiple conflicting patterns; lowering confidence.")
        emotional_tone = "Skeptical"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

def money_management(bankroll, base_bet, strategy, confidence=None, history=None, bet_outcome=None):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if strategy == "T3":
        if bet_outcome == 'win':
            if not st.session_state.t3_results:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
        elif bet_outcome == 'loss':
            st.session_state.t3_results.append('L')

        if len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []

        calculated_bet = base_bet * st.session_state.t3_level
    elif strategy == "Flat Betting":
        calculated_bet = base_bet
    else:
        calculated_bet = base_bet  # Fallback to flat betting

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

def calculate_bankroll(history, base_bet, strategy):
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_size = money_management(current_bankroll, base_bet, strategy, confidence, current_rounds, bet_outcome=None)
        bet_sizes.append(bet_size)
        if actual_result == bet:
            if bet == 'Banker':
                win_amount = bet_size * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += bet_size
            if strategy == "T3":
                money_management(current_bankroll, base_bet, strategy, confidence, current_rounds, bet_outcome='win')
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                money_management(current_bankroll, base_bet, strategy, confidence, current_rounds, bet_outcome='loss')
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def calculate_win_loss_tracker(history, base_bet, strategy, ai_mode):
    tracker = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if actual_result == 'Tie':
            tracker.append('T')  # Tie
        elif bet in (None, 'Pass'):
            tracker.append('S')  # Skip
        elif actual_result == bet:
            tracker.append('W')  # Win
        else:
            tracker.append('L')  # Loss
    return tracker

def main():
    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
            st.session_state.initial_bankroll = 1000.0
            st.session_state.base_bet = 10.0
            st.session_state.money_management_strategy = "Flat Betting"
            st.session_state.ai_mode = "Conservative"
            st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.screen_width = 1024  # Default width

        # JavaScript to detect screen width
        screen_width_js = """
        <script>
        function updateScreenWidth() {
            const width = window.innerWidth;
            localStorage.setItem('screen_width', width);
            document.getElementById('screen-width-input').value = width;
        }
        window.onload = updateScreenWidth;
        window.onresize = updateScreenWidth;
        </script>
        <input type="hidden" id="screen-width-input">
        """
        st.markdown(screen_width_js, unsafe_allow_html=True)

        # Retrieve screen width from localStorage
        screen_width = st.session_state.screen_width
        if 'screen_width' in st.session_state:
            try:
                # Simulate getting width from JavaScript (in practice, use a component like streamlit-js)
                width = st.text_input("Screen Width", key="screen_width_input", value=str(screen_width), disabled=True)
                screen_width = int(width) if width.isdigit() else 1024
            except:
                screen_width = 1024
        st.session_state.screen_width = screen_width

        # Responsive CSS
        st.markdown("""
            <style>
            .pattern-scroll {
                overflow-x: auto;
                white-space: nowrap;
                max-width: 100%;
                padding: 10px;
                border: 1px solid #e1e1e1;
                background-color: #f9f9f9;
            }
            .pattern-scroll::-webkit-scrollbar {
                height: 8px;
            }
            .pattern-scroll::-webkit-scrollbar-thumb {
                background-color: #888;
                border-radius: 4px;
            }
            .stButton > button {
                width: 100%;
                padding: 8px;
                margin: 5px 0;
            }
            .stNumberInput, .stSelectbox {
                width: 100% !important;
            }
            .stExpander {
                margin-bottom: 10px;
            }
            h1 {
                font-size: 2.5rem;
                text-align: center;
            }
            h3 {
                font-size: 1.5rem;
            }
            p, div, span {
                font-size: 1rem;
            }
            /* Laptop (>768px) */
            @media (max-width: 1024px) {
                h1 {
                    font-size: 2rem;
                }
                h3 {
                    font-size: 1.3rem;
                }
                p, div, span {
                    font-size: 0.95rem;
                }
                .pattern-circle {
                    width: 18px !important;
                    height: 18px !important;
                }
            }
            /* Mobile (<768px) */
            @media (max-width: 768px) {
                h1 {
                    font-size: 1.8rem;
                }
                h3 {
                    font-size: 1.2rem;
                }
                p, div, span {
                    font-size: 0.9rem;
                }
                .pattern-circle {
                    width: 16px !important;
                    height: 16px !important;
                }
                .stButton > button {
                    font-size: 0.9rem;
                    padding: 6px;
                }
                .stNumberInput input, .stSelectbox div {
                    font-size: 0.9rem;
                }
            }
            </style>
            <script>
            function autoScrollPatterns() {
                const containers = [
                    'bead-bin-scroll',
                    'big-road-scroll',
                    'big-eye-scroll',
                    'cockroach-scroll',
                    'win-loss-scroll'
                ];
                containers.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.scrollLeft = element.scrollWidth;
                    }
                });
            }
            window.onload = autoScrollPatterns;
            </script>
        """, unsafe_allow_html=True)

        # Game Settings
        with st.expander("Game Settings", expanded=False):
            # Responsive layout
            if screen_width > 768:
                cols = st.columns(4)
            else:
                cols = [st.container() for _ in range(4)]
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3"]
                money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
                st.markdown("*Flat Betting: Fixed bet size. T3: Adjusts bet level based on the last three bet outcomes (increase if more losses, decrease if more wins).*")
            with cols[3]:
                ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode

            st.markdown(f"**Selected Strategy: {money_management_strategy}**")

        # Game Input Buttons
        with st.expander("Input Game Results", expanded=True):
            if screen_width > 768:
                cols = st.columns(4)
            else:
                cols = [st.container() for _ in range(4)]
            with cols[0]:
                if st.button("Banker"):
                    st.session_state.history.append("Banker")
                    st.rerun()
            with cols[1]:
                if st.button("Player"):
                    st.session_state.history.append("Player")
                    st.rerun()
            with cols[2]:
                if st.button("Tie"):
                    st.session_state.history.append("Tie")
                    st.rerun()
            with cols[3]:
                if st.button("Undo", disabled=len(st.session_state.history) == 0):
                    if st.session_state.history:
                        st.session_state.history.pop()
                        if st.session_state.money_management_strategy == "T3":
                            st.session_state.t3_results = []
                            st.session_state.t3_level = 1
                        st.rerun()
                    else:
                        st.warning("Nothing to undo!")

        # Shoe Patterns
        with st.expander("Shoe Patterns", expanded=False):
            pattern_options = ["Bead Bin", "Big Road", "Big Eye.", "Cockroach", "Win/Loss"]
            selected_patterns = st.multiselect(
                "Select Patterns to Display",
                pattern_options,
                default=["Bead Bin", "Win/Loss"],
                key="pattern_select"
            )
            st.session_state.selected_patterns = selected_patterns

            # Pattern settings based on screen width
            max_display_cols = 14 if screen_width > 768 else 10
            circle_size = 22 if screen_width > 768 else 16

            if "Bead Bin" in st.session_state.selected_patterns:
                st.markdown("### Bead Bin")
                sequence = [r for r in st.session_state.history][-84:]
                sequence = ['P' if result == 'Player' else 'B' if result == 'Banker' else 'T' for result in sequence]
                grid = [['' for _ in range(max_display_cols)] for _ in range(6)]
                for i, result in enumerate(sequence):
                    if result in ['P', 'B', 'T']:
                        col = i // 6
                        row = i % 6
                        if col < max_display_cols:
                            color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                            grid[row][col] = f'<div class="pattern-circle" style="width: {circle_size}px; height: {circle_size}px; background-color: {color}; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>'
                st.markdown('<div id="bead-bin-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                for row in grid:
                    st.markdown(' '.join(row), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if not st.session_state.history:
                    st.markdown("No results yet. Enter results below.")

            if "Big Road" in st.session_state.selected_patterns:
                st.markdown("### Big Road")
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                if num_cols > 0:
                    display_cols = min(num_cols, max_display_cols)
                    st.markdown('<div id="big-road-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_road_grid[row][col]
                            if outcome == 'P':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size}px; height: {circle_size}px; background-color: #3182ce; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size}px; height: {circle_size}px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff; display: inline-block;"></div>')
                            elif outcome == 'T':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size}px; height: {circle_size}px; border: 2px solid #38a169; border-radius: 50%; display: inline-block;"></div>')
                            else:
                                row_display.append(f'<div style="width: {circle_size}px; height: {circle_size}px; display: inline-block;"></div>')
                        st.markdown(' '.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("No Big Road data.")

            if "Big Eye." in st.session_state.selected_patterns:
                st.markdown("### Big Eye Boy")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Red (●): Repeat Pattern, Blue (●): Break Pattern</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                big_eye_grid, big_eye_cols = build_big_eye_boy(big_road_grid, num_cols)
                if big_eye_cols > 0:
                    display_cols = min(big_eye_cols, max_display_cols)
                    st.markdown('<div id="big-eye-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_eye_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size - 2}px; height: {circle_size - 2}px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #000000; display: inline-block;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size - 2}px; height: {circle_size - 2}px; background-color: #3182ce; border-radius: 50%; border: 1px solid #000000; display: inline-block;"></div>')
                            else:
                                row_display.append(f'<div style="width: {circle_size - 2}px; height: {circle_size - 2}px; display: inline-block;"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("No recent Big Eye data.")

            if "Cockroach" in st.session_state.selected_patterns:
                st.markdown("### Cockroach")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Red (●): Repeat Pattern, Blue (●): Break Pattern</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                cockroach_grid, cockroach_cols = build_cockroach_pig(big_road_grid, num_cols)
                if cockroach_cols > 0:
                    display_cols = min(cockroach_cols, max_display_cols)
                    st.markdown('<div id="cockroach-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = cockroach_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size - 2}px; height: {circle_size - 2}px; background-color: #e53e3e; border-radius: 50%; border: 1px solid #000000; display: inline-block;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="width: {circle_size - 2}px; height: {circle_size - 2}px; background-color: #3182ce; border-radius: 50%; border: 1px solid #000000; display: inline-block;"></div>')
                            else:
                                row_display.append(f'<div style="width: {circle_size - 2}px; height: {circle_size - 2}px; display: inline-block;"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("No recent Cockroach data.")

            if "Win/Loss" in st.session_state.selected_patterns:
                st.markdown("### Win/Loss")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Green (●): Win, Red (●): Loss, Gray (●): Skip or Tie</p>", unsafe_allow_html=True)
                tracker = calculate_win_loss_tracker(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-max_display_cols:]
                row_display = []
                for result in tracker:
                    if result in ['W', 'L', 'S', 'T']:
                        color = '#38a169' if result == 'W' else '#e53e3e' if result == 'L' else '#A0AEC0'
                        row_display.append(f'<div class="pattern-circle" style="width: {circle_size - 2}px; height: {circle_size - 2}px; background-color: {color}; border-radius: 50%; border: 1px solid #000000; display: inline-block;"></div>')
                    else:
                        row_display.append(f'<div style="width: {circle_size - 2}px; height: {circle_size - 2}px; display: inline-block;"></div>')
                st.markdown('<div id="win-loss-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(' '.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if not st.session_state.history:
                    st.markdown("No results yet. Enter results below.")

        # Bet Prediction
        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
            st.markdown("### Prediction")
            if bet == 'Pass':
                st.warning("No Bet")
            else:
                current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy, confidence, st.session_state.history)
                st.success(f"Bet: {bet} | Confidence: {confidence}% | Bet Size: ${recommended_bet_size:.2f} | Mood: {emotional_tone}")
            st.info(f"Reasoning: {reason}")
            if pattern_insights:
                st.markdown("### Insights")
                for insight in pattern_insights:
                    st.markdown(f"- {insight}")

        # Bankroll Progression
        with st.expander("Bankroll Progress", expanded=True):
            bankroll_progress, bet_history = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)
            if bankroll_progress:
                st.markdown("### Bankroll Progress")
                total_hands = len(bankroll_progress)
                for i, (val, bet_size) in enumerate(zip(reversed(bankroll_progress), reversed(bet_history))):
                    hand_number = total_hands - i
                    bet_display = f"Bet ${bet_size:.2f}" if bet_size > 0 else "No Bet"
                    st.markdown(f"Hand {hand_number}: ${val:.2f} | {bet_display}")
                st.markdown(f"**Current Bankroll**: ${bankroll_progress[-1]:.2f}")

                st.markdown("### Bankroll Trend")
                labels = [f"Hand {i+1}" for i in range(len(bankroll_progress))]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=bankroll_progress,
                        mode='lines+markers',
                        name='Bankroll',
                        line=dict(color='#38a169', width=2),
                        marker=dict(size=6)
                    )
                )
                fig.update_layout(
                    title="Bankroll Over Time",
                    xaxis_title="Hand",
                    yaxis_title="Bankroll ($)",
                    xaxis=dict(tickangle=45),
                    yaxis=dict(rangemode="tozero"),
                    template="plotly_white",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"**Current Bankroll**: ${st.session_state.initial_bankroll:.2f}")
                st.markdown("No bankroll history yet. Enter results below.")

        # Reset
        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                final_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                st.session_state.history = []
                st.session_state.initial_bankroll = max(1.0, final_bankroll)
                st.session_state.base_bet = min(10.0, st.session_state.initial_bankroll)
                st.session_state.money_management_strategy = "Flat Betting"
                st.session_state.ai_mode = "Conservative"
                st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.rerun()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
