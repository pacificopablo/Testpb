import streamlit as st
import random
import math
import uuid

def main():
    # Initialize session state
    if 'baccarat' not in st.session_state:
        st.session_state.baccarat = {
            'pair_types': [],
            'next_prediction': "N/A",
            'base_amount': 10.0,
            'bet_amount': 10.0,
            'result_tracker': 0.0,
            'profit_lock': 0.0,
            'previous_result': None,
            'state_history': [],
            'current_dominance': "N/A",
            'streak_type': None,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'stats': {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}
        }
    
    baccarat = st.session_state.baccarat

    # Styling
    st.markdown("""
        <style>
        .main {background-color: #2C2F33; color: white; font-family: Helvetica;}
        .stButton>button {
            font-size: 14px;
            font-weight: bold;
            padding: 10px;
            background-color: #7289DA;
            color: white;
            border: none;
            border-radius: 5px;
            width: 120px;
        }
        .stButton>button:hover {
            background-color: #99AAB5;
        }
        .stTextInput>div>input {
            background-color: #23272A;
            color: white;
            font-size: 14px;
            border-radius: 5px;
        }
        .stTextArea textarea {
            background-color: #23272A;
            color: white;
            font-size: 12px;
            border-radius: 5px;
        }
        .title {font-size: 24px; font-weight: bold; text-align: center; padding: 20px;}
        .label {font-size: 14px; color: white;}
        .info {font-size: 16px; font-weight: bold; padding: 10px;}
        .player-bet {color: #1E90FF; font-weight: bold;}
        .banker-bet {color: #FF4500; font-weight: bold;}
        .na-bet {color: white; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title">Enhanced Dominant Pairs Baccarat Predictor</div>', unsafe_allow_html=True)

    # Base amount input
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        base_amount = st.text_input("Base Amount ($1-$100)", value=str(baccarat['base_amount']), key=str(uuid.uuid4()))
    with col2:
        if st.button("Set Amount"):
            try:
                amount = float(base_amount)
                if 1 <= amount <= 100:
                    baccarat['base_amount'] = amount
                    baccarat['bet_amount'] = amount
                    update_display(baccarat)
                else:
                    st.write("Base amount must be between $1 and $100.")
            except ValueError:
                st.write("Please enter a valid number.")

    # Info display
    st.markdown(f'<div class="info">Bet Amount: {"No Bet" if baccarat["bet_amount"] == 0 else f"${baccarat['bet_amount']:.2f}"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info">Bankroll: ${baccarat["result_tracker"]:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info">Profit Lock: ${baccarat["profit_lock"]:.2f}</div>', unsafe_allow_html=True)
    if baccarat['next_prediction'] == "Player":
        st.markdown(f'<div class="info">Bet: <span class="player-bet">{baccarat["next_prediction"]}</span></div>', unsafe_allow_html=True)
    elif baccarat['next_prediction'] == "Banker":
        st.markdown(f'<div class="info">Bet: <span class="banker-bet">{baccarat["next_prediction"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info">Bet: <span class="na-bet">{baccarat["next_prediction"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info">Streak: {baccarat["streak_type"] if baccarat["streak_type"] else "None"}</div>', unsafe_allow_html=True)

    # Stats display
    total_games = baccarat['stats']['wins'] + baccarat['stats']['losses']
    win_rate = (baccarat['stats']['wins'] / total_games * 100) if total_games > 0 else 0
    avg_streak = sum(baccarat['stats']['streaks']) / len(baccarat['stats']['streaks']) if baccarat['stats']['streaks'] else 0
    st.markdown(f'<div class="label">Win Rate: {win_rate:.1f}% | Avg Streak: {avg_streak:.1f} | Patterns: Odd: {baccarat["stats"]["odd_pairs"]}, Even: {baccarat["stats"]["even_pairs"]}</div>', unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player"):
            record_result(baccarat, 'P')
    with col2:
        if st.button("Banker"):
            record_result(baccarat, 'B')
    with col3:
        if st.button("Tie"):
            record_result(baccarat, 'T')
    with col4:
        if st.button("Undo"):
            undo(baccarat)

    # Deal history (newest at top)
    st.markdown('<div class="label">Deal History:</div>', unsafe_allow_html=True)
    history_text = ""
    for i, pair in enumerate(reversed(baccarat['pair_types'][-100:]), 1):
        pair_type = "Even" if pair[0] == pair[1] else "Odd"
        history_text += f"{pair} ({pair_type})\n"
    st.text_area("", value=history_text, height=200, key="history_area")
    st.components.v1.html("""
        <script>
        const textarea = document.querySelector('textarea');
        if (textarea) {
            textarea.scrollTop = 0;
        }
        </script>
    """, height=0)

    # Session control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset Bet"):
            reset_betting(baccarat)
    with col2:
        if st.button("Reset Session"):
            reset_all(baccarat)
    with col3:
        pass  # Empty column to maintain layout

def update_display(baccarat):
    st.session_state.baccarat = baccarat
    st.rerun()

def reset_betting(baccarat):
    if baccarat['result_tracker'] <= -10 * baccarat['base_amount']:
        st.write("Stop-loss reached. Resetting to resume betting.")
    if baccarat['result_tracker'] >= 0:
        baccarat['result_tracker'] = 0.0
    baccarat['bet_amount'] = baccarat['base_amount']
    baccarat['consecutive_wins'] = 0
    baccarat['consecutive_losses'] = 0
    baccarat['streak_type'] = None

    if len(baccarat['pair_types']) >= 5:
        recent_pairs = [p for p in baccarat['pair_types'][-10:] if p[0] != 'T' and p[1] != 'T']
        if recent_pairs:
            odd_count = sum(1 for a, b in recent_pairs if a != b)
            even_count = sum(1 for a, b in recent_pairs if a == b)
            result = baccarat['previous_result']
            last_three = [p[1] for p in baccarat['pair_types'][-3:] if p[1] != 'T']
            if len(last_three) >= 3 and all(r == last_three[0] for r in last_three):
                baccarat['streak_type'] = last_three[0]
                baccarat['next_prediction'] = "Player" if baccarat['streak_type'] == 'P' else "Banker"
                baccarat['current_dominance'] = f"Streak ({baccarat['streak_type']})"
            elif abs(odd_count - even_count) < 2:
                baccarat['current_dominance'] = "N/A"
                baccarat['next_prediction'] = "Hold"
            elif odd_count > even_count:
                baccarat['current_dominance'] = "Odd"
                baccarat['next_prediction'] = "Player" if result == 'B' else "Banker"
            else:
                baccarat['current_dominance'] = "Even"
                baccarat['next_prediction'] = "Player" if result == 'P' else "Banker"
    else:
        baccarat['next_prediction'] = "N/A"
        baccarat['current_dominance'] = "N/A"
        baccarat['streak_type'] = None
        baccarat['bet_amount'] = baccarat['base_amount']

    update_display(baccarat)
    st.write("Betting reset.")

def reset_all(baccarat):
    baccarat['pair_types'] = []
    baccarat['result_tracker'] = 0.0
    baccarat['profit_lock'] = 0.0
    baccarat['bet_amount'] = baccarat['base_amount']
    baccarat['base_amount'] = 10.0
    baccarat['next_prediction'] = "N/A"
    baccarat['previous_result'] = None
    baccarat['state_history'] = []
    baccarat['current_dominance'] = "N/A"
    baccarat['consecutive_wins'] = 0
    baccarat['consecutive_losses'] = 0
    baccarat['streak_type'] = None
    baccarat['stats'] = {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}
    update_display(baccarat)
    st.write("All session data reset, profit lock reset.")

def record_result(baccarat, result):
    state = {
        'pair_types': baccarat['pair_types'].copy(),
        'previous_result': baccarat['previous_result'],
        'result_tracker': baccarat['result_tracker'],
        'profit_lock': baccarat['profit_lock'],
        'bet_amount': baccarat['bet_amount'],
        'current_dominance': baccarat['current_dominance'],
        'next_prediction': baccarat['next_prediction'],
        'consecutive_wins': baccarat['consecutive_wins'],
        'consecutive_losses': baccarat['consecutive_losses'],
        'streak_type': baccarat['streak_type'],
        'stats': baccarat['stats'].copy()
    }
    baccarat['state_history'].append(state)

    if result == 'T':
        baccarat['stats']['ties'] += 1
        update_display(baccarat)
        return

    if baccarat['previous_result'] is None:
        baccarat['previous_result'] = result
        baccarat['next_prediction'] = "N/A"
        baccarat['bet_amount'] = baccarat['base_amount']
        update_display(baccarat)
        return

    if baccarat['previous_result'] != 'T':
        pair = (baccarat['previous_result'], result)
        baccarat['pair_types'].append(pair)
        pair_type = "Even" if pair[0] == pair[1] else "Odd"
        baccarat['stats']['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1

    last_three = [p[1] for p in baccarat['pair_types'][-3:] if p[1] != 'T']
    if len(last_three) >= 3 and all(r == result for r in last_three):
        baccarat['streak_type'] = result
        baccarat['stats']['streaks'].append(len(last_three))
    else:
        baccarat['streak_type'] = None

    if len(baccarat['pair_types']) >= 5:
        recent_pairs = [p for p in baccarat['pair_types'][-10:] if p[0] !='T' and p[1] != 'T']
        if recent_pairs:
            odd_count = sum(1 for a, b in recent_pairs if a != b)
            even_count = sum(1 for a, b in recent_pairs if a == b)

            if baccarat['streak_type']:
                baccarat['next_prediction'] = "Player" if baccarat['streak_type'] == 'P' else "Banker"
                baccarat['current_dominance'] = f"Streak ({baccarat['streak_type']})"
            elif abs(odd_count - even_count) < 2:
                baccarat['current_dominance'] = "N/A"
                baccarat['next_prediction'] = "Hold"
            elif odd_count > even_count:
                baccarat['current_dominance'] = "Odd"
                baccarat['next_prediction'] = "Player" if result == 'B' else "Banker"
            else:
                baccarat['current_dominance'] = "Even"
                baccarat['next_prediction'] = "Player" if result == 'P' else "Banker"

            if baccarat['bet_amount'] == 0:
                baccarat['bet_amount'] = baccarat['base_amount']

            if len(baccarat['pair_types']) >= 6 and baccarat['state_history'][-1]['next_prediction'] != "Hold":
                previous_prediction = baccarat['state_history'][-1]['next_prediction']
                effective_bet = min(5 * baccarat['base_amount'], baccarat['bet_amount'])
                if (previous_prediction == "Player" and result == 'P'):
                    baccarat['result_tracker'] += effective_bet
                    baccarat['stats']['wins'] += 1
                    baccarat['consecutive_wins'] += 1
                    baccarat['consecutive_losses'] = 0
                    if baccarat['result_tracker'] > baccarat['profit_lock']:
                        baccarat['profit_lock'] = baccarat['result_tracker']
                        baccarat['result_tracker'] = 0.0
                        baccarat['bet_amount'] = baccarat['base_amount']
                        st.write(f"New profit lock achieved: ${baccarat['profit_lock']:.2f}! Betting reset.")
                        reset_betting(baccarat)
                        return
                    if baccarat['consecutive_wins'] >= 2:
                        baccarat['bet_amount'] = max(baccarat['base_amount'], baccarat['bet_amount'] - baccarat['base_amount'])
                elif (previous_prediction == "Banker" and result == 'B'):
                    baccarat['result_tracker'] += effective_bet * 0.95
                    baccarat['stats']['wins'] += 1
                    baccarat['consecutive_wins'] += 1
                    baccarat['consecutive_losses'] = 0
                    if baccarat['result_tracker'] > baccarat['profit_lock']:
                        baccarat['profit_lock'] = baccarat['result_tracker']
                        baccarat['result_tracker'] = 0.0
                        baccarat['bet_amount'] = baccarat['base_amount']
                        st.write(f"New profit lock achieved: ${baccarat['profit_lock']:.2f}! Betting reset.")
                        reset_betting(baccarat)
                        return
                    if baccarat['consecutive_wins'] >= 2:
                        baccarat['bet_amount'] = max(baccarat['base_amount'], baccarat['bet_amount'] - baccarat['base_amount'])
                else:
                    baccarat['result_tracker'] -= effective_bet
                    baccarat['stats']['losses'] += 1
                    baccarat['consecutive_losses'] += 1
                    baccarat['consecutive_wins'] = 0
                    if baccarat['consecutive_losses'] >= 3:
                        baccarat['bet_amount'] = min(5 * baccarat['base_amount'], baccarat['bet_amount'] * 2)
                    elif baccarat['streak_type']:
                        baccarat['bet_amount'] = min(5 * baccarat['base_amount'], baccarat['bet_amount'] + baccarat['base_amount'])
                    else:
                        baccarat['bet_amount'] = min(5 * baccarat['base_amount'], baccarat['bet_amount'] + baccarat['base_amount'])

        if baccarat['result_tracker'] <= -10 * baccarat['base_amount']:
            st.write("Loss limit reached. Resetting to resume betting.")
            baccarat['bet_amount'] = baccarat['base_amount']
            reset_betting(baccarat)
            return

    baccarat['previous_result'] = result
    update_display(baccarat)

def undo(baccarat):
    if not baccarat['state_history']:
        st.write("No actions to undo.")
        return

    last_state = baccarat['state_history'].pop()
    baccarat['pair_types'] = last_state['pair_types']
    baccarat['previous_result'] = last_state['previous_result']
    baccarat['result_tracker'] = last_state['result_tracker']
    baccarat['profit_lock'] = last_state['profit_lock']
    baccarat['bet_amount'] = last_state['bet_amount']
    baccarat['current_dominance'] = last_state['current_dominance']
    baccarat['next_prediction'] = last_state['next_prediction']
    baccarat['consecutive_wins'] = last_state['consecutive_wins']
    baccarat['consecutive_losses'] = last_state['consecutive_losses']
    baccarat['streak_type'] = last_state['streak_type']
    baccarat['stats'] = last_state['stats']

    update_display(baccarat)
    st.write("Last action undone.")

if __name__ == "__main__":
    main()
