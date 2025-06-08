import streamlit as st
import random
import pandas as pd

def initialize_session_state():
    """Initialize session state variables if not already set."""
    if 'pair_types' not in st.session_state:
        st.session_state.pair_types = []
        st.session_state.next_prediction = "N/A"
        st.session_state.base_amount = 10.0
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.result_tracker = 0.0
        st.session_state.profit_lock = 0.0
        st.session_state.previous_result = None
        st.session_state.state_history = []
        st.session_state.current_dominance = "N/A"
        st.session_state.streak_type = None
        st.session_state.consecutive_wins = 0
        st.session_state.consecutive_losses = 0
        st.session_state.stats = {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}

def set_base_amount():
    """Set the base amount from user input."""
    try:
        amount = float(st.session_state.base_amount_input)
        if 1 <= amount <= 100:
            st.session_state.base_amount = amount
            st.session_state.bet_amount = st.session_state.base_amount
            st.success("Base amount updated successfully.")
        else:
            st.error("Base amount must be between $1 and $100.")
    except ValueError:
        st.error("Please enter a valid number.")

def reset_betting():
    """Reset betting parameters."""
    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.warning("Stop-loss reached. Resetting to resume betting.")
    if st.session_state.result_tracker >= 0:
        st.session_state.result_tracker = 0.0
    st.session_state.bet_amount = st.session_state.base_amount
    st.session_state.consecutive_wins = 0
    st.session_state.consecutive_losses = 0
    st.session_state.streak_type = None

    if len(st.session_state.pair_types) >= 5:
        recent_pairs = st.session_state.pair_types[-10:]
        odd_count = sum(1 for a, b in recent_pairs if a != b)
        even_count = sum(1 for a, b in recent_pairs if a == b)
        result = st.session_state.previous_result
        if abs(odd_count - even_count) < 2:
            st.session_state.current_dominance = "N/A"
            st.session_state.next_prediction = "Hold"
            st.session_state.bet_amount = 0
        elif odd_count > even_count:
            st.session_state.current_dominance = "Odd"
            st.session_state.next_prediction = "Player" if result == 'B' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount
        else:
            st.session_state.current_dominance = "Even"
            st.session_state.next_prediction = "Player" if result == 'P' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount
        last_three = [st.session_state.pair_types[-i][1] for i in range(1, min(4, len(st.session_state.pair_types)))]
        if len(last_three) >= 3 and all(r == last_three[0] for r in last_three):
            st.session_state.streak_type = last_three[0]
            st.session_state.next_prediction = "Player" if st.session_state.streak_type == 'P' else "Banker"
            st.session_state.current_dominance = f"Streak ({st.session_state.streak_type})"
            st.session_state.bet_amount = 2 * st.session_state.base_amount
    else:
        st.session_state.next_prediction = "N/A"
        st.session_state.current_dominance = "N/A"
        st.session_state.streak_type = None
        st.session_state.bet_amount = st.session_state.base_amount

    st.success("Betting reset.")

def reset_all():
    """Reset all session data."""
    st.session_state.pair_types = []
    st.session_state.result_tracker = 0.0
    st.session_state.profit_lock = 0.0
    st.session_state.bet_amount = st.session_state.base_amount
    st.session_state.base_amount = 10.0
    st.session_state.next_prediction = "N/A"
    st.session_state.previous_result = None
    st.session_state.state_history = []
    st.session_state.current_dominance = "N/A"
    st.session_state.consecutive_wins = 0
    st.session_state.consecutive_losses = 0
    st.session_state.streak_type = None
    st.session_state.stats = {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}
    st.success("All session data reset, profit lock reset.")

def record_result(result):
    """Record a game result and update state."""
    state = {
        'pair_types': st.session_state.pair_types.copy(),
        'previous_result': st.session_state.previous_result,
        'result_tracker': st.session_state.result_tracker,
        'profit_lock': st.session_state.profit_lock,
        'bet_amount': st.session_state.bet_amount,
        'current_dominance': st.session_state.current_dominance,
        'next_prediction': st.session_state.next_prediction,
        'consecutive_wins': st.session_state.consecutive_wins,
        'consecutive_losses': st.session_state.consecutive_losses,
        'streak_type': st.session_state.streak_type,
        'stats': st.session_state.stats.copy()
    }
    st.session_state.state_history.append(state)

    if result == 'T':
        st.session_state.stats['ties'] += 1
        return

    if st.session_state.previous_result is None:
        st.session_state.previous_result = result
        st.session_state.next_prediction = "N/A"
        st.session_state.bet_amount = st.session_state.base_amount
        return

    if st.session_state.previous_result != 'T':
        pair = (st.session_state.previous_result, result)
        st.session_state.pair_types.append(pair)
        pair_type = "Even" if pair[0] == pair[1] else "Odd"
        st.session_state.stats['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1

    last_three = [st.session_state.pair_types[-i][1] for i in range(1, min(4, len(st.session_state.pair_types)))]
    if len(last_three) >= 3 and all(r == result for r in last_three):
        st.session_state.streak_type = result
        st.session_state.stats['streaks'].append(len(last_three))
    else:
        st.session_state.streak_type = None

    if len(st.session_state.pair_types) >= 5:
        recent_pairs = st.session_state.pair_types[-10:]
        odd_count = sum(1 for a, b in recent_pairs if a != b)
        even_count = sum(1 for a, b in recent_pairs if a == b)

        if st.session_state.streak_type:
            st.session_state.next_prediction = "Player" if st.session_state.streak_type == 'P' else "Banker"
            st.session_state.current_dominance = f"Streak ({st.session_state.streak_type})"
            st.session_state.bet_amount = 2 * st.session_state.base_amount
        elif abs(odd_count - even_count) < 2:
            st.session_state.current_dominance = "N/A"
            st.session_state.next_prediction = "Hold"
            st.session_state.bet_amount = 0
        elif odd_count > even_count:
            st.session_state.current_dominance = "Odd"
            st.session_state.next_prediction = "Player" if result == 'B' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount
        else:
            st.session_state.current_dominance = "Even"
            st.session_state.next_prediction = "Player" if result == 'P' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount

        if st.session_state.bet_amount == 0:
            st.session_state.bet_amount = st.session_state.base_amount

        if len(st.session_state.pair_types) >= 6 and st.session_state.state_history[-1]['next_prediction'] != "Hold":
            previous_prediction = st.session_state.state_history[-1]['next_prediction']
            effective_bet = min(5 * st.session_state.base_amount, st.session_state.bet_amount)
            if (previous_prediction == "Player" and result == 'P'):
                st.session_state.result_tracker += effective_bet
                st.session_state.stats['wins'] += 1
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                if st.session_state.result_tracker > st.session_state.profit_lock:
                    st.session_state.profit_lock = st.session_state.result_tracker
                    st.session_state.result_tracker = 0.0
                    st.session_state.bet_amount = st.session_state.base_amount
                    st.info(f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.")
                    return
                if st.session_state.consecutive_wins >= 2:
                    st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
            elif (previous_prediction == "Banker" and result == 'B'):
                st.session_state.result_tracker += effective_bet * 0.95
                st.session_state.stats['wins'] += 1
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                if st.session_state.result_tracker > st.session_state.profit_lock:
                    st.session_state.profit_lock = st.session_state.result_tracker
                    st.session_state.result_tracker = 0.0
                    st.session_state.bet_amount = st.session_state.base_amount
                    st.info(f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.")
                    return
                if st.session_state.consecutive_wins >= 2:
                    st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
            else:
                st.session_state.result_tracker -= effective_bet
                st.session_state.stats['losses'] += 1
                st.session_state.consecutive_losses += 1
                st.session_state.consecutive_wins = 0
                if st.session_state.consecutive_losses >= 3:
                    st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount * 2)
                elif st.session_state.streak_type:
                    st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
                else:
                    st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)

    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.warning("Loss limit reached. Resetting to resume betting.")
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.next_prediction = "Player" if result == 'B' else "Banker" if result == 'P' else random.choice(["Player", "Banker"])
        return

    st.session_state.previous_result = result

def undo():
    """Undo the last action."""
    if not st.session_state.state_history:
        st.error("No actions to undo.")
        return

    last_state = st.session_state.state_history.pop()
    st.session_state.pair_types = last_state['pair_types']
    st.session_state.previous_result = last_state['previous_result']
    st.session_state.result_tracker = last_state['result_tracker']
    st.session_state.profit_lock = last_state['profit_lock']
    st.session_state.bet_amount = last_state['bet_amount']
    st.session_state.current_dominance = last_state['current_dominance']
    st.session_state.next_prediction = last_state['next_prediction']
    st.session_state.consecutive_wins = last_state['consecutive_wins']
    st.session_state.consecutive_losses = last_state['consecutive_losses']
    st.session_state.streak_type = last_state['streak_type']
    st.session_state.stats = last_state['stats']
    st.success("Last action undone.")

def simulate_games(num_games=100):
    """Simulate a number of games."""
    outcomes = ['P', 'B', 'T']
    weights = [0.446, 0.458, 0.096]
    for _ in range(num_games):
        result = random.choices(outcomes, weights)[0]
        record_result(result)
    st.success(f"Simulated {num_games} games. Check stats for results.")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #2C2F33;
            color: white;
        }
        .stButton>button {
            background-color: #7289DA;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #99AAB5;
        }
        .stTextInput>div>input {
            background-color: #23272A;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stDataFrame {
            background-color: #23272A;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("Enhanced Dominant Pairs Baccarat Predictor")

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        st.number_input("Base Amount ($1-$100)", min_value=1.0, max_value=100.0, value=st.session_state.base_amount, step=1.0, key="base_amount_input")
        if st.button("Set Amount"):
            set_base_amount()

        st.header("Session Actions")
        if st.button("Reset Bet"):
            reset_betting()
        if st.button("Reset Session"):
            reset_all()
        if st.button("New Session"):
            reset_all()
            st.success("New session started.")
        if st.button("Simulate 100 Games"):
            simulate_games(100)

    # Main content
    st.subheader("Betting Information")
    st.markdown(f"**Bet Amount:** {'No Bet' if st.session_state.bet_amount == 0 else f'${st.session_state.bet_amount:.2f}'}")
    st.markdown(f"**Bankroll:** ${st.session_state.result_tracker:.2f}")
    st.markdown(f"**Profit Lock:** ${st.session_state.profit_lock:.2f}")
    st.markdown(f"**Bet:** {st.session_state.next_prediction}")
    st.markdown(f"**Streak:** {st.session_state.streak_type if st.session_state.streak_type else 'None'}")

    # Stats
    total_games = st.session_state.stats['wins'] + st.session_state.stats['losses']
    win_rate = (st.session_state.stats['wins'] / total_games * 100) if total_games > 0 else 0
    avg_streak = sum(st.session_state.stats['streaks']) / len(st.session_state.stats['streaks']) if st.session_state.stats['streaks'] else 0
    st.markdown(f"**Win Rate:** {win_rate:.1f}% | **Avg Streak:** {avg_streak:.1f} | **Patterns:** Odd: {st.session_state.stats['odd_pairs']}, Even: {st.session_state.stats['even_pairs']}")

    # Result input buttons
    st.subheader("Record Result")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player"):
            record_result('P')
    with col2:
        if st.button("Banker"):
            record_result('B')
    with col3:
        if st.button("Tie"):
            record_result('T')
    with col4:
        if st.button("Undo"):
            undo()

    # Deal History
    st.subheader("Deal History")
    if st.session_state.pair_types:
        history_data = [
            {"Pair": f"{pair[0]}{pair[1]}", "Type": "Even" if pair[0] == pair[1] else "Odd"}
            for pair in st.session_state.pair_types[-100:]
        ]
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=200)
    else:
        st.write("No history yet.")

if __name__ == "__main__":
    main()
