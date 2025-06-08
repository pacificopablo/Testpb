import streamlit as st
import random
import pandas as pd
import uuid

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
        st.session_state.stats = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'streaks': [],
            'odd_pairs': 0,
            'even_pairs': 0,
            'bet_history': []
        }

def set_base_amount():
    """Set the base amount from user input."""
    try:
        amount = float(st.session_state.base_amount_input)
        if 1 <= amount <= 100:
            st.session_state.base_amount = amount
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.alert = {"type": "success", "message": "Base amount updated successfully.", "id": str(uuid.uuid4())}
        else:
            st.session_state.alert = {"type": "error", "message": "Base amount must be between $1 and $100.", "id": str(uuid.uuid4())}
    except ValueError:
        st.session_state.alert = {"type": "error", "message": "Please enter a valid number.", "id": str(uuid.uuid4())}

def reset_betting():
    """Reset betting parameters."""
    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.session_state.alert = {"type": "warning", "message": "Stop-loss reached. Resetting to resume betting.", "id": str(uuid.uuid4())}
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

    st.session_state.alert = {"type": "success", "message": "Betting reset.", "id": str(uuid.uuid4())}
    st.rerun()

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
    st.session_state.stats = {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'streaks': [],
        'odd_pairs': 0,
        'even_pairs': 0,
        'bet_history': []
    }
    st.session_state.alert = {"type": "success", "message": "All session data reset, profit lock reset.", "id": str(uuid.uuid4())}
    st.rerun()

def record_result(result):
    """Record a game result and update state."""
    # Save current state for undo
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

    # Handle Tie
    if result == 'T':
        st.session_state.stats['ties'] += 1
        st.session_state.previous_result = result
        st.session_state.alert = {"type": "info", "message": "Tie recorded. No bet placed.", "id": str(uuid.uuid4())}
        st.rerun()
        return

    # Handle first result
    if st.session_state.previous_result is None:
        st.session_state.previous_result = result
        st.session_state.next_prediction = "N/A"
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.alert = {"type": "info", "message": "Waiting for more results to start betting.", "id": str(uuid.uuid4())}
        st.rerun()
        return

    # Record pair
    if st.session_state.previous_result != 'T':
        pair = (st.session_state.previous_result, result)
        st.session_state.pair_types.append(pair)
        pair_type = "Even" if pair[0] == pair[1] else "Odd"
        st.session_state.stats['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1

    # Update prediction and bet amount
    if len(st.session_state.pair_types) >= 5:
        recent_pairs = st.session_state.pair_types[-10:]
        odd_count = sum(1 for a, b in recent_pairs if a != b)
        even_count = sum(1 for a, b in recent_pairs if a == b)
        last_three = [st.session_state.pair_types[-i][1] for i in range(1, min(4, len(st.session_state.pair_types)))]
        if len(last_three) >= 3 and all(r == last_three[0] for r in last_three):
            st.session_state.streak_type = last_three[0]
            st.session_state.next_prediction = "Player" if st.session_state.streak_type == 'P' else "Banker"
            st.session_state.current_dominance = f"Streak ({st.session_state.streak_type})"
            st.session_state.bet_amount = 2 * st.session_state.base_amount
            st.session_state.stats['streaks'].append(len(last_three))
        elif abs(odd_count - even_count) < 2:
            st.session_state.current_dominance = "N/A"
            st.session_state.next_prediction = "Hold"
            st.session_state.bet_amount = 0
            st.session_state.streak_type = None
        elif odd_count > even_count:
            st.session_state.current_dominance = "Odd"
            st.session_state.next_prediction = "Player" if st.session_state.previous_result == 'B' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.streak_type = None
        else:
            st.session_state.current_dominance = "Even"
            st.session_state.next_prediction = "Player" if st.session_state.previous_result == 'P' else "Banker"
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.streak_type = None

    # Evaluate bet outcome
    if len(st.session_state.pair_types) >= 6 and st.session_state.state_history[-1]['next_prediction'] != "Hold":
        previous_prediction = st.session_state.state_history[-1]['next_prediction']
        effective_bet = min(5 * st.session_state.base_amount, st.session_state.bet_amount)
        print(f"Bet: {previous_prediction}, Amount: {effective_bet}, Result: {result}, Tracker Before: {st.session_state.result_tracker}")  # Debug
        outcome = ""
        if previous_prediction == "Player" and result == 'P':
            st.session_state.result_tracker += effective_bet
            st.session_state.stats['wins'] += 1
            st.session_state.consecutive_wins += 1
            st.session_state.consecutive_losses = 0
            outcome = f"Won ${effective_bet:.2f}"
            st.session_state.alert = {"type": "success", "message": f"Bet won! +${effective_bet:.2f}", "id": str(uuid.uuid4())}
            if st.session_state.result_tracker > st.session_state.profit_lock:
                st.session_state.profit_lock = st.session_state.result_tracker
                st.session_state.result_tracker = 0.0
                st.session_state.bet_amount = st.session_state.base_amount
                st.session_state.alert = {"type": "info", "message": f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.", "id": str(uuid.uuid4())}
            elif st.session_state.consecutive_wins >= 2:
                st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
        elif previous_prediction == "Banker" and result == 'B':
            st.session_state.result_tracker += effective_bet * 0.95
            st.session_state.stats['wins'] += 1
            st.session_state.consecutive_wins += 1
            st.session_state.consecutive_losses = 0
            outcome = f"Won ${effective_bet * 0.95:.2f}"
            st.session_state.alert = {"type": "success", "message": f"Bet won! +${effective_bet * 0.95:.2f}", "id": str(uuid.uuid4())}
            if st.session_state.result_tracker > st.session_state.profit_lock:
                st.session_state.profit_lock = st.session_state.result_tracker
                st.session_state.result_tracker = 0.0
                st.session_state.bet_amount = st.session_state.base_amount
                st.session_state.alert = {"type": "info", "message": f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.", "id": str(uuid.uuid4())}
            elif st.session_state.consecutive_wins >= 2:
                st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
        else:
            st.session_state.result_tracker -= effective_bet
            st.session_state.stats['losses'] += 1
            st.session_state.consecutive_losses += 1
            st.session_state.consecutive_wins = 0
            outcome = f"Lost ${effective_bet:.2f}"
            st.session_state.alert = {"type": "error", "message": f"Bet lost! -${effective_bet:.2f}", "id": str(uuid.uuid4())}
            if st.session_state.consecutive_losses >= 3:
                st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount * 2)
            elif st.session_state.streak_type:
                st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
            else:
                st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
        # Record bet outcome
        st.session_state.stats['bet_history'].append({
            'prediction': previous_prediction,
            'result': result,
            'bet_amount': effective_bet,
            'outcome': outcome
        })
        print(f"Tracker After: {st.session_state.result_tracker}")  # Debug

    # Check stop-loss
    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.session_state.alert = {"type": "warning", "message": "Loss limit reached. Resetting to resume betting.", "id": str(uuid.uuid4())}
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.next_prediction = "Player" if result == 'B' else "Banker" if result == 'P' else random.choice(["Player", "Banker"])
        st.rerun()
        return

    st.session_state.previous_result = result
    if len(st.session_state.pair_types) < 5:
        st.session_state.alert = {"type": "info", "message": f"Result recorded. Need {5 - len(st.session_state.pair_types)} more results to start betting.", "id": str(uuid.uuid4())}
    st.rerun()

def undo():
    """Undo the last action."""
    if not st.session_state.state_history:
        st.session_state.alert = {"type": "error", "message": "No actions to undo.", "id": str(uuid.uuid4())}
        st.rerun()
        return

    last_state = st.session_state.state_history.pop()
    print(f"Restoring state: {last_state}")  # Debug
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
    st.session_state.alert = {"type": "success", "message": "Last action undone.", "id": str(uuid.uuid4())}
    st.rerun()

def simulate_games(num_games=100):
    """Simulate a number of games."""
    outcomes = ['P', 'B', 'T']
    weights = [0.446, 0.458, 0.096]
    for _ in range(num_games):
        result = random.choices(outcomes, weights)[0]
        record_result(result)
    st.session_state.alert = {"type": "success", "message": f"Simulated {num_games} games. Check stats and bet history for results.", "id": str(uuid.uuid4())}
    st.rerun()

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()

    # Custom CSS with Tailwind CDN
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
        body, .stApp {
            background-color: #1F2528;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #E5E7EB;
        }
        .card {
            background-color: #2C2F33;
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #6366F1;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: background-color 0.2s;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #4F46E5;
        }
        .stNumberInput input {
            background-color: #23272A;
            color: white;
            border: 1px solid #4B5563;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        .stDataFrame table {
            background-color: #23272A;
            color: white;
            border-collapse: collapse;
        }
        .stDataFrame th {
            background-color: #374151;
            color: white;
            font-weight: 600;
            padding: 0.75rem;
        }
        .stDataFrame td {
            padding: 0.75rem;
            border-bottom: 1px solid #4B5563;
        }
        .stDataFrame tr:nth-child(even) {
            background-color: #2D3748;
        }
        h1 {
            font-size: 2.25rem;
            font-weight: 700;
            color: #F3F4F6;
            margin-bottom: 1rem;
        }
        h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #D1D5DB;
            margin-bottom: 0.75rem;
        }
        .alert {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .alert-success {
            background-color: #10B981;
            color: white;
        }
        .alert-error {
            background-color: #EF4444;
            color: white;
        }
        .alert-info {
            background-color: #3B82F6;
            color: white;
        }
        .alert-warning {
            background-color: #F59E0B;
            color: white;
        }
        .sidebar .stButton>button {
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display alert if present
    if 'alert' in st.session_state:
        alert_class = f"alert alert-{st.session_state.alert['type']}"
        st.markdown(f'<div class="{alert_class}">{st.session_state.alert["message"]}</div>', unsafe_allow_html=True)

    # Title
    st.markdown('<h1>Baccarat Predictor</h1>', unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.markdown('<h2>Controls</h2>', unsafe_allow_html=True)
        with st.expander("Bet Settings", expanded=True):
            st.number_input("Base Amount ($1-$100)", min_value=1.0, max_value=100.0, value=st.session_state.base_amount, step=1.0, key="base_amount_input")
            if st.button("Set Amount"):
                set_base_amount()

        with st.expander("Session Actions"):
            if st.button("Reset Bet"):
                reset_betting()
            if st.button("Reset Session"):
                reset_all()
            if st.button("New Session"):
                reset_all()
                st.session_state.alert = {"type": "success", "message": "New session started.", "id": str(uuid.uuid4())}
                st.rerun()
            if st.button("Simulate 100 Games"):
                simulate_games(100)

    # Main content with card layout
    with st.container():
        st.markdown('<h2>Betting Overview</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Next Bet</p>
                    <p class="text-xl font-bold text-white">{st.session_state.next_prediction}</p>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Bet Amount</p>
                    <p class="text-xl font-bold text-white">{'No Bet' if st.session_state.bet_amount == 0 else f'${st.session_state.bet_amount:.2f}'}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Bankroll</p>
                    <p class="text-xl font-bold text-white">${st.session_state.result_tracker:.2f}</p>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Profit Lock</p>
                    <p class="text-xl font-bold text-white">${st.session_state.profit_lock:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        # Statistics
        total_games = st.session_state.stats['wins'] + st.session_state.stats['losses']
        win_rate = (st.session_state.stats['wins'] / total_games * 100) if total_games > 0 else 0
        avg_streak = sum(st.session_state.stats['streaks']) / len(st.session_state.stats['streaks']) if st.session_state.stats['streaks'] else 0
        st.markdown(f"""
            <div class="card">
                <p class="text-sm font-semibold text-gray-400">Statistics</p>
                <p class="text-base text-white">Win Rate: {win_rate:.1f}%</p>
                <p class="text-base text-white">Avg Streak: {avg_streak:.1f}</p>
                <p class="text-base text-white">Patterns: Odd: {st.session_state.stats['odd_pairs']}, Even: {st.session_state.stats['even_pairs']}</p>
                <p class="text-base text-white">Streak: {st.session_state.streak_type if st.session_state.streak_type else 'None'}</p>
            </div>
        """, unsafe_allow_html=True)

        # Result input buttons
        st.markdown('<h2>Record Result</h2>', unsafe_allow_html=True)
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
        st.markdown('<h2>Deal History</h2>', unsafe_allow_html=True)
        if st.session_state.pair_types:
            history_data = [
                {"Pair": f"{pair[0]}{pair[1]}", "Type": "Even" if pair[0] == pair[1] else "Odd"}
                for pair in st.session_state.pair_types[-100:]
            ]
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=300)
        else:
            st.markdown('<p class="text-gray-400">No history yet.</p>', unsafe_allow_html=True)

        # Bet History
        st.markdown('<h2>Bet History</h2>', unsafe_allow_html=True)
        if st.session_state.stats.get('bet_history'):
            bet_history = pd.DataFrame(st.session_state.stats['bet_history'])
            st.dataframe(bet_history, use_container_width=True, height=200)
        else:
            st.markdown('<p class="text-gray-400">No bets placed yet.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
