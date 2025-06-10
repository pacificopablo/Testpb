import streamlit as st
import random
import pandas as pd
import uuid
from collections import deque

def initialize_session_state():
    """Initialize session state variables if not already set."""
    if 'pair_types' not in st.session_state:
        st.session_state.pair_types = deque(maxlen=100)  # Store up to 100 pairs
        st.session_state.results = deque(maxlen=200)  # Store raw results
        st.session_state.base_amount = 10.0
        st.session_state.result_tracker = 1000.0  # Start with initial bankroll
        st.session_state.session_profit = 0.0  # Track profits separately
        st.session_state.profit_lock = 0.0
        st.session_state.previous_result = None
        st.session_state.state_history = []
        st.session_state.next_prediction = "N/A"
        st.session_state.current_dominance = "N/A"
        st.session_state.bet_amount = 10.0
        st.session_state.max_profit = 0.0
        st.session_state.stats = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'streaks': [],
            'odd_pairs': 0,
            'even_pairs': 0,
            'alternating_pairs': 0,
            'bet_history': []
        }
        st.session_state.alerts = []  # List to store active alerts
        st.session_state.profit_lock_threshold = 2 * st.session_state.base_amount
        # Money management variables
        st.session_state.betting_strategy = "Flatbet"
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.stop_loss = 0.8  # Stop at 80% of initial bankroll
        st.session_state.win_limit = 1.5  # Win at 150% of initial bankroll
        st.session_state.initial_bankroll = 1000.0  # Default initial bankroll
        st.session_state.game_count = 0  # Track number of recorded games

def set_money_management():
    """Set the base amount, initial bankroll, and money management parameters from user input."""
    try:
        base_amount = float(st.session_state.base_amount_input)
        initial_bankroll = float(st.session_state.initial_bankroll_input)
        if 1 <= base_amount <= 100:
            if initial_bankroll >= 10:
                if base_amount > initial_bankroll * 0.05:
                    st.session_state.alerts.append({"type": "warning", "message": "Base amount exceeds 5% of initial bankroll, which may be risky.", "id": str(uuid.uuid4())})
                st.session_state.base_amount = base_amount
                st.session_state.initial_bankroll = initial_bankroll
                st.session_state.result_tracker = initial_bankroll
                st.session_state.session_profit = 0.0
                st.session_state.profit_lock_threshold = 2 * base_amount
                st.session_state.bet_amount = base_amount
                st.session_state.game_count = 0
                st.session_state.alerts.append({"type": "success", "message": f"Base amount (${base_amount:.2f}) and initial bankroll (${initial_bankroll:.2f}) updated successfully.", "id": str(uuid.uuid4())})
            else:
                st.session_state.alerts.append({"type": "error", "message": "Initial bankroll must be at least $10.", "id": str(uuid.uuid4())})
        else:
            st.session_state.alerts.append({"type": "error", "message": "Base amount must be between $1 and $100.", "id": str(uuid.uuid4())})
    except ValueError:
        st.session_state.alerts.append({"type": "error", "message": "Please enter valid numbers for base amount and initial bankroll.", "id": str(uuid.uuid4())})

def set_betting_strategy():
    """Set the betting strategy and reset strategy-specific parameters."""
    st.session_state.betting_strategy = st.session_state.strategy_select
    if st.session_state.betting_strategy == "T3":
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
    st.session_state.bet_amount = st.session_state.base_amount
    st.session_state.alerts.append({"type": "success", "message": f"Betting strategy set to {st.session_state.betting_strategy}.", "id": str(uuid.uuid4())})

def reset_betting():
    """Reset betting parameters."""
    st.session_state.result_tracker = st.session_state.initial_bankroll
    st.session_state.session_profit = 0.0
    st.session_state.bet_amount = st.session_state.base_amount
    st.session_state.max_profit = 0.0
    st.session_state.next_prediction = "N/A"
    st.session_state.current_dominance = "N/A"
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.game_count = 0
    st.session_state.alerts.append({"type": "success", "message": "Betting parameters reset.", "id": str(uuid.uuid4())})

def reset_all():
    """Reset all session data."""
    st.session_state.pair_types = deque(maxlen=100)
    st.session_state.results = deque(maxlen=200)
    st.session_state.result_tracker = 1000.0
    st.session_state.session_profit = 0.0
    st.session_state.profit_lock = 0.0
    st.session_state.base_amount = 10.0
    st.session_state.previous_result = None
    st.session_state.state_history = []
    st.session_state.next_prediction = "N/A"
    st.session_state.current_dominance = "N/A"
    st.session_state.bet_amount = 10.0
    st.session_state.max_profit = 0.0
    st.session_state.stats = {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'streaks': [],
        'odd_pairs': 0,
        'even_pairs': 0,
        'alternating_pairs': 0,
        'bet_history': []
    }
    st.session_state.profit_lock_threshold = 2 * st.session_state.base_amount
    st.session_state.betting_strategy = "Flatbet"
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.initial_bankroll = 1000.0
    st.session_state.game_count = 0
    st.session_state.alerts.append({"type": "success", "message": "All session data reset, profit lock reset.", "id": str(uuid.uuid4())})

def apply_betting_strategy(outcome, result, bet_selection):
    """Apply the selected betting strategy and return bet amount and outcome."""
    bet_amount = st.session_state.bet_amount
    bet_outcome = None

    if st.session_state.betting_strategy == "Flatbet":
        bet_amount = min(st.session_state.base_amount, st.session_state.result_tracker)
    elif st.session_state.betting_strategy == "T3":
        proposed_bet = st.session_state.base_amount * st.session_state.t3_level
        bet_amount = min(proposed_bet, st.session_state.result_tracker)
        if bet_amount < proposed_bet:
            st.session_state.t3_level = max(1, int(bet_amount / st.session_state.base_amount))
            st.session_state.alerts.append({"type": "warning", "message": f"T3 bet reduced to ${bet_amount:.2f} due to bankroll limit.", "id": str(uuid.uuid4())})

        # Update T3 results based on outcome
        if outcome:
            if outcome == 'win':
                if len(st.session_state.t3_results) == 0:  # First result in sequence
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                st.session_state.t3_results.append('W')
            elif outcome == 'loss':
                st.session_state.t3_results.append('L')
            
            # Update T3 level after 3 results
            if len(st.session_state.t3_results) == 3:
                wins = st.session_state.t3_results.count('W')
                losses = st.session_state.t3_results.count('L')
                if wins > losses:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                elif losses > wins:
                    st.session_state.t3_level += 1
                st.session_state.t3_results = []
        
        # Cap T3 level to prevent excessive bets
        max_t3_bet = st.session_state.result_tracker / st.session_state.base_amount
        st.session_state.t3_level = min(st.session_state.t3_level, max(1, int(max_t3_bet)))

    # Update bet amount for next round
    next_bet = st.session_state.base_amount if st.session_state.betting_strategy == "Flatbet" else st.session_state.base_amount * st.session_state.t3_level
    st.session_state.bet_amount = min(next_bet, st.session_state.result_tracker)

    return bet_amount, bet_outcome

def record_result(result):
    """Record a game result and update state with Dominant Pairs betting logic."""
    if st.session_state.initial_bankroll > 0:
        if st.session_state.result_tracker <= st.session_state.initial_bankroll * st.session_state.stop_loss:
            st.session_state.alerts.append({"type": "warning", "message": f"Stop-loss reached ({st.session_state.stop_loss*100:.0f}% of initial bankroll). Please reset betting to continue.", "id": str(uuid.uuid4())})
            return
        if st.session_state.session_profit >= st.session_state.initial_bankroll * (st.session_state.win_limit - 1):
            lock_amount = st.session_state.session_profit
            st.session_state.profit_lock += lock_amount
            st.session_state.session_profit = 0.0
            st.session_state.result_tracker = st.session_state.initial_bankroll
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.alerts.append({"type": "success", "message": f"Win limit reached! Locked ${lock_amount:.2f}. Total locked: ${st.session_state.profit_lock:.2f}. Please reset betting.", "id": str(uuid.uuid4())})
            return

    st.session_state.game_count += 1

    state = {
        'pair_types': list(st.session_state.pair_types),
        'results': list(st.session_state.results),
        'previous_result': st.session_state.previous_result,
        'result_tracker': st.session_state.result_tracker,
        'session_profit': st.session_state.session_profit,
        'profit_lock': st.session_state.profit_lock,
        'stats': st.session_state.stats.copy(),
        'next_prediction': st.session_state.next_prediction,
        'current_dominance': st.session_state.current_dominance,
        'bet_amount': st.session_state.bet_amount,
        'max_profit': st.session_state.max_profit,
        't3_level': st.session_state.t3_level,
        't3_results': st.session_state.t3_results.copy(),
        'betting_strategy': st.session_state.betting_strategy,
        'initial_bankroll': st.session_state.initial_bankroll,
        'game_count': st.session_state.game_count
    }
    st.session_state.state_history.append(state)

    if result == 'T':
        st.session_state.stats['ties'] += 1
        st.session_state.previous_result = result
        st.session_state.alerts.append({"type": "info", "message": "Tie recorded.", "id": str(uuid.uuid4())})
        return

    st.session_state.results.append(result)

    if st.session_state.previous_result is None or st.session_state.previous_result == 'T':
        st.session_state.previous_result = result
        st.session_state.next_prediction = "N/A"
        st.session_state.alerts.append({"type": "info", "message": f"Result {result} recorded.", "id": str(uuid.uuid4())})
        return

    pair = (st.session_state.previous_result, result)
    st.session_state.pair_types.append(pair)
    pair_type = "Even" if pair[0] == pair[1] else "Odd"
    st.session_state.stats['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1

    if len(st.session_state.pair_types) >= 2:
        last_two_pairs = list(st.session_state.pair_types)[-2:]
        if last_two_pairs[0][1] != last_two_pairs[1][1]:
            st.session_state.stats['alternating_pairs'] += 1

    if len(st.session_state.pair_types) >= 5:
        odd_count = st.session_state.stats['odd_pairs']
        even_count = st.session_state.stats['even_pairs']
        if odd_count > even_count:
            st.session_state.current_dominance = "Odd"
            st.session_state.next_prediction = "Player" if result == "B" else "Banker"
        else:
            st.session_state.current_dominance = "Even"
            st.session_state.next_prediction = "Player" if result == "P" else "Banker"

        if st.session_state.bet_amount == 0.0:
            st.session_state.bet_amount = min(st.session_state.base_amount, st.session_state.result_tracker)

        if len(st.session_state.pair_types) >= 6:
            previous_prediction = st.session_state.state_history[-1]['next_prediction']
            if previous_prediction != "N/A":
                bet_amount, _ = apply_betting_strategy(None, result, previous_prediction)
                if bet_amount > 0:
                    if (previous_prediction == "Player" and result == "P") or \
                       (previous_prediction == "Banker" and result == "B"):
                        win_amount = bet_amount * 0.95 if previous_prediction == "Banker" else bet_amount
                        st.session_state.result_tracker += win_amount
                        st.session_state.session_profit += win_amount
                        st.session_state.stats['wins'] += 1
                        outcome = 'win'
                        if st.session_state.session_profit >= st.session_state.profit_lock_threshold:
                            lock_amount = st.session_state.session_profit
                            st.session_state.profit_lock += lock_amount
                            st.session_state.session_profit = 0.0
                            st.session_state.result_tracker = st.session_state.initial_bankroll
                            st.session_state.bet_amount = st.session_state.base_amount
                            st.session_state.t3_level = 1
                            st.session_state.t3_results = []
                            st.session_state.alerts.append({"type": "success", "message": f"Profit locked at ${lock_amount:.2f}. Total locked: ${st.session_state.profit_lock:.2f}.", "id": str(uuid.uuid4())})
                        elif st.session_state.session_profit > st.session_state.max_profit:
                            st.session_state.max_profit = st.session_state.session_profit
                            st.session_state.alerts.append({"type": "success", "message": f"New max profit: ${st.session_state.max_profit:.2f}", "id": str(uuid.uuid4())})
                    else:
                        bet_amount = min(bet_amount, st.session_state.result_tracker)
                        st.session_state.result_tracker -= bet_amount
                        st.session_state.session_profit -= bet_amount
                        st.session_state.stats['losses'] += 1
                        outcome = 'loss'
                        st.session_state.alerts.append({"type": "error", "message": f"Loss! -${bet_amount:.2f}", "id": str(uuid.uuid4())})
                        if st.session_state.result_tracker <= 0:
                            st.session_state.alerts.append({"type": "error", "message": "Bankroll depleted! Please reset betting to continue.", "id": str(uuid.uuid4())})
                            return

                    apply_betting_strategy(outcome, result, previous_prediction)

                    st.session_state.stats['bet_history'].append({
                        'Bet': previous_prediction,
                        'Result': result,
                        'Amount': bet_amount,
                        'Outcome': 'Win' if outcome == 'win' else 'Loss',
                        'Bankroll': st.session_state.result_tracker,
                        'Profit': st.session_state.session_profit,
                        'Strategy': st.session_state.betting_strategy,
                        'T3_Level': st.session_state.t3_level if st.session_state.betting_strategy == "T3" else None
                    })

    st.session_state.previous_result = result
    st.session_state.alerts.append({"type": "info", "message": f"Result {result} recorded. Next bet: {st.session_state.next_prediction} (${st.session_state.bet_amount:.2f})", "id": str(uuid.uuid4())})

def undo():
    """Undo the last action."""
    if not st.session_state.state_history:
        st.session_state.alerts.append({"type": "error", "message": "No actions to undo.", "id": str(uuid.uuid4())})
        return

    last_state = st.session_state.state_history.pop()
    st.session_state.pair_types = deque(last_state['pair_types'], maxlen=100)
    st.session_state.results = deque(last_state['results'], maxlen=200)
    st.session_state.previous_result = last_state['previous_result']
    st.session_state.result_tracker = last_state['result_tracker']
    st.session_state.session_profit = last_state['session_profit']
    st.session_state.profit_lock = last_state['profit_lock']
    st.session_state.stats = last_state['stats']
    st.session_state.next_prediction = last_state['next_prediction']
    st.session_state.current_dominance = last_state['current_dominance']
    st.session_state.bet_amount = last_state['bet_amount']
    st.session_state.max_profit = last_state['max_profit']
    st.session_state.t3_level = last_state['t3_level']
    st.session_state.t3_results = last_state['t3_results']
    st.session_state.betting_strategy = last_state['betting_strategy']
    st.session_state.initial_bankroll = last_state['initial_bankroll']
    st.session_state.game_count = last_state['game_count']
    st.session_state.alerts.append({"type": "success", "message": "Last action undone.", "id": str(uuid.uuid4())})

def simulate_games():
    """Simulate 100 games."""
    outcomes = ['P', 'B', 'T']
    weights = [0.446, 0.458, 0.096]
    for _ in range(100):
        if st.session_state.result_tracker <= 0:
            st.session_state.alerts.append({"type": "error", "message": "Bankroll depleted during simulation!", "id": str(uuid.uuid4())})
            break
        if st.session_state.result_tracker <= st.session_state.initial_bankroll * st.session_state.stop_loss:
            st.session_state.alerts.append({"type": "warning", "message": f"Stop-loss reached during simulation.", "id": str(uuid.uuid4())})
            break
        if st.session_state.session_profit >= st.session_state.initial_bankroll * (st.session_state.win_limit - 1):
            lock_amount = st.session_state.session_profit
            st.session_state.profit_lock += lock_amount
            st.session_state.session_profit = 0.0
            st.session_state.result_tracker = st.session_state.initial_bankroll
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.alerts.append({"type": "success", "message": f"Win limit reached during simulation! Locked ${lock_amount:.2f}.", "id": str(uuid.uuid4())})
            break
        result = random.choices(outcomes, weights)[0]
        record_result(result)
    st.session_state.alerts.append({"type": "success", "message": "Simulated up to 100 games. Check stats and history for results.", "id": str(uuid.uuid4())})

def clear_alerts():
    """Clear all alerts."""
    st.session_state.alerts = []

def main():
    """Main Streamlit application."""
    initialize_session_state()

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
        .card-player {
            background-color: #3B82F6;
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .card-banker {
            background-color: #EF4444;
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
        .stButton>button:disabled {
            background-color: #4B5563;
            cursor: not-allowed;
        }
        .stNumberInput input, .stSelectbox select {
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
        .result-history {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            scroll-behavior: smooth;
            gap: 0.25rem;
            padding: 0.5rem;
            max-width: 100%;
        }
        .result-item {
            min-width: 2rem;
            height: 2rem;
            line-height: 2rem;
            text-align: center;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: bold;
            color: white;
        }
        .result-p {
            background-color: #3B82F6;
        }
        .result-b {
            background-color: #EF4444;
        }
        .result-t {
            background-color: #10B981;
        }
        .strategy-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: #D1D5DB;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    alert_container = st.container()
    with alert_container:
        for alert in st.session_state.alerts[-3:]:
            alert_class = f"alert alert-{alert['type'].lower()}"
            st.markdown(f'<div class="{alert_class}" style="word-wrap: break-word;">{alert["message"]}</div>', unsafe_allow_html=True)
        if st.session_state.alerts:
            if st.button("Clear Alerts"):
                clear_alerts()

    st.markdown('<h1>Baccarat Tracker</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<h2>Controls</h2>', unsafe_allow_html=True)
        
        with st.expander("Money Management", expanded=True):
            st.number_input("Initial Bankroll ($10-$10000)", min_value=10.0, max_value=10000.0, value=st.session_state.initial_bankroll, step=10.0, key="initial_bankroll_input")
            st.number_input("Base Amount ($1-$100)", min_value=1.0, max_value=100.0, value=st.session_state.base_amount, step=1.0, key="base_amount_input")
            st.markdown(f'<p class="text-sm text-gray-400">Profit Lock Threshold: ${st.session_state.profit_lock_threshold:.2f} (2x Base)</p>', unsafe_allow_html=True)
            st.markdown('<p class="strategy-label">Select Betting Strategy</p>', unsafe_allow_html=True)
            strategy_options = ["Flatbet", "T3"]
            st.selectbox("Betting Strategy", strategy_options, key="strategy_select", help="Flatbet: Fixed bet amount. T3: Dynamic bet sizing based on win/loss patterns.")
            st.markdown(f'<p class="text-sm text-gray-400">Current Strategy: {st.session_state.betting_strategy}</p>', unsafe_allow_html=True)
            if st.session_state.betting_strategy == "T3":
                st.markdown(f'<p class="text-sm text-gray-400">T3 Level: {st.session_state.t3_level}, Results: {st.session_state.t3_results}</p>', unsafe_allow_html=True)
            st.button("Apply Money Management", on_click=lambda: [set_money_management(), set_betting_strategy()])

        with st.expander("Session Actions"):
            st.button("Reset Betting", on_click=reset_betting)
            st.button("Reset Session", on_click=reset_all)
            st.button("New Session", on_click=lambda: [reset_all(), st.session_state.alerts.append({"type": "success", "message": "New session started.", "id": str(uuid.uuid4())})])
            st.button("Simulate 100 Games", on_click=simulate_games)

    with st.container():
        st.markdown('<h2>Overview</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            next_bet_class = "card"
            if st.session_state.next_prediction == "Player":
                next_bet_class = "card-player"
            elif st.session_state.next_prediction == "Banker":
                next_bet_class = "card-banker"
                
            st.markdown(f"""
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Bankroll</p>
                    <p class="text-xl font-bold text-white">${st.session_state.result_tracker:.2f}</p>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Session Profit</p>
                    <p class="text-xl font-bold text-white">${st.session_state.session_profit:.2f}</p>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Profit Lock</p>
                    <p class="text-xl font-bold text-green-400">${st.session_state.profit_lock:.2f}</p>
                </div>
                <div class="{next_bet_class}">
                    <p class="text-sm font-semibold text-gray-200">Next Bet</p>
                    <p class="text-xl font-bold text-white">{st.session_state.next_prediction}</p>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Bet Amount</p>
                    <p class="text-xl font-bold text-white">${st.session_state.bet_amount:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<h2>Result History</h2>', unsafe_allow_html=True)
        if st.session_state.results:
            recent_results = list(st.session_state.results)[-20:]
            result_items = [
                f'<span class="result-item result-{r.lower()}">{r}</span>'
                for r in recent_results
            ]
            result_html = "".join(result_items)
            st.markdown(f"""
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Last 20 Results (P: Player, B: Banker, T: Tie)</p>
                    <div class="result-history" id="resultHistory">
                        {result_html}
                    </div>
                </div>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {{
                        const resultDiv = document.getElementById('resultHistory');
                        if (resultDiv) {{
                            resultDiv.scrollLeft = resultDiv.scrollWidth;
                        }}
                    }});
                </script>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<p class="text-gray-400">No results yet.</p>', unsafe_allow_html=True)

        st.markdown('<h2>Record Result</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("Player", on_click=lambda: record_result('P'))
        with col2:
            st.button("Banker", on_click=lambda: record_result('B'))
        with col3:
            st.button("Tie", on_click=lambda: record_result('T'))
        with col4:
            st.button("Undo", on_click=undo)

        st.markdown('<h2>Deal History</h2>', unsafe_allow_html=True)
        if st.session_state.pair_types:
            history_data = [
                {"Pair": f"{pair[0]}{pair[1]}", "Type": "Even" if pair[0] == pair[1] else "Odd"}
                for pair in st.session_state.pair_types
            ]
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=300)
        else:
            st.markdown('<p class="text-gray-400">No history yet.</p>', unsafe_allow_html=True)

        total_games = st.session_state.stats['wins'] + st.session_state.stats['losses']
        win_rate = (st.session_state.stats['wins'] / total_games * 100) if total_games > 0 else 0
        avg_streak = sum(st.session_state.stats['streaks']) / len(st.session_state.stats['streaks']) if st.session_state.stats['streaks'] else 0
        st.markdown(f"""
            <div class="card">
                <p class="text-sm font-semibold text-gray-400">Statistics</p>
                <p class="text-base text-white">Win Rate: {win_rate:.1f}%</p>
                <p class="text-base text-white">Avg Streak: {avg_streak:.1f}</p>
                <p class="text-base text-white">Patterns: Odd: {st.session_state.stats['odd_pairs']}, Even: {st.session_state.stats['even_pairs']}, Alternating: {st.session_state.stats['alternating_pairs']}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2>Bet History</h2>', unsafe_allow_html=True)
        if st.session_state.stats.get('bet_history'):
            bet_history = pd.DataFrame(st.session_state.stats['bet_history'])
            st.dataframe(bet_history, use_container_width=True, height=200)
        else:
            st.markdown('<p class="text-gray-400">No bets placed yet.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
