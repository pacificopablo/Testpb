
import streamlit as st
import random
import pandas as pd
import uuid
import json
from collections import deque

def initialize_session_state():
    """Initialize session state variables if not already set."""
    if 'pair_types' not in st.session_state:
        st.session_state.pair_types = deque(maxlen=100)  # Store up to 100 pairs
        st.session_state.results = deque(maxlen=200)  # Store raw results
        st.session_state.next_prediction = "N/A"
        st.session_state.base_amount = 10.0
        st.session_state.bet_amount = 0.0  # Initialize to 0 since no prediction yet
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
            'alternating_pairs': 0,
            'bet_history': []
        }
        st.session_state.pattern_confidence = {"Odd": 0.0, "Even": 0.0, "Alternating": 0.0, "Streak": 0.0, "Choppy": 0.0}
        st.session_state.alerts = []  # List to store active alerts

def set_base_amount():
    """Set the base amount from user input."""
    try:
        amount = float(st.session_state.base_amount_input)
        if 1 <= amount <= 100:
            st.session_state.base_amount = amount
            st.session_state.bet_amount = st.session_state.base_amount if st.session_state.next_prediction in ["Player", "Banker"] else 0
            st.session_state.alerts.append({"type": "success", "message": "Base amount updated successfully.", "id": str(uuid.uuid4())})
        else:
            st.session_state.alerts.append({"type": "error", "message": "Invalid base amount. Must be between $1 and $100.", "id": str(uuid.uuid4())})
    except ValueError:
        st.session_state.alerts.append({"type": "error", "message": "Please enter a valid number.", "id": str(uuid.uuid4())})

def analyze_patterns():
    """Analyze recent pairs to determine dominant patterns and confidence scores."""
    results = list(st.session_state.results)
    pairs = list(st.session_state.pair_types)
    if len(pairs) < 2:
        st.session_state.bet_amount = 0
        return {"Odd": 0.0, "Even": 0.0, "Alternating": 0.0, "Streak": 0.0, "Choppy": 0.0}, "N/A", "N/A", None

    # Calculate alternation rate for dynamic window sizes
    recent_pairs = pairs[-10:] if len(pairs) >= 10 else pairs
    alternation_rate = sum(1 for i in range(len(recent_pairs)-1) if recent_pairs[i][1] != recent_pairs[i+1][1]) / (len(recent_pairs)-1) if len(recent_pairs) > 1 else 0
    window_sizes = [3, 5, 8] if alternation_rate > 0.7 else [5, 10, 8]  # Shorter windows for choppy shoes

    pattern_scores = {"Odd": 0.0, "Even": 0.0, "Alternating": 0.0, "Streak": 0.0, "Choppy": 0.0}
    total_weight = 0.0

    for window in window_sizes:
        if len(pairs) >= window:
            recent_pairs = pairs[-window:]
            recent_results = results[-window-1:] if len(results) >= window+1 else results

            # Odd and Even pairs
            odd_count = sum(1 for a, b in recent_pairs if a != b)
            even_count = sum(1 for a, b in recent_pairs if a == b)
            total_pairs = odd_count + even_count
            odd_score = (odd_count / total_pairs) * (window / 20) if total_pairs > 0 else 0
            even_score = (even_count / total_pairs) * (window / 20) if total_pairs > 0 else 0
            pattern_scores["Odd"] += odd_score
            pattern_scores["Even"] += even_score
            total_weight += window / 20

            # Alternating pattern (e.g., P-B-P-B)
            alternating_count = sum(1 for i in range(len(recent_pairs)-1) if recent_pairs[i][1] != recent_pairs[i+1][1])
            alternating_score = (alternating_count / (window-1)) * (window / 20) if window > 1 else 0
            pattern_scores["Alternating"] += alternating_score

            # Streak detection (2+ identical results)
            streak_length = 1
            current_streak = recent_results[-1] if recent_results else None
            for i in range(2, len(recent_results)+1):
                if recent_results[-i] == current_streak and recent_results[-i] != 'T':
                    streak_length += 1
                else:
                    break
            streak_score = (streak_length / 5) * (window / 20) if streak_length >= 2 else 0
            pattern_scores["Streak"] += streak_score

            # Choppy pattern (frequent single alternations)
            choppy_count = sum(1 for i in range(len(recent_results)-1) if recent_results[i] != recent_results[i+1] and recent_results[i] != 'T' and recent_results[i+1] != 'T')
            choppy_score = (choppy_count / (window-1)) * (window / 20) if window > 1 else 0
            pattern_scores["Choppy"] += choppy_score

    # Normalize scores
    if total_weight > 0:
        for pattern in pattern_scores:
            pattern_scores[pattern] /= total_weight

    # Determine dominant pattern
    dominant_pattern = max(pattern_scores, key=pattern_scores.get)
    confidence = pattern_scores[dominant_pattern]
    streak_type = None

    # Set prediction based on dominant pattern
    last_result = st.session_state.previous_result
    if confidence < 0.5 or len(pairs) < 8 or dominant_pattern == "Choppy":  # Lower threshold, higher min pairs
        prediction = "Hold"
        dominance = "Choppy" if dominant_pattern == "Choppy" else "N/A"
        st.session_state.bet_amount = 0
        if dominant_pattern == "Choppy":
            st.session_state.alerts.append({"type": "warning", "message": "Choppy shoe detected. Holding bets.", "id": str(uuid.uuid4())})
    elif dominant_pattern == "Odd":
        prediction = "Player" if last_result == 'B' else "Banker"
        dominance = "Odd"
        st.session_state.bet_amount = st.session_state.base_amount
    elif dominant_pattern == "Even":
        prediction = "Player" if last_result == 'P' else "Banker"
        dominance = "Even"
        st.session_state.bet_amount = st.session_state.base_amount
    elif dominant_pattern == "Alternating":
        prediction = "Player" if last_result == 'B' else "Banker"
        dominance = "Alternating"
        st.session_state.bet_amount = st.session_state.base_amount
    else:  # Streak
        last_results = [r for r in results[-4:] if r != 'T']
        if len(last_results) >= 2 and all(r == last_results[-1] for r in last_results[-2:]):
            streak_type = last_results[-1]
            prediction = "Player" if streak_type == 'P' else "Banker"
            dominance = f"Streak ({streak_type})"
            st.session_state.bet_amount = st.session_state.base_amount
        else:
            prediction = "Hold"
            dominance = "N/A"
            st.session_state.bet_amount = 0

    # Adjust for shoe position
    if len(results) < 20:
        prediction = "Hold"
        dominance = "N/A"
        st.session_state.bet_amount = 0

    # Cap bet amount in choppy conditions
    if alternation_rate > 0.7:
        st.session_state.bet_amount = min(st.session_state.bet_amount, 3 * st.session_state.base_amount)

    return pattern_scores, dominance, prediction, streak_type

def reset_betting():
    """Reset betting parameters and update prediction."""
    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.session_state.alerts.append({"type": "warning", "message": "Stop-loss reached. Resetting to resume betting.", "id": str(uuid.uuid4())})
    if st.session_state.result_tracker >= 0:
        st.session_state.result_tracker = 0.0
    st.session_state.consecutive_wins = 0
    st.session_state.consecutive_losses = 0
    st.session_state.streak_type = None

    pattern_scores, dominance, prediction, streak_type = analyze_patterns()
    st.session_state.pattern_confidence = pattern_scores
    st.session_state.current_dominance = dominance
    st.session_state.next_prediction = prediction
    st.session_state.streak_type = streak_type
    st.session_state.bet_amount = 0 if prediction == "Hold" else st.session_state.base_amount
    st.session_state.alerts.append({"type": "success", "message": "Betting reset.", "id": str(uuid.uuid4())})

def reset_all():
    """Reset all session data."""
    st.session_state.pair_types = deque(maxlen=100)
    st.session_state.results = deque(maxlen=200)
    st.session_state.result_tracker = 0.0
    st.session_state.profit_lock = 0.0
    st.session_state.bet_amount = 0.0
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
        'alternating_pairs': 0,
        'bet_history': []
    }
    st.session_state.pattern_confidence = {"Odd": 0.0, "Even": 0.0, "Alternating": 0.0, "Streak": 0.0, "Choppy": 0.0}
    st.session_state.alerts.append({"type": "success", "message": "All session data reset, profit lock reset.", "id": str(uuid.uuid4())})

def record_result(result):
    """Record a game result and update state."""
    current_prediction = st.session_state.next_prediction
    st.session_state.results.append(result)

    # Save current state before modifications
    state = {
        'pair_types': list(st.session_state.pair_types),
        'results': list(st.session_state.results),
        'previous_result': st.session_state.previous_result,
        'result_tracker': st.session_state.result_tracker,
        'profit_lock': st.session_state.profit_lock,
        'bet_amount': st.session_state.bet_amount,
        'current_dominance': st.session_state.current_dominance,
        'next_prediction': st.session_state.next_prediction,
        'consecutive_wins': st.session_state.consecutive_wins,
        'consecutive_losses': st.session_state.consecutive_losses,
        'streak_type': st.session_state.streak_type,
        'stats': st.session_state.stats.copy(),
        'pattern_confidence': st.session_state.pattern_confidence.copy()
    }
    st.session_state.state_history.append(state)

    # Handle Tie
    if result == 'T':
        st.session_state.stats['ties'] += 1
        st.session_state.previous_result = result
        st.session_state.bet_amount = 0  # No bet on Tie
        st.session_state.alerts.append({"type": "info", "message": "Tie recorded. No bet placed.", "id": str(uuid.uuid4())})
        return

    # Handle first result
    if st.session_state.previous_result is None:
        st.session_state.previous_result = result
        st.session_state.next_prediction = "N/A"
        st.session_state.bet_amount = 0
        st.session_state.alerts.append({"type": "info", "message": "Waiting for more results to start betting.", "id": str(uuid.uuid4())})
        return

    # Record pair
    if st.session_state.previous_result != 'T':
        pair = (st.session_state.previous_result, result)
        st.session_state.pair_types.append(pair)
        pair_type = "Even" if pair[0] == pair[1] else "Odd"
        st.session_state.stats['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1
        if len(st.session_state.pair_types) >= 2:
            last_two_pairs = list(st.session_state.pair_types)[-2:]
            if last_two_pairs[0][1] != last_two_pairs[1][1]:
                st.session_state.stats['alternating_pairs'] += 1

    # Define recent_pairs for alternation rate
    recent_pairs = list(st.session_state.pair_types)[-10:] if len(st.session_state.pair_types) >= 10 else list(st.session_state.pair_types)
    alternation_rate = sum(1 for i in range(len(recent_pairs)-1) if recent_pairs[i][1] != recent_pairs[i+1][1]) / (len(recent_pairs)-1) if len(recent_pairs) > 1 else 0

    # Evaluate bet outcome (after 8 pairs for choppy shoes)
    pattern_scores, dominance, _, _ = analyze_patterns()
    min_pairs = 8 if alternation_rate > 0.7 else 5
    if len(st.session_state.pair_types) >= min_pairs and current_prediction != "Hold":
        effective_bet = min(5 * st.session_state.base_amount, st.session_state.bet_amount)
        outcome = ""
        if current_prediction == "Player" and result == 'P':
            st.session_state.result_tracker += effective_bet
            st.session_state.stats['wins'] += 1
            st.session_state.consecutive_wins += 1
            st.session_state.consecutive_losses = 0
            outcome = f"Won ${effective_bet:.2f}"
            st.session_state.alerts.append({"type": "success", "message": f"Bet won! +${effective_bet:.2f}", "id": str(uuid.uuid4())})
            if st.session_state.result_tracker > st.session_state.profit_lock:
                st.session_state.profit_lock = st.session_state.result_tracker
                st.session_state.result_tracker = 0.0
                st.session_state.bet_amount = st.session_state.base_amount
                st.session_state.alerts.append({"type": "info", "message": f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.", "id": str(uuid.uuid4())})
            elif st.session_state.consecutive_wins >= 2:
                st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
        elif current_prediction == "Banker" and result == 'B':
            st.session_state.result_tracker += effective_bet * 0.95
            st.session_state.stats['wins'] += 1
            st.session_state.consecutive_wins += 1
            st.session_state.consecutive_losses = 0
            outcome = f"Won ${effective_bet * 0.95:.2f}"
            st.session_state.alerts.append({"type": "success", "message": f"Bet won! +${effective_bet * 0.95:.2f}", "id": str(uuid.uuid4())})
            if st.session_state.result_tracker > st.session_state.profit_lock:
                st.session_state.profit_lock = st.session_state.result_tracker
                st.session_state.result_tracker = 0.0
                st.session_state.bet_amount = st.session_state.base_amount
                st.session_state.alerts.append({"type": "info", "message": f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.", "id": str(uuid.uuid4())})
            elif st.session_state.consecutive_wins >= 2:
                st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
        else:
            st.session_state.result_tracker -= effective_bet
            st.session_state.stats['losses'] += 1
            st.session_state.consecutive_losses += 1
            st.session_state.consecutive_wins = 0
            outcome = f"Lost ${effective_bet:.2f}"
            st.session_state.alerts.append({"type": "error", "message": f"Bet lost! -${effective_bet:.2f}", "id": str(uuid.uuid4())})
            if st.session_state.current_dominance == "Choppy" and st.session_state.consecutive_losses > 0:
                st.session_state.bet_amount = 0
                st.session_state.next_prediction = "Hold"
                st.session_state.alerts.append({"type": "info", "message": "Pausing bets due to choppy shoe.", "id": str(uuid.uuid4())})
                return
            elif st.session_state.consecutive_losses >= 3:
                st.session_state.bet_amount = min(3 * st.session_state.base_amount, st.session_state.bet_amount * 1.5)  # Softer progression
            elif st.session_state.streak_type:
                st.session_state.bet_amount = min(3 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
            else:
                st.session_state.bet_amount = min(3 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
        st.session_state.stats['bet_history'].append({
            'prediction': current_prediction,
            'result': result,
            'bet_amount': effective_bet,
            'outcome': outcome
        })

    # Check stop-loss
    if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
        st.session_state.alerts.append({"type": "warning", "message": "Loss limit reached. Resetting to resume betting.", "id": str(uuid.uuid4())})
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.next_prediction = "Player" if result == 'B' else "Banker" if result == 'P' else random.choice(["Player", "Banker"])
        return

    # Update prediction for next round
    pattern_scores, dominance, prediction, streak_type = analyze_patterns()
    st.session_state.pattern_confidence = pattern_scores
    st.session_state.current_dominance = dominance
    st.session_state.next_prediction = prediction
    st.session_state.streak_type = streak_type
    st.session_state.previous_result = result

    # Ensure bet_amount is 0 if not enough pairs or prediction is Hold
    min_pairs = 8 if alternation_rate > 0.7 else 5
    if len(st.session_state.pair_types) < min_pairs or prediction == "Hold":
        st.session_state.bet_amount = 0
        if len(st.session_state.pair_types) < min_pairs:
            st.session_state.alerts.append({"type": "info", "message": f"Result recorded. Need {min_pairs - len(st.session_state.pair_types)} more results to start betting.", "id": str(uuid.uuid4())})

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
    st.session_state.profit_lock = last_state['profit_lock']
    st.session_state.bet_amount = last_state['bet_amount']
    st.session_state.current_dominance = last_state['current_dominance']
    st.session_state.next_prediction = last_state['next_prediction']
    st.session_state.consecutive_wins = last_state['consecutive_wins']
    st.session_state.consecutive_losses = last_state['consecutive_losses']
    st.session_state.streak_type = last_state['streak_type']
    st.session_state.stats = last_state['stats']
    st.session_state.pattern_confidence = last_state['pattern_confidence']
    st.session_state.alerts.append({"type": "success", "message": "Last action undone.", "id": str(uuid.uuid4())})

def simulate_games():
    """Simulate 100 games."""
    outcomes = ['P', 'B', 'T']
    weights = [0.446, 0.458, 0.096]
    for _ in range(100):
        result = random.choices(outcomes, weights)[0]
        record_result(result)
    st.session_state.alerts.append({"type": "success", "message": "Simulated 100 games. Check stats and bet history for results.", "id": str(uuid.uuid4())})

def simulate_choppy_games():
    """Simulate 100 games with choppy shoe characteristics."""
    outcomes = ['P', 'B', 'T']
    weights = [0.48, 0.48, 0.04]  # High alternation, low ties
    for _ in range(100):
        if random.random() < 0.8 and st.session_state.previous_result:
            result = 'B' if st.session_state.previous_result == 'P' else 'P'
        else:
            result = random.choices(outcomes, weights)[0]
        record_result(result)
    st.session_state.alerts.append({"type": "success", "message": "Simulated 100 choppy games.", "id": str(uuid.uuid4())})

def clear_alerts():
    """Clear all alerts."""
    st.session_state.alerts = []

def main():
    """Main Streamlit application."""
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
        .next-bet-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: bold;
            text-align: center;
            min-width: 4rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Alert container
    alert_container = st.container()
    with alert_container:
        for alert in st.session_state.alerts[-3:]:
            alert_class = f"alert alert-{alert['type'].lower()}"
            st.markdown(f'<div class="{alert_class}">{alert["message"]}</div>', unsafe_allow_html=True)
        if st.session_state.alerts:
            if st.button("Clear Alerts"):
                clear_alerts()

    # Title
    st.markdown('<h1>Baccarat Predictor</h1>', unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.markdown('<h2>Controls</h2>', unsafe_allow_html=True)
        with st.expander("Bet Settings", expanded=True):
            st.number_input("Base Amount ($1-$100)", min_value=1.0, max_value=100.0, value=st.session_state.base_amount, step=1.0, key="base_amount_input")
            st.button("Set Amount", on_click=set_base_amount)

        with st.expander("Session Actions"):
            st.button("Reset Bet", on_click=reset_betting)
            st.button("Reset Session", on_click=reset_all)
            st.button("New Session", on_click=lambda: [reset_all(), st.session_state.alerts.append({"type": "success", "message": "New session started.", "id": str(uuid.uuid4())})])
            st.button("Simulate 100 Games", on_click=simulate_games)
            st.button("Simulate 100 Choppy Games", on_click=simulate_choppy_games)

    # Main content with card layout
    with st.container():
        st.markdown('<h2>Betting Overview</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
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
        with col2:
            badge_class = (
                "next-bet-badge bg-blue-500" if st.session_state.next_prediction == "Player" else
                "next-bet-badge bg-red-500" if st.session_state.next_prediction == "Banker" else
                "next-bet-badge bg-yellow-500" if st.session_state.next_prediction == "Hold" else
                "next-bet-badge bg-gray-500"
            )
            st.markdown(f"""
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Next Bet</p>
                    <span class="{badge_class} text-white">{st.session_state.next_prediction}</span>
                </div>
                <div class="card">
                    <p class="text-sm font-semibold text-gray-400">Bet Amount</p>
                    <p class="text-xl font-bold text-white">{'No Bet' if st.session_state.bet_amount == 0 else f'${st.session_state.bet_amount:.2f}'}</p>
                </div>
            """, unsafe_allow_html=True)

        # Result History
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

        # Result input buttons
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

        # Deal History
        st.markdown('<h2>Deal History</h2>', unsafe_allow_html=True)
        if st.session_state.pair_types:
            history_data = [
                {"Pair": f"{pair[0]}{pair[1]}", "Type": "Even" if pair[0] == pair[1] else "Odd"}
                for pair in st.session_state.pair_types
            ]
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=300)
        else:
            st.markdown('<p class="text-gray-400">No history yet.</p>', unsafe_allow_html=True)

        # Statistics
        total_games = st.session_state.stats['wins'] + st.session_state.stats['losses']
        win_rate = (st.session_state.stats['wins'] / total_games * 100) if total_games > 0 else 0
        avg_streak = sum(st.session_state.stats['streaks']) / len(st.session_state.stats['streaks']) if st.session_state.stats['streaks'] else 0
        recent_pairs = list(st.session_state.pair_types)[-10:] if len(st.session_state.pair_types) >= 10 else list(st.session_state.pair_types)
        alternation_rate = sum(1 for i in range(len(recent_pairs)-1) if recent_pairs[i][1] != recent_pairs[i+1][1]) / (len(recent_pairs)-1) if len(recent_pairs) > 1 else 0
        st.markdown(f"""
            <div class="card">
                <p class="text-sm font-semibold text-gray-400">Statistics</p>
                <p class="text-base text-white">Win Rate: {win_rate:.1f}%</p>
                <p class="text-base text-white">Avg Streak: {avg_streak:.1f}</p>
                <p class="text-base text-white">Alternation Rate: {alternation_rate:.2f}</p>
                <p class="text-base text-white">Patterns: Odd: {st.session_state.stats['odd_pairs']}, Even: {st.session_state.stats['even_pairs']}, Alternating: {st.session_state.stats['alternating_pairs']}</p>
                <p class="text-base text-white">Streak: {st.session_state.streak_type if st.session_state.streak_type else 'None'}</p>
            </div>
        """, unsafe_allow_html=True)

        # Pattern Confidence Chart
        st.markdown('<h2>Pattern Confidence</h2>', unsafe_allow_html=True)
        chart_config = {
            "type": "line",
            "data": {
                "labels": ["Odd", "Even", "Alternating", "Streak", "Choppy"],
                "datasets": [
                    {
                        "label": "Pattern Confidence",
                        "data": [
                            st.session_state.pattern_confidence.get("Odd", 0),
                            st.session_state.pattern_confidence.get("Even", 0),
                            st.session_state.pattern_confidence.get("Alternating", 0),
                            st.session_state.pattern_confidence.get("Streak", 0),
                            st.session_state.pattern_confidence.get("Choppy", 0)
                        ],
                        "borderColor": "#10B981",
                        "backgroundColor": "rgba(16, 185, 129, 0.2)",
                        "fill": True,
                        "tension": 0.4
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True, "labels": {"color": "#E5E7EB"}},
                    "title": {"display": True, "text": "Pattern Confidence Scores", "color": "#E5E7EB"}
                },
                "scales": {
                    "x": {"ticks": {"color": "#E5E7EB"}, "grid": {"color": "#4B5563"}},
                    "y": {
                        "beginAtZero": True,
                        "max": 1,
                        "ticks": {"color": "#E5E7EB"},
                        "grid": {"color": "#4B5563"}
                    }
                }
            }
        }
        st.markdown(f"""
            <div class="card">
                <canvas id="patternChart"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const ctx = document.getElementById('patternChart').getContext('2d');
                    new Chart(ctx, {json.dumps(chart_config)});
                }});
            </script>
        """, unsafe_allow_html=True)

        # Bet History
        st.markdown('<h2>Bet History</h2>', unsafe_allow_html=True)
        if st.session_state.stats.get('bet_history'):
            bet_history = pd.DataFrame(st.session_state.stats['bet_history'])
            st.dataframe(bet_history, use_container_width=True, height=200)
        else:
            st.markdown('<p class="text-gray-400">No bets placed yet.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
