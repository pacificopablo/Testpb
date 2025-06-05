import streamlit as st
import random
import json

def get_prediction(history, full_history):
    """Generate a bet prediction based on historical patterns."""
    if len(full_history) < 2:
        return "Default: Bet on **Banker** (Confidence: 50.00%)"

    history_list = history

    # Pattern checks using full_history for long patterns
    if len(full_history) >= 7:
        last_seven = full_history[-7:]
        if last_seven == ['P', 'P', 'B', 'B', 'B', 'P', 'P']:
            return f"Bet: **Banker** (Pattern: PPBBBPP, Confidence: 80.00%)"
        if last_seven == ['B', 'B', 'P', 'P', 'P', 'B', 'B']:
            return f"Bet: **Player** (Pattern: BBPPPBB, Confidence: 75.00%)"

    # Pattern checks using history (last 7 outcomes)
    if len(history_list) >= 4:
        pattern = history_list[-4:]
        if pattern == ['P', 'B', 'B', 'P']:
            return f"Bet: **Banker** (Pattern: PBBP, Confidence: 75.75%)"
        if pattern == ['B', 'P', 'P', 'B']:
            return f"Bet: **Player** (Pattern: BPPB, Confidence: 75.00%)"
        if pattern == ['B', 'P', 'B', 'P']:
            return f"Bet: **Banker** (Pattern: BP, Confidence: 70.00%)"
        if pattern == ['P', 'B', 'P', 'B']:
            return f"Bet: **Player** (Pattern: PB, Confidence: 70.00%)"

    if len(history_list) >= 3:
        pattern = history_list[-3:]
        if pattern == ['B', 'B', 'B']:
            return f"Bet: **Banker** (Pattern: BBB, Confidence: 75.00%)"
        elif pattern == ['P', 'P', 'P']:
            return f"Bet: **Player** (Pattern: PPP, Confidence: 75.00%)"

    if len(history_list) >= 2:
        last_two = history_list[-2:]
        if last_two == ['P', 'P']:
            return f"Bet: **Player** (Pattern: PP, Confidence: 65.00%)"
        elif last_two == ['B', 'B']:
            return f"Bet: **Banker** (Pattern: BB, Confidence: 65.00%)"
        else:
            next_bet = 'B' if last_two[1] == 'P' else 'P'
            return f"Bet: **{'Player' if next_bet == 'P' else 'Banker'}** (Pattern: CHOP, Confidence: 65.65%)"

    if len(history_list) >= 1:
        last = history_list[-1]
        return f"Bet: **{'Player' if last == 'P' else 'Banker'}** (Pattern: FOLLOW THE LAST, Confidence: 60.00%)"

    # Fallback: Weighted transitions
    transitions = {'B': {'B': 0.1, 'P': 0.1}, 'P': {'B': 0.1, 'P': 0.1}}
    for i in range(len(full_history) - 1):
        current = full_history[i]
        next_outcome = full_history[i + 1]
        transitions[current][next_outcome] += 1
    for i in range(len(history_list) - 1):
        current = history_list[i]
        next_outcome = history_list[i + 1]
        transitions[current][next_outcome] += 2
    last_outcome = history_list[-1] if history_list else full_history[-1]
    total = transitions[last_outcome]['B'] + transitions[last_outcome]['P']
    prob_b = transitions[last_outcome]['B'] / total
    prob_p = transitions[last_outcome]['P'] / total
    confidence_threshold = 0.05 if len(full_history) > 20 else 0.1
    if prob_b > prob_p + confidence_threshold:
        return f"Bet: **Banker** (Pattern: B, Confidence: {prob_b:.2%})"
    elif prob_p > prob_b + confidence_threshold:
        return f"Bet: **Player** (Pattern: P, Confidence: {prob_p:.2%})"
    return f"Default: Bet on **Banker** (Pattern: Default, Confidence: {max(prob_b, prob_p):.2%})"

def deal_baccarat_hand():
    """Simulate a single baccarat hand using simplified rules. Returns 'B', 'P', or 'T'."""
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0] * 32  # 8 decks, 10/J/Q/K = 0
    random.shuffle(deck)

    def hand_value(cards):
        return sum(cards) % 10

    player_cards = [deck.pop(), deck.pop()]
    banker_cards = [deck.pop(), deck.pop()]
    player_total = hand_value(player_cards)
    banker_total = hand_value(banker_cards)

    if player_total >= 8 or banker_total >= 8:
        if player_total > banker_total:
            return 'P'
        elif banker_total > player_total:
            return 'B'
        return 'T'

    if player_total <= 5:
        player_cards.append(deck.pop())
        player_total = hand_value(player_cards)

    if banker_total <= 5:
        banker_cards.append(deck.pop())
        banker_total = hand_value(banker_cards)

    if player_total > banker_total:
        return 'P'
    elif banker_total > player_total:
        return 'B'
    return 'T'

def render_bead_plate(history):
    """Render bead plate as text: 6 rows, up to 40 columns, filled top-to-bottom, left-to-right."""
    rows, cols = 6, 40
    bead_plate = [[' ' for _ in range(cols)] for _ in range(rows)]
    row, col = 0, 0
    for outcome in history:
        if outcome in ['B', 'P', 'T']:
            bead_plate[row][col] = outcome
            row += 1
            if row >= rows:
                row = 0
                col += 1
                if col >= cols:
                    break
    # Convert to text representation
    lines = []
    for row in bead_plate:
        line = ''.join('🔴' if c == 'B' else '🔵' if c == 'P' else '🟢' if c == 'T' else '⬜' for c in row)
        lines.append(line)
    return '\n'.join(lines)

def simulate_shoe(state):
    """Simulate a full shoe (~70 hands) and return analysis."""
    sim_state = {
        'history': [],
        'full_history': [],
        'bet_history': [],
        'bankroll': state['bankroll'],
        'base_bet': state['base_bet'],
        'system': state['system'],
        'level': state['level'],
        'results': [],
        'rounds_per_level': state['rounds_per_level'],
        'level_direction': state['level_direction'],
        'bets_placed': 0,
        'bets_won': 0,
        'pending_bet': None,
        'pattern_counts': {
            'PPBBBPP': 0, 'BBPPPBB': 0, 'PBBP': 0, 'BPPB': 0, 'BP': 0, 'PB': 0, 'PPP': 0,
            'BBB': 0, 'PP': 0, 'BB': 0, 'CHOP': 0, 'FOLLOW THE LAST': 0, 'B': 0, 'P': 0, 'Default': 0
        },
        'min_bankroll': state['bankroll'],
        'max_bankroll': state['bankroll']
    }

    hand_count = 0
    max_hands = 70

    while hand_count < max_hands:
        outcome = deal_baccarat_hand()
        bet_amount = 0
        bet_selection = None
        bet_outcome = None

        if sim_state['pending_bet']:
            bet_amount, bet_selection = sim_state['pending_bet']
            sim_state['bets_placed'] += 1
            if outcome == bet_selection:
                sim_state['bankroll'] += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                sim_state['bets_won'] += 1
                bet_outcome = 'win'
                sim_state['results'].append('W')
            elif outcome != 'T':
                sim_state['bankroll'] -= bet_amount
                bet_outcome = 'loss'
                sim_state['results'].append('L')

            sim_state['min_bankroll'] = min(sim_state['min_bankroll'], sim_state['bankroll'])
            sim_state['max_bankroll'] = max(sim_state['max_bankroll'], sim_state['bankroll'])

            if sim_state['system'] == 'T5' and len(sim_state['results']) == 3:
                if sim_state['results'] == ['W', 'W', 'W']:
                    sim_state['level'] = max(1, sim_state['level'] - 1)
                    sim_state['results'] = []
                elif sim_state['results'] == ['L', 'L', 'L']:
                    sim_state['level'] += 1
                    sim_state['results'] = []
            elif len(sim_state['results']) == sim_state['rounds_per_level']:
                wins = sim_state['results'].count('W')
                losses = sim_state['results'].count('L')
                if wins > losses:
                    sim_state['level'] += sim_state['level_direction']['wins']
                elif losses > wins:
                    sim_state['level'] += sim_state['level_direction']['losses']
                sim_state['results'] = []

            sim_state['pending_bet'] = None

        sim_state['history'].append(outcome)
        sim_state['history'] = sim_state['history'][-7:]
        sim_state['full_history'].append(outcome)

        if len(sim_state['history']) >= 7:
            prediction = get_prediction(sim_state['history'], sim_state['full_history'])
            pattern = prediction.split("Pattern: ")[1].split(",")[0]
            sim_state['pattern_counts'][pattern] += 1
            bet_selection = 'B' if "**Banker**" in prediction else 'P'
            bet_amount = sim_state['base_bet'] * abs(sim_state['level'])
            if bet_amount <= sim_state['bankroll']:
                sim_state['pending_bet'] = (bet_amount, bet_selection)
            else:
                sim_state['pending_bet'] = None

        sim_state['bet_history'].append((outcome, bet_amount, bet_selection, bet_outcome, sim_state['system'], sim_state['level'], sim_state['results'][:]))

        hand_count += 1

    win_rate = (sim_state['bets_won'] / sim_state['bets_placed'] * 100) if sim_state['bets_placed'] > 0 else 0
    net_profit = sim_state['bankroll'] - state['bankroll']
    bankroll_change = (net_profit / state['bankroll'] * 100) if state['bankroll'] > 0 else 0

    analysis = (
        f"\n=== Shoe Analysis ===\n"
        f"Hands Played: {hand_count}\n"
        f"Bets Placed: {sim_state['bets_placed']}\n"
        f"Bets Won: {sim_state['bets_won']}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Initial Bankroll: ${state['bankroll']:.2f}\n"
        f"Final Bankroll: ${sim_state['bankroll']:.2f}\n"
        f"Net Profit/Loss: ${net_profit:.2f} ({bankroll_change:.2f}%)\n"
        f"Min Bankroll: ${sim_state['min_bankroll']:.2f}\n"
        f"Max Bankroll: ${sim_state['max_bankroll']:.2f}\n"
        f"Final Level: {sim_state['level']}\n"
        f"\nPattern Frequencies:\n" +
        "\n".join(f"{pattern}: {count} times" for pattern, count in sim_state['pattern_counts'].items() if count > 0)
    )
    return analysis

def add_result(state, result):
    """Add a result and update state, ensuring Tie has no bearing on wager."""
    bet_amount = 0
    bet_selection = None
    bet_outcome = None

    if state['pending_bet']:
        bet_amount, bet_selection = state['pending_bet']
        state['bets_placed'] += 1
        if result == bet_selection:
            state['bankroll'] += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            state['bets_won'] += 1
            bet_outcome = 'win'
            state['results'].append('W')
        elif result != 'T':
            state['bankroll'] -= bet_amount
            bet_outcome = 'loss'
            state['results'].append('L')

        if state['system'] == 'T5' and len(state['results']) == 3:
            if state['results'] == ['W', 'W', 'W']:
                state['level'] = max(1, state['level'] - 1)
                state['results'] = []
            elif state['results'] == ['L', 'L', 'L']:
                state['level'] += 1
                state['results'] = []
        elif len(state['results']) == state['rounds_per_level']:
            wins = state['results'].count('W')
            losses = state['results'].count('L')
            if wins > losses:
                state['level'] += state['level_direction']['wins']
            elif losses > wins:
                state['level'] += state['level_direction']['losses']
            state['results'] = []

        state['pending_bet'] = None

    state['history'].append(result)
    state['history'] = state['history'][-7:]
    state['full_history'].append(result)

    if len(state['history']) >= 7:
        prediction = get_prediction(state['history'], state['full_history'])
        bet_selection = 'B' if "**Banker**" in prediction else 'P'
        bet_amount = state['base_bet'] * abs(state['level'])
        if bet_amount > state['bankroll']:
            state['pending_bet'] = None
            state['prediction'] = f"{prediction} | Skip betting (bet ${bet_amount:.2f} exceeds bankroll)."
        else:
            state['pending_bet'] = (bet_amount, bet_selection)
            state['prediction'] = f"Bet ${bet_amount:.2f} on **{bet_selection}** ({state['system']} Level {state['level']}) | {prediction}"
    else:
        state['pending_bet'] = None
        state['prediction'] = f"Need {7 - len(state['history'])} more results for prediction."

    state['bet_history'].append((result, bet_amount, bet_selection, bet_outcome, state['system'], state['level'], state['results'][:]))

def initialize_state():
    """Initialize the application state."""
    return {
        'history': [],
        'full_history': [],
        'bead_plate': [],
        'prediction': "",
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_bankroll': 0.0,
        'system': 'T3',
        'level': 1,
        'results': [],
        'rounds_per_level': 3,
        'level_direction': {'wins': 1, 'losses': -1},
        'bet_history': [],
        'pending_bet': None,
        'bets_placed': 0,
        'bets_won': 0,
        'session_active': False
    }

def check_session_end(state):
    """Check if session should end based on bankroll thresholds."""
    if state['session_active']:
        if state['bankroll'] < state['initial_bankroll'] * 0.8:
            st.error("Session Ended: Bankroll below 80% of initial.")
            state.update(initialize_state())
            st.session_state['session_active'] = False
        elif state['bankroll'] >= state['initial_bankroll'] * 1.5:
            st.success("Session Ended: Bankroll above 150% of initial.")
            state.update(initialize_state())
            st.session_state['session_active'] = False

def main():
    st.title("Baccarat Predictor")

    # Initialize session state
    if 'state' not in st.session_state:
        st.session_state['state'] = initialize_state()
    state = st.session_state['state']

    # Setup section
    st.header("Session Setup")
    col1, col2 = st.columns(2)
    with col1:
        bankroll = st.number_input("Initial Bankroll ($)", min_value=0.0, step=10.0, disabled=state['session_active'])
    with col2:
        base_bet = st.number_input("Base Bet ($)", min_value=0.0, step=1.0, disabled=state['session_active'])

    system = st.radio("Betting System", ['T3', 'T5'], disabled=state['session_active'], index=0 if state['system'] == 'T3' else 1)

    col_start, col_reset = st.columns(2)
    with col_start:
        if st.button("Start Session", disabled=state['session_active']):
            if bankroll <= 0 or base_bet <= 0:
                st.error("Bankroll and base bet must be positive.")
            elif base_bet > bankroll * 0.05:
                st.error("Base bet cannot exceed 5% of bankroll.")
            elif system not in ['T3', 'T5']:
                st.error("Please select T3 or T5.")
            else:
                state.update({
                    'bankroll': bankroll,
                    'base_bet': base_bet,
                    'initial_bankroll': bankroll,
                    'history': [],
                    'full_history': [],
                    'bead_plate': [],
                    'results': [],
                    'bet_history': [],
                    'pending_bet': None,
                    'bets_placed': 0,
                    'bets_won': 0,
                    'system': system,
                    'level': 1,
                    'rounds_per_level': 3 if system == 'T3' else 5,
                    'level_direction': {'wins': 1, 'losses': -1} if system == 'T3' else {'wins': -1, 'losses': 1},
                    'prediction': "",
                    'session_active': True
                })
                st.session_state['state'] = state
                st.rerun()

    with col_reset:
        if st.button("Reset", disabled=not state['session_active']):
            state.update(initialize_state())
            st.session_state['state'] = state
            st.rerun()

    # Game controls
    st.header("Game Controls")
    col_b, col_p, col_t, col_undo, col_analyze = st.columns(5)
    with col_b:
        if st.button("Banker (B)", disabled=not state['session_active']):
            add_result(state, 'B')
            check_session_end(state)
            st.session_state['state'] = state
            st.rerun()
    with col_p:
        if st.button("Player (P)", disabled=not state['session_active']):
            add_result(state, 'P')
            check_session_end(state)
            st.session_state['state'] = state
            st.rerun()
    with col_t:
        if st.button("Tie (T)", disabled=not state['session_active']):
            add_result(state, 'T')
            check_session_end(state)
            st.session_state['state'] = state
            st.rerun()
    with col_undo:
        if st.button("Undo", disabled=not state['session_active'] or not state['history']):
            if state['history']:
                state['history'].pop()
                state['full_history'].pop()
                if state['bet_history']:
                    last_bet = state['bet_history'].pop()
                    result, bet_amount, bet_selection, bet_outcome, system, level, results = last_bet
                    if bet_amount > 0 and bet_outcome:
                        state['bets_placed'] -= 1
                        if bet_outcome == 'win':
                            state['bankroll'] -= bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                            state['bets_won'] -= 1
                        elif bet_outcome == 'loss':
                            state['bankroll'] += bet_amount
                        state['level'] = level
                        state['results'] = results[:]
                state['pending_bet'] = None
                state['prediction'] = ""
                st.session_state['state'] = state
                st.rerun()
    with col_analyze:
        if st.button("Analyze Shoe", disabled=not state['session_active']):
            analysis = simulate_shoe(state)
            st.session_state['analysis'] = analysis
            st.session_state['state'] = state
            st.rerun()

    # Display section
    st.header("Game Status")
    st.markdown(f"**History**: {''.join(state['history']) if state['history'] else 'No results yet'}")
    st.markdown(f"**Bankroll**: ${state['bankroll']:.2f}")
    st.markdown(f"**Base Bet**: ${state['base_bet']:.2f}")
    st.markdown(f"**Session**: {state['bets_placed']} bets, {state['bets_won']} wins")
    st.markdown(f"**{state['system']} Status**: Level {state['level']}, Results: {state['results']}")
    st.markdown(f"**Prediction**: {state['prediction']}")

    # Bead plate
    st.header("Bead Plate")
    st.text(render_bead_plate(state['full_history']))

    # Analysis display
    if 'analysis' in st.session_state:
        st.header("Shoe Analysis")
        st.text(st.session_state['analysis'])

if __name__ == "__main__":
    main()
