
import streamlit as st
from collections import deque

def get_prediction(history, bet_history):
    if len(history) < 3:
        return "Default: Bet on Banker"

    # Initialize scores for Banker and Player
    scores = {'B': 0.0, 'P': 0.0}
    weights = {'bias': 0.4, 'streak': 0.5, 'alternation': 0.3}  # Initial pattern weights

    # Adjust weights based on recent bet success
    recent_bets = bet_history[-3:] if len(bet_history) >= 3 else bet_history
    correct_preds = sum(1 for _, _, sel, outcome, _, _ in recent_bets if outcome == 'win' and sel)
    if recent_bets:
        success_rate = correct_preds / len(recent_bets)
        weights['bias'] *= (1.0 + 0.1 * success_rate)
        weights['streak'] *= (1.0 + 0.15 * success_rate)
        weights['alternation'] *= (1.0 + 0.05 * success_rate)

    # Pattern: Bias (frequent outcomes)
    count = {'B': history.count('B'), 'P': history.count('P')}
    if count['B'] >= 3:
        scores['B'] += weights['bias'] * (count['B'] / 5)
        if count['P'] >= 3:
            scores['P'] += weights['bias'] * (count['P'] / 5)

    # Pattern: Streak (three consecutive identical outcomes)
    last3 = list(history)[-3:]
    if last3 == [last3[0]] * 3:
        target = 'P' if last3[0] == 'B' else 'B'
        scores[target] += weights['streak']

    # Pattern: Alternation (BPBPB or PBPBP)
    if ''.join(history) in ("BPBPB", "PBPBP""):
        scores[list(history)[-1]] += weights['alternation']

    # Pattern: OTB4L (bet opposite of second-to-last bet)
    second_last = list(history)[-2]
    scores['P' if second_last == 'B' else 'B'] += weights['alternation'] * 0.8

    # Normalize to probabilities
    total = scores['B'] + scores['P']
    if total == 0:
        return "Default: Bet Banker"
    prob_b = scores['B'] / total
    prob_p = scores['P'] / total

    # Select bet with confidence threshold
    if prob_b > prob_p + 0.1:
        return f"AI Bet: Banker (Confidence: {prob_b:.2%})"
    elif prob_p > prob_b + 0.1:
        return f"AI Bet: Player (Confidence: {prob_p:.2%})"
    return "Default: Bet Banker"

def add_result(result):
    state = st.session_state
    bet_amount = 0
    bet_selection = None
    bet_outcome = None

    # Process pending bet outcome
    if state.pending_bet:
        bet_amount, bet_selection = state.pending_bet
        state.bets_placed += 1
        if result == bet_selection:
            state.bankroll += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            state.bets_won += 1
            bet_outcome = 'win'
            state.t3_results.append('W')
        else:
            state.bankroll -= bet_amount
            bet_outcome = 'loss'
            state.t3_results.append('L')

        # Evaluate T3 level after three rounds
        if len(state.t3_results) == 3:
            wins = state.t3_results.count('W')
            losses = state.t3_results.count('L')
            if wins > losses:
                state.t3_level += 1  # More wins: move forward
            elif losses > wins:
                state.t3_level -= 1  # More losses: move back
            state.t3_results = []  # Reset for next three rounds
        state.pending_bet = None

    state.history.append(result)

    # Generate new prediction and bet
    if len(state.history) >= 5:
        prediction = get_prediction(state.history, state.bet_history)
        bet_selection = 'B' if "Banker" in prediction else 'P' if "Player" in prediction else None
        if bet_selection:
            bet_amount = state.base_bet * abs(state.t3_level)  # Use absolute value for wager
            if bet_amount > state.bankroll:
                state.pending_bet = None
                state.prediction = f"{prediction} | Skip betting (bet ${bet_amount:.2f} exceeds bankroll)."
            else:
                state.pending_bet = (bet_amount, bet_selection)
                state.prediction = f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {state.t3_level}) | {prediction}"
        else:
            state.pending_bet = None
            state.prediction = "No valid bet selection."
    else:
        state.pending_bet = None
        state.prediction = f"Need {5 - len(state.history)} more results for prediction."

    state.bet_history.append((result, bet_amount, bet_selection, bet_outcome, state.t3_level, state.t3_results[:]))

def main():
    st.title("Baccarat Predictor with AI-Powered T3")

    state = st.session_state
    if 'history' not in state:
        state.history = deque(maxlen=5)
        state.prediction = ""
        state.bankroll = 0.0
        state.base_bet = 0.0
        state.initial_bankroll = 0.0
        state.t3_level = 1
        state.t3_results = []
        state.bet_history = []
        state.pending_bet = None
        state.bets_placed = 0
        state.bets_won = 0
        state.session_active = False

    st.markdown("**Session Setup**")
    bankroll_input = st.number_input("Enter Initial Bankroll ($):", min_value=0.0, step=10.0, key="bankroll_input")
    base_bet_input = st.number_input("Enter Base Bet ($):", min_value=0.0, step=1.0, key="base_bet_input")
    
    if st.button("Start Session"):
        if bankroll_input <= 0 or base_bet_input <= 0:
            st.error("Bankroll and base bet must be positive numbers.")
        elif base_bet_input > bankroll_input * 0.05:
            st.error("Base bet cannot exceed 5% of bankroll.")
        else:
            state.update({
                'bankroll': bankroll_input,
                'base_bet': base_bet_input,
                'initial_bankroll': bankroll_input,
                'history': deque(maxlen=5),
                't3_results': [],
                'bet_history': [],
                'pending_bet': None,
                'bets_placed': 0,
                'bets_won': 0,
                't3_level': 1,
                'prediction': "",
                'session_active': True
            })
            st.success(f"Session started with Bankroll: ${bankroll_input:.2f}, Base Bet: ${base_bet_input:.2f}")

    if st.button("Reset Session"):
        state.update({
            'bankroll': 0.0,
            'base_bet': 0.0,
            'initial_bankroll': 0.0,
            'history': deque(maxlen=5),
            't3_results': [],
            'bet_history': [],
            'pending_bet': None,
            'bets_placed': 0,
            'bets_won': 0,
            't3_level': 1,
            'prediction': "",
            'session_active': False
        })
        st.success("Session reset. Enter new bankroll and bet to start.")

    if state.session_active:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Banker (B)"):
                if state.bankroll < state.initial_bankroll * 0.8:
                    st.error("Bankroll below 80% of initial. Session ended.")
                    state.session_active = False
                elif state.bankroll >= state.initial_bankroll * 1.5:
                    st.success("Bankroll above 150% of initial. Session ended.")
                    state.session_active = False
                else:
                    add_result('B')
        with col2:
            if st.button("Player (P)"):
                if state.bankroll < state.initial_bankroll * 0.8:
                    st.error("Bankroll below 80% of initial. Session ended.")
                    state.session_active = False
                elif state.bankroll >= state.initial_bankroll * 1.5:
                    st.success("Bankroll above 150% of initial. Session ended.")
                    state.session_active = False
                else:
                    add_result('P')

    if st.button("Undo Last Result"):
        if not state.history:
            st.warning("No results to undo.")
        else:
            state.history.pop()
            if state.bet_history:
                last_bet = state.bet_history.pop()
                result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results = last_bet
                if bet_amount > 0:
                    state.bets_placed -= 1
                    if bet_outcome == 'win':
                        state.bankroll -= bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                        state.bets_won -= 1
                    elif bet_outcome == 'loss':
                        state.bankroll += bet_amount
                    state.t3_level = t3_level
                    state.t3_results = t3_results[:]
            state.pending_bet = None
            state.prediction = ""
            state.session_active = True

    st.markdown(f"""
**Current History:** {"".join(state.history) if state.history else "No results yet"}  
**Bankroll:** ${state.bankroll:.2f}  
**Base Bet:** ${state.base_bet:.2f}  
**Session:** {state.bets_placed} bets, {state.bets_won} wins  
**T3 Status:** Level {state.t3_level}, Results: {state.t3_results}  
**Prediction:** {state.prediction}
    """)

    with st.expander("Debug Statements"):
        st.write({
            "History": list(state.history),
            "Prediction": state.prediction,
            "Bankroll": state.bankroll,
            "Base Bet": state.base_bet,
            "T3 Level": state.t3_level,
            "T3 Results": state.t3_results,
            "Transactions": state.bet_history,
            "Pending Transaction": state.pending_bet
        })

if __name__ == "__main__":
    main()
