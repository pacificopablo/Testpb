import streamlit as st

def get_prediction(history):
    last5 = history[-5:]
    if len(last5) < 3:
        return "Default: Bet Banker"

    count = {'B': last5.count('B'), 'P': last5.count('P')}

    if count['B'] >= 3:
        return "Banker (Bias)"
    if count['P'] >= 3:
        return "Player (Bias)"
    
    last3 = last5[-3:]
    if ''.join(last5) in ("BPBPB", "PBPBP"):
        return f"Zigzag Breaker → Bet {last5[-1]}"
    if last3 == [last3[0]] * 3:
        return f"Dragon Slayer → Bet {'Player' if last3[0] == 'B' else 'Banker'}"
    
    second_last = last5[-2]
    return f"OTB4L → Bet {'Player' if second_last == 'B' else 'Banker'}"

def add_result(result):
    state = st.session_state
    bet_amount = 0
    bet_selection = None
    bet_outcome = None

    if state.pending_bet:
        bet_amount, bet_selection = state.pending_bet
        state.bets_placed += 1
        if result == bet_selection:
            state.bankroll += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            state.bets_won += 1
            bet_outcome = 'win'
            if not state.t3_results:
                state.t3_level = max(1, state.t3_level - 1)
            state.t3_results.append('W')
        else:
            state.bankroll -= bet_amount
            bet_outcome = 'loss'
            state.t3_results.append('L')

        if len(state.t3_results) == 3:
            wins = state.t3_results.count('W')
            state.t3_level += -1 if wins > 1 else 1 if state.t3_results.count('L') > 1 else 0
            state.t3_results = []
        state.pending_bet = None

    state.history.append(result)

    if len(state.history) >= 5:
        prediction = get_prediction(state.history)
        bet_selection = 'B' if "Banker" in prediction else 'P' if "Player" in prediction else None
        if bet_selection:
            bet_amount = state.base_bet * state.t3_level
            if bet_amount <= state.bankroll:
                state.pending_bet = (bet_amount, bet_selection)
                state.prediction = f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {state.t3_level})"
            else:
                state.pending_bet = None
                state.prediction = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)."
        else:
            state.pending_bet = None
            state.prediction = "No valid bet selection."
    else:
        state.pending_bet = None
        state.prediction = f"Need {5 - len(state.history)} more results for prediction."

    state.bet_history.append((result, bet_amount, bet_selection, bet_outcome, state.t3_level, state.t3_results[:]))

def main():
    st.title("Baccarat Predictor with T3")

    state = st.session_state
    if 'history' not in state:
        state.history = []
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
                'history': [],
                'bet_history': [],
                'pending_bet': None,
                'bets_placed': 0,
                'bets_won': 0,
                't3_level': 1,
                't3_results': [],
                'prediction': "",
                'session_active': True
            })
            st.success(f"Session started with Bankroll: ${bankroll_input:.2f}, Base Bet: ${base_bet_input:.2f}")

    if st.button("Reset Session"):
        state.update({
            'bankroll': 0.0,
            'base_bet': 0.0,
            'initial_bankroll': 0.0,
            'history': [],
            'bet_history': [],
            'pending_bet': None,
            'bets_placed': 0,
            'bets_won': 0,
            't3_level': 1,
            't3_results': [],
            'prediction': "",
            'session_active': False
        })
        st.success("Session reset. Enter new bankroll and bet to start.")

    if state.session_active:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Banker (B)"):
                if state.bankroll <= state.initial_bankroll * 0.8:
                    st.error("Bankroll below 80% of initial. Session ended.")
                    state.session_active = False
                elif state.bankroll >= state.initial_bankroll * 1.5:
                    st.success("Bankroll above 150% of initial. Session ended.")
                    state.session_active = False
                else:
                    add_result('B')
        with col2:
            if st.button("Player (P)"):
                if state.bankroll <= state.initial_bankroll * 0.8:
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

    # Display statuses
    st.markdown(f"""
**Current History:** {"".join(state.history[-5:]) if state.history else "No results yet"}  
**Bankroll:** ${state.bankroll:.2f}  
**Base Bet:** ${state.base_bet:.2f}  
**Session:** {state.bets_placed} bets, {state.bets_won} wins  
**T3 Status:** Level {state.t3_level}, Results: {state.t3_results}  
**Prediction:** {state.prediction}
    """)

    with st.expander("Debug: Session State"):
        st.write({
            'History': state.history,
            'Prediction': state.prediction,
            'Bankroll': state.bankroll,
            'Base Bet': state.base_bet,
            'T3 Level': state.t3_level,
            'T3 Results': state.t3_results,
            'Bet History': state.bet_history,
            'Pending Bet': state.pending_bet
        })

if __name__ == "__main__":
    main()
