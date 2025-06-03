import streamlit as st

def get_prediction(history):
    last5 = history[-5:]
    count = {'B': 0, 'P': 0}
    for r in last5:
        if r in ['B', 'P']:
            count[r] += 1
    
    if count['B'] >= 3:
        return "Banker (Bias)"
    if count['P'] >= 3:
        return "Player (Bias)"
    if ''.join(last5) == "BPBPB" or ''.join(last5) == "PBPBP":
        return f"Zigzag Breaker → Bet {last5[-1]}"
    if len(last5) >= 3 and all(v == last5[-1] for v in last5[-3:]):
        return f"Dragon Slayer → Bet {'Player' if last5[-1] == 'B' else 'Banker'}"
    if len(last5) >= 3:
        second_last = last5[-2]
        return f"OTB4L → Bet {'Player' if second_last == 'B' else 'Banker'}"
    
    return "Default: Bet Banker"

def add_result(result):
    # Resolve pending bet if exists
    bet_amount = 0
    bet_selection = None
    bet_outcome = None
    if st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        st.session_state.bets_placed += 1
        if result == bet_selection:
            if bet_selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95  # Banker pays 0.95:1
            else:  # Player
                st.session_state.bankroll += bet_amount
            st.session_state.bets_won += 1
            bet_outcome = 'win'
            # T3: Decrease level by 1 on first-step win, minimum 1
            if len(st.session_state.t3_results) == 0:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
        else:
            st.session_state.bankroll -= bet_amount
            bet_outcome = 'loss'
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
        st.session_state.pending_bet = None

    # Add new result
    st.session_state.history.append(result)

    # Make prediction and apply T3
    if len(st.session_state.history) >= 5:  # Minimum for get_prediction
        prediction = get_prediction(st.session_state.history)
        # Extract bet selection from prediction
        bet_selection = None
        if "Banker" in prediction:
            bet_selection = 'B'
        elif "Player" in prediction:
            bet_selection = 'P'
        
        if bet_selection:
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, bet_selection)
                st.session_state.prediction = f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {st.session_state.t3_level})"
            else:
                st.session_state.pending_bet = None
                st.session_state.prediction = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)."
        else:
            st.session_state.pending_bet = None
            st.session_state.prediction = "No valid bet selection."
    else:
        st.session_state.pending_bet = None
        st.session_state.prediction = f"Need {5 - len(st.session_state.history)} more results for prediction."

    # Store bet history with T3 state
    st.session_state.bet_history.append((result, bet_amount, bet_selection, bet_outcome, st.session_state.t3_level, st.session_state.t3_results[:]))

def main():
    st.title("Baccarat Predictor with T3")

    # Initialize session state
    if 'history' not in st.session_state or not isinstance(st.session_state.history, list):
        st.session_state.history = []
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ""
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 0.0
    if 'base_bet' not in st.session_state:
        st.session_state.base_bet = 0.0
    if 'initial_bankroll' not in st.session_state:
        st.session_state.initial_bankroll = 0.0
    if 't3_level' not in st.session_state:
        st.session_state.t3_level = 1
    if 't3_results' not in st.session_state:
        st.session_state.t3_results = []
    if 'bet_history' not in st.session_state:
        st.session_state.bet_history = []  # Stores (result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results)
    if 'pending_bet' not in st.session_state:
        st.session_state.pending_bet = None  # Stores (bet_amount, bet_selection)
    if 'bets_placed' not in st.session_state:
        st.session_state.bets_placed = 0
    if 'bets_won' not in st.session_state:
        st.session_state.bets_won = 0
    if 'session_active' not in st.session_state:
        st.session_state.session_active = False

    # Input for bankroll and base bet
    st.markdown("**Session Setup**")
    bankroll_input = st.number_input("Enter Initial Bankroll ($):", min_value=0.0, step=10.0, key="bankroll_input")
    base_bet_input = st.number_input("Enter Base Bet ($):", min_value=0.0, step=1.0, key="base_bet_input")
    
    if st.button("Start Session"):
        if bankroll_input <= 0 or base_bet_input <= 0:
            st.error("Bankroll and base bet must be positive numbers.")
        elif base_bet_input > bankroll_input * 0.05:
            st.error("Base bet cannot exceed 5% of bankroll.")
        else:
            st.session_state.bankroll = bankroll_input
            st.session_state.base_bet = base_bet_input
            st.session_state.initial_bankroll = bankroll_input
            st.session_state.history = []
            st.session_state.bet_history = []
            st.session_state.pending_bet = None
            st.session_state.bets_placed = 0
            st.session_state.bets_won = 0
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.prediction = ""
            st.session_state.session_active = True
            st.success(f"Session started with Bankroll: ${bankroll_input:.2f}, Base Bet: ${base_bet_input:.2f}")

    if st.button("Reset Session"):
        st.session_state.bankroll = 0.0
        st.session_state.base_bet = 0.0
        st.session_state.initial_bankroll = 0.0
        st.session_state.history = []
        st.session_state.bet_history = []
        st.session_state.pending_bet = None
        st.session_state.bets_placed = 0
        st.session_state.bets_won = 0
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.prediction = ""
        st.session_state.session_active = False
        st.success("Session reset. Enter new bankroll and bet to start.")

    # Create two columns for result buttons
    if st.session_state.session_active:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Banker (B)"):
                if st.session_state.bankroll <= st.session_state.initial_bankroll * 0.8:
                    st.error("Bankroll below 80% of initial. Session ended. Reset or exit.")
                    st.session_state.session_active = False
                elif st.session_state.bankroll >= st.session_state.initial_bankroll * 1.5:
                    st.success("Bankroll above 150% of initial. Session ended. Reset or exit.")
                    st.session_state.session_active = False
                else:
                    add_result('B')
        with col2:
            if st.button("Player (P)"):
                if st.session_state.bankroll <= st.session_state.initial_bankroll * 0.8:
                    st.error("Bankroll below 80% of initial. Session ended. Reset or exit.")
                    st.session_state.session_active = False
                elif st.session_state.bankroll >= st.session_state.initial_bankroll * 1.5:
                    st.success("Bankroll above 150% of initial. Session ended. Reset or exit.")
                    st.session_state.session_active = False
                else:
                    add_result('P')

    # Undo button
    if st.button("Undo Last Result"):
        if not st.session_state.history:
            st.warning("No results to undo.")
        else:
            last_result = st.session_state.history.pop()
            if st.session_state.bet_history:
                last_bet = st.session_state.bet_history.pop()
                result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results = last_bet
                if bet_amount > 0:
                    st.session_state.bets_placed -= 1
                    if bet_outcome == 'win':
                        if bet_selection == 'B':
                            st.session_state.bankroll -= bet_amount * 0.95
                        else:  # Player
                            st.session_state.bankroll -= bet_amount
                        st.session_state.bets_won -= 1
                    elif bet_outcome == 'loss':
                        st.session_state.bankroll += bet_amount
                    # Restore T3 state
                    st.session_state.t3_level = t3_level
                    st.session_state.t3_results = t3_results[:]
            if st.session_state.pending_bet and len(st.session_state.history) >= 4:
                st.session_state.pending_bet = None
            st.session_state.prediction = ""
            st.session_state.session_active = True

    # Display current history after button actions
    st.markdown("**Current History:** " + ("".join(st.session_state.history) if st.session_state.history else "No results yet"))

    # Display statuses
    st.markdown(f"**Bankroll:** ${st.session_state.bankroll:.2f}")
    st.markdown(f"**Base Bet:** ${st.session_state.base_bet:.2f}")
    st.markdown(f"**Session:** {st.session_state.bets_placed} bets, {st.session_state.bets_won} wins")
    st.markdown(f"**T3 Status:** Level {st.session_state.t3_level}, Results: {st.session_state.t3_results}")
    if st.session_state.prediction:
        st.markdown(f"**Prediction:** {st.session_state.prediction}")

    # Debug section
    with st.expander("Debug: Session State"):
        st.write("History:", st.session_state.history)
        st.write("Prediction:", st.session_state.prediction)
        st.write("Bankroll:", st.session_state.bankroll)
        st.write("Base Bet:", st.session_state.base_bet)
        st.write("T3 Level:", st.session_state.t3_level)
        st.write("T3 Results:", st.session_state.t3_results)
        st.write("Bet History:", st.session_state.bet_history)
        st.write("Pending Bet:", st.session_state.pending_bet)

if __name__ == "__main__":
    main()
