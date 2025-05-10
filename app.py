def place_result(result):
    if st.session_state.target_hit:
        reset_session_auto()
        return
    st.session_state.last_was_tie = (result == 'T')
    bet_amount = 0
    if st.session_state.pending_bet and result != 'T':
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        old_bankroll = st.session_state.bankroll
        if win:
            if selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:
                st.session_state.bankroll += bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('W')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins += 1
                if st.session_state.parlay_wins == 2:
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = 1
                    st.session_state.parlay_wins = 0
                    st.session_state.parlay_using_base = True
                    if old_step != st.session_state.parlay_step:
                        st.session_state.parlay_step_changes += 1
                else:
                    st.session_state.parlay_using_base = False
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
        else:
            st.session_state.bankroll -= bet_amount
            if st.session_state.strategy == 'T3':
                st.session_state.t3_results.append('L')
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_wins = 0
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
            st.session_state.losses += 1
            st.session_state.consecutive_losses += 1
            st.session_state.loss_log.append({
                'sequence': st.session_state.sequence[-10:],
                'prediction': selection,
                'result': result,
                'confidence': st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
            })
            if len(st.session_state.loss_log) > 50:
                st.session_state.loss_log = st.session_state.loss_log[-50:]
        st.session_state.prediction_accuracy['total'] += 1  # Corrected typo here
        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level if st.session_state.strategy == 'T3' else 1,
            "T3_Results": st.session_state.t3_results.copy() if st.session_state.strategy == 'T3' else [],
            "Parlay_Step": st.session_state.parlay_step if st.session_state.strategy == 'Parlay16' else 1
        })
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]
        st.session_state.pending_bet = None
    if not st.session_state.pending_bet and result != 'T':
        st.session_state.consecutive_losses = 0
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]
    if check_target_hit():
        st.session_state.target_hit = True
        return
    pred, conf = predict_next()
    if conf < 50.5:
        st.session_state.pending_bet = None
        st.session_state.advice = f"No bet (Confidence: {conf:.1f}% < 50.5%)"
    else:
        if st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
        elif st.session_state.strategy == 'Parlay16':
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = (st.session_state.base_bet / 10) * PARLAY_TABLE[st.session_state.parlay_step][key]
            if bet_amount > st.session_state.bankroll:
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
                bet_amount = (st.session_state.base_bet / 10) * PARLAY_TABLE[st.session_state.parlay_step]['base']
        if bet_amount > st.session_state.bankroll:
            st.session_state.pending_bet = None
            st.session_state.advice = "No bet: Insufficient bankroll."
            if st.session_state.strategy == 'Parlay16':
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
        else:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"
    if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        old_level = st.session_state.t3_level
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = st.session_state.t3_level + 1
        elif losses == 3:
            st.session_state.t3_level = st.session_state.t3_level + 2
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_results = []
