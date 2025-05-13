# --- Betting Logic ---
def place_result(result: str):
    """Process a game result with error handling."""
    logging.debug("Entering place_result")
    try:
        if st.session_state.target_hit:
            reset_session()
            return

        st.session_state.last_was_tie = (result == 'T')
        bet_amount = 0
        bet_placed = False
        selection = None
        win = False

        previous_state = {
            "bankroll": st.session_state.bankroll,
            "t3_level": st.session_state.t3_level,
            "t3_results": st.session_state.t3_results.copy(),
            "parlay_step": st.session_state.parlay_step,
            "parlay_wins": st.session_state.parlay_wins,
            "parlay_using_base": st.session_state.parlay_using_base,
            "z1003_loss_count": st.session_state.z1003_loss_count,
            "z1003_bet_factor": st.session_state.z1003_bet_factor,
            "z1003_continue": st.session_state.z1003_continue,
            "z1003_level_changes": st.session_state.z1003_level_changes,
            "pending_bet": st.session_state.pending_bet,
            "wins": st.session_state.wins,
            "losses": st.session_state.losses,
            "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
            "consecutive_losses": st.session_state.consecutive_losses,
            "t3_level_changes": st.session_state.t3_level_changes,
            "parlay_step_changes": st.session_state.parlay_step_changes,
            "pattern_volatility": st.session_state.pattern_volatility,
            "pattern_success": st.session_state.pattern_success.copy(),
            "pattern_attempts": st.session_state.pattern_attempts.copy(),
            "safety_net_percentage": st.session_state.safety_net_percentage,
            "consecutive_wins": st.session_state.consecutive_wins,
            "last_win_confidence": st.session_state.last_win_confidence,
            "insights": st.session_state.insights.copy(),
        }

        if st.session_state.pending_bet and result != 'T':
            bet_amount, selection = st.session_state.pending_bet
            win = result == selection
            bet_placed = True
            if win:
                st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                st.session_state.wins += 1
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                st.session_state.last_win_confidence = predict_next()[1]
                logging.debug(f"Win recorded: Total wins={st.session_state.wins}, Consecutive wins={st.session_state.consecutive_wins}")
                if st.session_state.consecutive_wins >= 3:
                    st.session_state.base_bet *= 1.05
                    st.session_state.base_bet = round(st.session_state.base_bet, 2)
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
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                    else:
                        st.session_state.parlay_using_base = False
                elif st.session_state.strategy == 'Z1003.1':
                    _, conf, _ = predict_next()
                    if conf > 50.0 and st.session_state.pattern_volatility < 0.4:
                        st.session_state.z1003_continue = True
                    else:
                        st.session_state.z1003_loss_count = 0
                        st.session_state.z1003_continue = False
                st.session_state.prediction_accuracy[selection] += 1
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_success[pattern] += 1
                        st.session_state.pattern_attempts[pattern] += 1
            else:
                st.session_state.bankroll -= bet_amount
                st.session_state.losses += 1
                st.session_state.consecutive_wins = 0
                st.session_state.consecutive_losses += 1
                logging.debug(f"Loss recorded: Total losses={st.session_state.losses}, Consecutive losses={st.session_state.consecutive_losses}")
                _, conf, _ = predict_next()
                st.session_state.loss_log.append({
                    'sequence': st.session_state.sequence[-10:],
                    'prediction': selection,
                    'result': result,
                    'confidence': f"{conf:.1f}",
                    'insights': st.session_state.insights.copy()
                })
                if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                    st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_attempts[pattern] += 1
            st.session_state.prediction_accuracy['total'] += 1
            st.session_state.pending_bet = None

        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) > SEQUENCE_LIMIT:
            st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LIMIT:]

        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level,
            "Parlay_Step": st.session_state.parlay_step,
            "Z1003_Loss_Count": st.session_state.z1003_loss_count,
            "Previous_State": previous_state,
            "Bet_Placed": bet_placed,
            "Consecutive_Wins": st.session_state.consecutive_wins,
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]

        if check_target_hit():
            st.session_state.target_hit = True
            return

        pred, conf, insights = predict_next()
        if st.session_state.strategy == 'Z1003.1' and st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            bet_amount, advice = None, "No bet: Stopped after three losses (Z1003.1 rule)"
        else:
            bet_amount, advice = calculate_bet_amount(pred, conf)
        st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
        st.session_state.advice = advice
        st.session_state.insights = insights

        if st.session_state.strategy == 'T3':
            update_t3_level()

        # Validate win/loss counts
        if st.session_state.wins < 0 or st.session_state.losses < 0:
            logging.error(f"Invalid win/loss counts: wins={st.session_state.wins}, losses={st.session_state.losses}")
            st.session_state.wins = max(0, st.session_state.wins)
            st.session_state.losses = max(0, st.session_state.losses)

        logging.debug("place_result completed")
    except Exception as e:
        logging.error(f"place_result error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error processing result. Try resetting the session.")

# --- UI Components ---
def render_result_input():
    """Render the result input buttons."""
    logging.debug("Entering render_result_input")
    try:
        st.subheader("Enter Result")
        st.markdown("""
        <style>
        div.stButton > button {
            width: 90px; height: 35px; font-size: 14px; font-weight: bold; border-radius: 6px; border: 1px solid;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); cursor: pointer; transition: all 0.15s ease;
            display: flex; align-items: center; justify-content: center;
        }
        div.stButton > button:hover { transform: scale(1.08); box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3); }
        div.stButton > button:active { transform: scale(0.95); box-shadow: none; }
        div.stButton > button[kind="player_btn"] { background: linear-gradient(to bottom, #007bff, #0056b3); border-color: #0056b3; color: white; }
        div.stButton > button[kind="player_btn"]:hover { background: linear-gradient(to bottom, #339cff, #007bff); }
        div.stButton > button[kind="banker_btn"] { background: linear-gradient(to bottom, #dc3545, #a71d2a); border-color: #a71d2a; color: white; }
        div.stButton > button[kind="banker_btn"]:hover { background: linear-gradient(to bottom, #ff6666, #dc3545); }
        div.stButton > button[kind="tie_btn"] { background: linear-gradient(to bottom, #28a745, #1e7e34); border-color: #1e7e34; color: white; }
        div.stButton > button[kind="tie_btn"]:hover { background: linear-gradient(to bottom, #4caf50, #28a745); }
        div.stButton > button[kind="undo_btn"] { background: linear-gradient(to bottom, #6c757d, #545b62); border-color: #545b62; color: white; }
        div.stButton > button[kind="undo_btn"]:hover { background: linear-gradient(to bottom, #8e959c, #6c757d); }
        @media (max-width: 600px) { div.stButton > button { width: 80%; max-width: 150px; height: 40px; font-size: 12px; } }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Player", key="player_btn"):
                place_result("P")
        with col2:
            if st.button("Banker", key="banker_btn"):
                place_result("B")
        with col3:
            if st.button("Tie", key="tie_btn"):
                place_result("T")
        with col4:
            if st.button("Undo Last", key="undo_btn"):
                try:
                    if not st.session_state.sequence:
                        st.warning("No results to undo.")
                    else:
                        if st.session_state.history:
                            last = st.session_state.history.pop()
                            previous_state = last['Previous_State']
                            for key, value in previous_state.items():
                                st.session_state[key] = value
                            st.session_state.sequence.pop()
                            if last['Bet_Placed'] and not last['Win'] and st.session_state.loss_log:
                                if st.session_state.loss_log[-1]['result'] == last['Result']:
                                    st.session_state.loss_log.pop()
                            if last['Bet_Placed']:
                                if last['Win']:
                                    logging.debug(f"Undo win: Reducing wins from {st.session_state.wins} to {st.session_state.wins - 1}")
                                else:
                                    logging.debug(f"Undo loss: Reducing losses from {st.session_state.losses} to {st.session_state.losses - 1}")
                            if st.session_state.pending_bet:
                                amount, pred = st.session_state.pending_bet
                                conf = predict_next()[1]
                                st.session_state.advice = f"Next Bet: ${amount:.2f} on {pred}"
                            else:
                                st.session_state.advice = "No bet pending."
                            st.session_state.last_was_tie = False
                            st.success("Undone last action.")
                            st.rerun()
                        else:
                            st.session_state.sequence.pop()
                            st.session_state.pending_bet = None
                            st.session_state.advice = "No bet pending."
                            st.session_state.last_was_tie = False
                            st.success("Undone last result.")
                            st.rerun()
                except Exception as e:
                    logging.error(f"Undo error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Error undoing last action: {str(e)}")
        logging.debug("render_result_input completed")
    except Exception as e:
        logging.error(f"render_result_input error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering result input. Try resetting the session.")
