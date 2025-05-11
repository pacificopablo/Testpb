import streamlit as st
from collections import defaultdict

# --- APP CONFIG ---
try:
    st.set_page_config(layout="centered", page_title="BACCARAT PLAYER/BANKER PREDICTOR")
except Exception as e:
    st.error(f"Error setting page config: {e}")

# --- SESSION STATE INIT ---
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
    st.session_state.pending_prediction = None
    st.session_state.advice = ""
    st.session_state.insights = {}
    st.session_state.pattern_volatility = 0.0
    st.session_state.base_bet = 10.0
    st.session_state.flat_bet_amount = 10.0
    st.session_state.t3_level = 1
    st.session_state.bet_history = []
    st.session_state.betting_strategy = "T3"
    st.session_state.undo_stack = []
    st.session_state.bankroll = 1000.0  # Initialize bankroll
    st.session_state.last_bet_outcome = None  # Track latest bet outcome

# --- PREDICTION FUNCTION ---
def predict_next():
    try:
        sequence = st.session_state.sequence
        base_bet = st.session_state.base_bet
        t3_level = st.session_state.t3_level
        strategy = st.session_state.betting_strategy
        flat_bet_amount = st.session_state.flat_bet_amount
        total_p_b = 0.4462 + 0.4586
        default_p = 0.4462 / total_p_b
        default_b = 0.4586 / total_p_b

        if len(sequence) < 2:
            insights = {
                "Overall": f"No prediction: Need at least 2 outcomes (Current: {len(sequence)})",
                "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
            }
            return None, 0, insights, 0.0
        elif len(sequence) < 3:
            insights = {
                "Overall": f"No prediction: Need at least 3 outcomes for trigram (Current: {len(sequence)})",
                "Betting Strategy": f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else ""),
            }
            return None, 0, insights, 0.0

        window_size = 50
        recent_sequence = sequence[-window_size:]

        bigram_transitions = defaultdict(lambda: defaultdict(int))
        trigram_transitions = defaultdict(lambda: defaultdict(int))
        pattern_transitions = defaultdict(lambda: defaultdict(int))
        streak_count = 0
        current_streak = None
        chop_count = 0
        double_count = 0
        pattern_changes = 0
        last_pattern = None

        for i in range(len(recent_sequence) - 1):
            if i < len(recent_sequence) - 2:
                bigram = tuple(recent_sequence[i:i+2])
                next_outcome = recent_sequence[i+2]
                bigram_transitions[bigram][next_outcome] += 1
            if i < len(recent_sequence) - 3:
                trigram = tuple(recent_sequence[i:i+3])
                next_outcome = recent_sequence[i+3]
                trigram_transitions[trigram][next_outcome] += 1
            if i > 0:
                if recent_sequence[i] == recent_sequence[i-1]:
                    if current_streak == recent_sequence[i]:
                        streak_count += 1
                    else:
                        current_streak = recent_sequence[i]
                        streak_count = 1
                    if i > 1 and recent_sequence[i-1] == recent_sequence[i-2]:
                        double_count += 1
                else:
                    current_streak = None
                    streak_count = 0
                    if i > 1 and recent_sequence[i] != recent_sequence[i-2]:
                        chop_count += 1
            if i < len(recent_sequence) - 2:
                current_pattern = 'streak' if streak_count >= 2 else 'chop' if chop_count >= 2 else 'double' if double_count >= 1 else 'other'
                if last_pattern and last_pattern != current_pattern:
                    pattern_changes += 1
                last_pattern = current_pattern
                next_outcome = recent_sequence[i+2]
                pattern_transitions[current_pattern][next_outcome] += 1

        st.session_state.pattern_volatility = pattern_changes / max(len(recent_sequence) - 2, 1)

        bigram = tuple(recent_sequence[-2:])
        total_transitions = sum(bigram_transitions[bigram].values())
        if total_transitions > 0:
            bigram_p_prob = bigram_transitions[bigram]['P'] / total_transitions
            bigram_b_prob = bigram_transitions[bigram]['B'] / total_transitions
        else:
            bigram_p_prob = default_p
            bigram_b_prob = default_b
        bigram_pred = 'P' if bigram_p_prob > bigram_b_prob else 'B'

        trigram = tuple(recent_sequence[-3:])
        total_transitions = sum(trigram_transitions[trigram].values())
        if total_transitions > 0:
            trigram_p_prob = trigram_transitions[trigram]['P'] / total_transitions
            trigram_b_prob = trigram_transitions[trigram]['B'] / total_transitions
        else:
            trigram_p_prob = default_p
            trigram_b_prob = default_b
        trigram_pred = 'P' if trigram_p_prob > trigram_b_prob else 'B'

        if bigram_pred == trigram_pred:
            pred = bigram_pred
            overall_p = (bigram_p_prob + trigram_p_prob) / 2
            overall_b = (bigram_b_prob + trigram_b_prob) / 2
            conf = max(overall_p, overall_b) * 100
            if strategy == "T3":
                bet_amount = base_bet * t3_level
                bet_info = f"T3: Bet ${bet_amount:.2f} (Level {t3_level})"
            else:
                bet_amount = flat_bet_amount
                bet_info = f"FlatBet: ${bet_amount:.2f}"
        else:
            pred = None
            overall_p = (bigram_p_prob + trigram_p_prob) / 2
            overall_b = (bigram_b_prob + trigram_b_prob) / 2
            conf = max(overall_p, overall_b) * 100
            bet_amount = 0.0
            bet_info = f"{strategy}: No bet" + (f" (Level {t3_level})" if strategy == "T3" else "")

        insights = {
            'Overall': f"P: {overall_p*100:.1f}%, B: {overall_b*100:.1f}%",
            'Volatility': f"{st.session_state.pattern_volatility:.2f}",
            'Betting Strategy': bet_info,
        }
        if strategy == "T3":
            if len(st.session_state.bet_history) > 0:
                wins = sum(1 for pred, actual, _ in st.session_state.bet_history[-3:] if pred == actual)
                total_bets = min(3, len(st.session_state.bet_history))
                losses = total_bets - wins
                win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
                insights['Current T3 Cycle'] = f"W: {wins}, L: {losses}"
                insights['Bet History'] = f"Last 3 Bets: Win Rate: {win_rate:.1f}% ({wins}/{total_bets})"
            else:
                insights['Current T3 Cycle'] = "W: 0, L: 0"
        if pred is None:
            insights['Status'] = "No prediction: Bigram and trigram predictions differ"

        return pred, conf, insights, bet_amount
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, 0, {"Error": str(e)}, 0.0

# --- UNDO FUNCTION ---
def undo_last_action():
    try:
        if st.session_state.undo_stack:
            last_state = st.session_state.undo_stack.pop()
            st.session_state.sequence = last_state['sequence']
            st.session_state.pending_prediction = last_state['pending_prediction']
            st.session_state.advice = last_state['advice']
            st.session_state.insights = last_state['insights']
            st.session_state.pattern_volatility = last_state['pattern_volatility']
            st.session_state.t3_level = last_state['t3_level']
            st.session_state.bet_history = last_state['bet_history']
            st.session_state.bankroll = last_state['bankroll']
            st.session_state.last_bet_outcome = last_state.get('last_bet_outcome', None)
            pred, conf, insights, bet_amount = predict_next()
            st.session_state.pending_prediction = pred
            st.session_state.insights = insights
            st.session_state.advice = (
                f"Prediction: {pred} ({conf:.1f}%), {insights['Betting Strategy']}"
                if pred
                else f"No prediction: {insights.get('Status', 'Bigram and trigram predictions differ')}, {insights['Betting Strategy']}"
            )
            if st.session_state.pattern_volatility > 0.5:
                st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        else:
            st.warning("Nothing to undo.")
    except Exception as e:
        st.error(f"Error during undo: {e}")

# --- PROCESS RESULT ---
def place_result(result):
    try:
        current_state = {
            'sequence': st.session_state.sequence.copy(),
            'pending_prediction': st.session_state.pending_prediction,
            'advice': st.session_state.advice,
            'insights': st.session_state.insights.copy(),
            'pattern_volatility': st.session_state.pattern_volatility,
            't3_level': st.session_state.t3_level,
            'bet_history': st.session_state.bet_history.copy(),
            'bankroll': st.session_state.bankroll,
            'last_bet_outcome': st.session_state.last_bet_outcome,
        }
        st.session_state.undo_stack.append(current_state)
        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) > 100:
            st.session_state.sequence = st.session_state.sequence[-100:]
        pred, conf, insights, bet_amount = predict_next()
        if pred is None:
            st.session_state.pending_prediction = None
            st.session_state.advice = f"No prediction: Bigram and trigram predictions differ, {insights['Betting Strategy']}"
            st.session_state.last_bet_outcome = "No bet placed"
            if st.session_state.pattern_volatility > 0.5:
                st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        else:
            st.session_state.pending_prediction = pred
            st.session_state.advice = f"Prediction: {pred} ({conf:.1f}%), {insights['Betting Strategy']}"
            if st.session_state.betting_strategy == "T3":
                st.session_state.bet_history.append((pred, result, bet_amount))
                if pred == result:
                    # Win: Add bet_amount (1:1 payout for Player, 0.95:1 for Banker)
                    if pred == 'P':
                        st.session_state.bankroll += bet_amount
                        st.session_state.last_bet_outcome = f"Win: Bet on Player, Result Player (+${bet_amount:.2f})"
                    elif pred == 'B':
                        st.session_state.bankroll += bet_amount * 0.95  # Banker commission
                        st.session_state.last_bet_outcome = f"Win: Bet on Banker, Result Banker (+${bet_amount * 0.95:.2f})"
                else:
                    # Loss: Subtract bet_amount
                    st.session_state.bankroll -= bet_amount
                    st.session_state.last_bet_outcome = f"Loss: Bet on {pred}, Result {result} (-${bet_amount:.2f})"
                if len(st.session_state.bet_history) >= 3:
                    wins = sum(1 for p, a, _ in st.session_state.bet_history[-3:] if p == a)
                    losses = 3 - wins
                    if wins == 3:
                        st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
                    elif wins == 2 and losses == 1:
                        st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                    elif wins == 1 and losses == 2:
                        st.session_state.t3_level += 1
                    elif losses == 3:
                        st.session_state.t3_level += 2
                    st.session_state.bet_history = []
            else:  # FlatBet
                if pred == result:
                    if pred == 'P':
                        st.session_state.bankroll += bet_amount
                        st.session_state.last_bet_outcome = f"Win: Bet on Player, Result Player (+${bet_amount:.2f})"
                    elif pred == 'B':
                        st.session_state.bankroll += bet_amount * 0.95
                        st.session_state.last_bet_outcome = f"Win: Bet on Banker, Result Banker (+${bet_amount * 0.95:.2f})"
                else:
                    st.session_state.bankroll -= bet_amount
                    st.session_state.last_bet_outcome = f"Loss: Bet on {pred}, Result {result} (-${bet_amount:.2f})"
            if st.session_state.pattern_volatility > 0.5:
                st.session_state.advice += f", High pattern volatility ({st.session_state.pattern_volatility:.2f})"
        st.session_state.insights = insights
        # Warn if bankroll is low
        if st.session_state.bankroll < st.session_state.base_bet:
            st.session_state.advice += " | Warning: Bankroll too low for base bet!"
    except Exception as e:
        st.error(f"Error placing result: {e}")

# --- UI ---
try:
    st.title("BACCARAT PLAYER/BANKER PREDICTOR")

    st.subheader("Enter Game Result")
    st.session_state.betting_strategy = st.selectbox(
        "Select Betting Strategy", ["T3", "FlatBet"], index=["T3", "FlatBet"].index(st.session_state.betting_strategy)
    )
    st.session_state.base_bet = st.number_input(
        "Base Bet Amount ($)", min_value=0.01, value=st.session_state.base_bet, step=1.0, format="%.2f"
    )
    st.markdown(f"**Current Base Bet**: ${st.session_state.base_bet:.2f}")
    st.session_state.bankroll = st.number_input(
        "Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0, format="%.2f"
    )
    st.markdown(f"**Current Bankroll**: ${st.session_state.bankroll:.2f}")
    if st.session_state.betting_strategy == "FlatBet":
        st.session_state.flat_bet_amount = st.number_input(
            "Flat Bet Amount ($)", min_value=0.01, value=st.session_state.flat_bet_amount, step=1.0, format="%.2f"
        )

    st.markdown("""
    <style>
    div.stButton > button {
        width: 90px;
        height: 35px;
        font-size: 14px;
        font-weight: bold;
        border-radius: 6px;
        border: 1px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div.stButton > button:hover {
        transform: scale(1.08);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
    }
    div.stButton > button:active {
        transform: scale(0.95);
        box-shadow: none;
    }
    div.stButton > button[kind="player_btn"] {
        background: linear-gradient(to bottom, #007bff, #0056b3);
        border-color: #0056b3;
        color: white;
    }
    div.stButton > button[kind="player_btn"]:hover {
        background: linear-gradient(to bottom, #339cff, #007bff);
    }
    div.stButton > button[kind="banker_btn"] {
        background: linear-gradient(to bottom, #dc3545, #a71d2a);
        border-color: #a71d2a;
        color: white;
    }
    div.stButton > button[kind="banker_btn"]:hover {
        background: linear-gradient(to bottom, #ff6666, #dc3545);
    }
    div.stButton > button[kind="undo_btn"] {
        background: linear-gradient(to bottom, #6c757d, #495057);
        border-color: #495057;
        color: white;
    }
    div.stButton > button[kind="undo_btn"]:hover {
        background: linear-gradient(to bottom, #adb5bd, #6c757d);
    }
    @media (max-width: 600px) {
        div.stButton > button {
            width: 80%;
            max-width: 150px;
            height: 40px;
            font-size: 12px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Player", key="player_btn"):
            place_result("P")
    with col2:
        if st.button("Banker", key="banker_btn"):
            place_result("B")
    with col3:
        undo_disabled = len(st.session_state.undo_stack) == 0
        if st.button("Undo", key="undo_btn", disabled=undo_disabled):
            undo_last_action()

    st.subheader("Current Sequence (Bead Plate)")
    sequence = st.session_state.sequence[-90:]
    grid = [[] for _ in range(15)]
    for i, result in enumerate(sequence):
        col_index = i // 6
        if col_index < 15:
            grid[col_index].append(result)
    for col in grid:
        while len(col) < 6:
            col.append('')
    bead_plate_html = "<div style='display: flex; flex-direction: row; gap: 5px; max-width: 100%; overflow-x: auto;'>"
    for col in grid:
        col_html = "<div style='display: flex; flex-direction: column; gap: 5px;'>"
        for result in col:
            if result == '':
                col_html += "<div style='width: 20px; height: 20px; border: 1px solid #ddd; border-radius: 50%;'></div>"
            elif result == 'P':
                col_html += "<div style='width: 20px; height: 20px; background-color: blue; border-radius: 50%;'></div>"
            elif result == 'B':
                col_html += "<div style='width: 20px; height: 20px; background-color: red; border-radius: 50%;'></div>"
        col_html += "</div>"
        bead_plate_html += col_html
    bead_plate_html += "</div>"
    st.markdown(bead_plate_html, unsafe_allow_html=True)

    if st.session_state.pending_prediction:
        side = st.session_state.pending_prediction
        color = 'blue' if side == 'P' else 'red'
        advice_parts = st.session_state.advice.split(', ')
        prob = advice_parts[0].split('(')[-1].split('%')[0] if '(' in advice_parts[0] else '0'
        bet_info = advice_parts[1] if len(advice_parts) > 1 else 'No bet'
        st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Prob: {prob}% | {bet_info}</h4>", unsafe_allow_html=True)
    else:
        st.info(st.session_state.advice)

    # Display latest bet outcome
    if st.session_state.last_bet_outcome:
        if "Win" in st.session_state.last_bet_outcome:
            st.success(st.session_state.last_bet_outcome)
        elif "Loss" in st.session_state.last_bet_outcome:
            st.error(st.session_state.last_bet_outcome)
        else:
            st.info(st.session_state.last_bet_outcome)

    st.subheader("Prediction Insights")
    if st.session_state.insights:
        st.markdown("**Factors Contributing to Prediction:**")
        for factor, contribution in st.session_state.insights.items():
            st.markdown(f"- **{factor}**: {contribution}")
        if st.session_state.pattern_volatility > 0.5:
            st.warning(f"**High Pattern Volatility**: {st.session_state.pattern_volatility:.2f}")
    else:
        st.markdown("No insights available yet. Enter at least 3 Player or Banker results to generate predictions.")
except Exception as e:
    st.error(f"Error rendering UI: {e}")
