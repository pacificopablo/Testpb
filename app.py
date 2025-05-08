import streamlit as st
import random

st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 0.0
    st.session_state.base_bet = 0.0
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.strategy = 'T3'
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = ""
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_mode = 'Profit %'
    st.session_state.target_value = 10.0
    st.session_state.initial_bankroll = 0.0
    st.session_state.target_hit = False
    st.session_state.contrarian_betting = False
    st.session_state.prediction_accuracy = {
        'streak': {'correct': 0, 'total': 0},
        'alternation': {'correct': 0, 'total': 0},
        'dominance': {'correct': 0, 'total': 0},
        'contrarian': {'correct': 0, 'total': 0}
    }

# --- RESET BUTTON ---
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    strategy = st.selectbox("Choose Strategy", ["T3", "Flatbet"], index=["T3", "Flatbet"].index(st.session_state.strategy))
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
    contrarian_betting = st.checkbox("Enable Contrarian Betting", value=st.session_state.contrarian_betting)
    start_clicked = st.form_submit_button("Start Session")

if start_clicked:
    if bankroll <= 0:
        st.error("Bankroll must be positive.")
    elif base_bet <= 0:
        st.error("Base bet must be positive.")
    elif base_bet > bankroll:
        st.error("Base bet cannot exceed bankroll.")
    else:
        st.session_state.bankroll = bankroll
        st.session_state.base_bet = base_bet
        st.session_state.strategy = strategy
        st.session_state.sequence = []
        st.session_state.pending_bet = None
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.advice = ""
        st.session_state.history = []
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.session_state.target_mode = target_mode
        st.session_state.target_value = target_value
        st.session_state.initial_bankroll = bankroll
        st.session_state.target_hit = False
        st.session_state.contrarian_betting = contrarian_betting
        st.session_state.prediction_accuracy = {
            'streak': {'correct': 0, 'total': 0},
            'alternation': {'correct': 0, 'total': 0},
            'dominance': {'correct': 0, 'total': 0},
            'contrarian': {'correct': 0, 'total': 0}
        }
        st.success("Session started!")

# --- FUNCTIONS ---
def predict_next():
    recent = st.session_state.sequence[-10:]  # Short-term trends
    long_term = st.session_state.sequence[-50:]  # Long-term trends
    if not recent:
        return random.choice(['P', 'B']), 50, 'random'

    # Calculate prediction accuracies
    accuracies = {}
    for strategy in ['streak', 'alternation', 'dominance', 'contrarian']:
        total = st.session_state.prediction_accuracy[strategy]['total']
        correct = st.session_state.prediction_accuracy[strategy]['correct']
        accuracies[strategy] = correct / total if total > 0 else 0.5

    # Streak detection
    streak_side = None
    streak_length = 0
    current_side = recent[-1]
    for i in range(len(recent) - 1, -1, -1):
        if recent[i] == current_side:
            streak_length += 1
        else:
            break
    if streak_length >= 3:
        streak_side = current_side

    # Alternation detection
    alternation = False
    if len(recent) >= 4:
        alternating = all(recent[i] != recent[i-1] for i in range(-1, -4, -1))
        if alternating:
            alternation = True

    # Dominance (weighted recent + long-term)
    p_recent = recent.count('P') / len(recent) if recent else 0.5
    b_recent = recent.count('B') / len(recent) if recent else 0.5
    p_long = long_term.count('P') / len(long_term) if long_term else 0.5
    b_long = long_term.count('B') / len(long_term) if long_term else 0.5
    p_weighted = 0.7 * p_recent + 0.3 * p_long
    b_weighted = 0.7 * b_recent + 0.3 * b_long

    # Contrarian logic (bet against streak or dominant side)
    contrarian_side = None
    if streak_side and st.session_state.contrarian_betting:
        contrarian_side = 'P' if streak_side == 'B' else 'B'
    elif p_weighted > b_weighted and st.session_state.contrarian_betting:
        contrarian_side = 'B'
    elif b_weighted > p_weighted and st.session_state.contrarian_betting:
        contrarian_side = 'P'

    # Choose best strategy based on accuracy
    best_strategy = max(accuracies, key=accuracies.get) if max(accuracies.values()) > 0.5 else 'dominance'

    # Decision logic
    if contrarian_side and accuracies['contrarian'] >= 0.5:
        confidence = 60 + (accuracies['contrarian'] * 20)
        return contrarian_side, min(confidence, 90), 'contrarian'
    elif streak_side and accuracies['streak'] >= max(0.5, accuracies['alternation'], accuracies['dominance']):
        confidence = 55 + (streak_length * 5) * accuracies['streak']
        return streak_side, min(confidence, 90), 'streak'
    elif alternation and accuracies['alternation'] >= max(0.5, accuracies['streak'], accuracies['dominance']):
        next_side = 'P' if recent[-1] == 'B' else 'B'
        confidence = 55 + (len(recent) / 10 * 15) * accuracies['alternation']
        return next_side, min(confidence, 90), 'alternation'
    else:
        if p_weighted > b_weighted:
            confidence = 50 + (p_weighted * 30) * accuracies['dominance']
            return 'P', min(confidence, 90), 'dominance'
        elif b_weighted > p_weighted:
            confidence = 50 + (b_weighted * 30) * accuracies['dominance']
            return 'B', min(confidence, 90), 'dominance'
        else:
            return random.choice(['P', 'B']), 50, 'random'

def check_target_hit():
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        if st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit:
            return True
    else:
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.base_bet
        if unit_profit >= st.session_state.target_value:
            return True
    return False

def reset_session_auto():
    st.session_state.bankroll = st.session_state.initial_bankroll
    st.session_state.sequence = []
    st.session_state.pending_bet = None
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.advice = "Session reset: Target reached."
    st.session_state.history = []
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.target_hit = False

def place_result(result):
    if st.session_state.target_hit:
        reset_session_auto()
        return

    bet_amount = 0
    if st.session_state.pending_bet:
        bet_amount, selection, strategy = st.session_state.pending_bet
        win = result == selection
        if win:
            if selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:
                st.session_state.bankroll += bet_amount
            st.session_state.t3_results.append('W')
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[strategy]['correct'] += 1
        else:
            st.session_state.bankroll -= bet_amount
            st.session_state.t3_results.append('L')
            st.session_state.losses += 1
        st.session_state.prediction_accuracy[strategy]['total'] += 1

        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level,
            "T3_Results": st.session_state.t3_results.copy(),
            "Strategy": strategy
        })
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]

        if len(st.session_state.t3_results) == 3:
            w = st.session_state.t3_results.count('W')
            l = st.session_state.t3_results.count('L')
            if w == 3:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif w == 2:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif l == 2:
                st.session_state.t3_level += 1
            elif l == 3:
                st.session_state.t3_level += 2
            st.session_state.t3_results = []

        st.session_state.pending_bet = None

    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > 100:
        st.session_state.sequence = st.session_state.sequence[-100:]

    if check_target_hit():
        st.session_state.target_hit = True
        return

    pred, conf, strategy = predict_next()
    bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
    if bet_amount <= st.session_state.bankroll and conf >= 60:
        st.session_state.pending_bet = (bet_amount, pred, strategy)
        st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.0f}% - {strategy.capitalize()})"
    else:
        st.session_state.pending_bet = None
        st.session_state.advice = "Skip bet: Low confidence or insufficient bankroll"

# --- RESULT INPUT ---
st.subheader("Enter Result")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Player (P)"):
        place_result("P")
with col2:
    if st.button("Banker (B)"):
        place_result("B")
with col3:
    if st.button("Undo Last"):
        if st.session_state.history and st.session_state.sequence:
            st.session_state.sequence.pop()
            last = st.session_state.history.pop()
            if last['Win']:
                st.session_state.wins -= 1
                st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                st.session_state.prediction_accuracy[last['Strategy']]['correct'] -= 1
            else:
                st.session_state.losses -= 1
                st.session_state.bankroll += last['Amount']
            st.session_state.prediction_accuracy[last['Strategy']]['total'] -= 1
            st.session_state.t3_level = last['T3_Level']
            st.session_state.t3_results = last['T3_Results']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."

# --- DISPLAY SEQUENCE ---
st.subheader("Current Sequence")
latest_sequence = st.session_state.sequence[-20:] if 'sequence' in st.session_state else []
st.text(", ".join(latest_sequence or ["None"]))

# --- PREDICTION DISPLAY ---
if st.session_state.pending_bet:
    amount, side, strategy = st.session_state.pending_bet
    color = 'blue' if side == 'P' else 'red'
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.0f} | Win Prob: {conf}% ({strategy.capitalize()})</h4>", unsafe_allow_html=True)
else:
    if not st.session_state.target_hit:
        st.info(st.session_state.advice)

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
st.markdown(f"**Contrarian Betting**: {'Enabled' if st.session_state.contrarian_betting else 'Disabled'}")

# --- PREDICTION ACCURACY ---
st.subheader("Prediction Accuracy")
for strategy, stats in st.session_state.prediction_accuracy.items():
    total = stats['total']
    if total > 0:
        accuracy = (stats['correct'] / total) * 100
        st.markdown(f"**{strategy.capitalize()}**: {stats['correct']}/{total} ({accuracy:.1f}%)")

# --- UNIT PROFIT ---
if st.session_state.base_bet > 0:
    units_profit = int((st.session_state.bankroll - st.session_state.initial_bankroll) // st.session_state.base_bet)
    st.markdown(f"**Units Profit**: {units_profit}")

# --- HISTORY TABLE ---
if st.session_state.history:
    st.subheader("Bet History")
    n = st.slider("Show last N bets", 5, 50, 10)
    st.dataframe([
        {
            "Bet": h["Bet"],
            "Result": h["Result"],
            "Amount": f"${h['Amount']:.0f}",
            "Outcome": "Win" if h["Win"] else "Loss",
            "T3 Level": h["T3_Level"],
            "Strategy": h["Strategy"]
        }
        for h in st.session_state.history[-n:]
    ])
