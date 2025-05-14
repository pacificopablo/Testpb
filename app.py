import streamlit as st
import random
from collections import defaultdict
import pandas as pd
import plotly.express as px

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
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False

# --- RESET BUTTON ---
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- SETUP FORM ---
st.subheader("Setup")
with st.form("setup_form"):
    bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
    base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
    betting_strategy = st.selectbox(
        "Choose Betting Strategy",
        ["T3", "Flatbet"],
        index=["T3", "Flatbet"].index(st.session_state.strategy),
        help="T3: Adjusts bet size based on wins/losses. Flatbet: Uses a fixed bet size."
    )
    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
    target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
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
        st.session_state.strategy = betting_strategy
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
        st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
        st.session_state.consecutive_losses = 0
        st.session_state.loss_log = []
        st.session_state.last_was_tie = False
        st.success("Session started!")

# --- FUNCTIONS ---
def predict_next():
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    if len(sequence) < 2:
        return 'B', 45.86
    bigram = sequence[-2:]
    transitions = defaultdict(int)
    for i in range(len(sequence) - 2):
        if sequence[i:i+2] == bigram:
            next_outcome = sequence[i+2]
            transitions[next_outcome] += 1
    total_transitions = sum(transitions.values())
    if total_transitions > 0:
        prob_p = (transitions['P'] / total_transitions) * 100
        prob_b = (transitions['B'] / total_transitions) * 100
    else:
        prob_p = 44.62
        prob_b = 45.86
    if prob_p > prob_b:
        return 'P', prob_p
    return 'B', prob_b

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
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False

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
            st.session_state.t3_results.append('W')
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
        else:
            st.session_state.bankroll -= bet_amount
            st.session_state.t3_results.append('L')
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
        st.session_state.prediction_accuracy['total'] += 1
        st.session_state.history.append({
            "Bet": selection,
            "Result": result,
            "Amount": bet_amount,
            "Win": win,
            "T3_Level": st.session_state.t3_level,
            "T3_Results": st.session_state.t3_results.copy()
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
        bet_amount = st.session_state.base_bet * st.session_state.t3_level if st.session_state.strategy == 'T3' else st.session_state.base_bet
        if bet_amount > st.session_state.bankroll:
            st.session_state.pending_bet = None
            st.session_state.advice = "No bet: Insufficient bankroll."
        else:
            st.session_state.pending_bet = (bet_amount, pred)
            st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"
    if len(st.session_state.t3_results) == 3 and st.session_state.strategy == 'T3':
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = st.session_state.t3_level + 1
        elif losses == 3:
            st.session_state.t3_level = st.session_state.t3_level + 2
        st.session_state.t3_results = []

def get_prediction_insights():
    insights = {}
    total_bets = st.session_state.prediction_accuracy['total']
    p_accuracy = (st.session_state.prediction_accuracy['P'] / total_bets * 100) if total_bets > 0 else 0
    b_accuracy = (st.session_state.prediction_accuracy['B'] / total_bets * 100) if total_bets > 0 else 0
    win_ratio = (st.session_state.wins / total_bets * 100) if total_bets > 0 else 0
    insights['basic_stats'] = {
        'total_bets': total_bets,
        'player_accuracy': p_accuracy,
        'banker_accuracy': b_accuracy,
        'win_ratio': win_ratio
    }
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    for h in st.session_state.history:
        if h['Win']:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
    insights['streaks'] = {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'consecutive_losses': st.session_state.consecutive_losses
    }
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
    p_count = sequence.count('P')
    b_count = sequence.count('B')
    t_count = sequence.count('T')
    total_outcomes = len(sequence)
    insights['outcome_freq'] = {
        'player': (p_count / total_outcomes * 100) if total_outcomes > 0 else 0,
        'banker': (b_count / total_outcomes * 100) if total_outcomes > 0 else 0,
        'tie': (t_count / total_outcomes * 100) if total_outcomes > 0 else 0
    }
    confidence_history = [
        float(log['confidence']) for log in st.session_state.loss_log
    ] + [
        float(st.session_state.advice.split('(')[-1].split('%')[0]) 
        if '(' in st.session_state.advice else 0
        for _ in range(1 if st.session_state.pending_bet else 0)
    ]
    insights['confidence'] = {
        'avg_confidence': sum(confidence_history) / len(confidence_history) if confidence_history else 0,
        'high_confidence_losses': len([c for c in confidence_history if c >= 60 and c in [float(log['confidence']) for log in st.session_state.loss_log]])
    }
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(sequence) - 2):
        bigram = tuple(sequence[i:i+2])
        next_outcome = sequence[i+2]
        bigram_transitions[bigram][next_outcome] += 1
    insights['bigrams'] = bigram_transitions
    advice = []
    if insights['confidence']['avg_confidence'] < 50.5 and total_bets > 10:
        advice.append("Low average prediction confidence detected. Consider pausing bets or switching to Flatbet strategy.")
    if insights['streaks']['consecutive_losses'] >= 3:
        advice.append(f"Warning: {insights['streaks']['consecutive_losses']} consecutive losses. Consider reducing bet size or pausing.")
    if insights['outcome_freq']['tie'] > 20:
        advice.append("High frequency of Ties observed. Be cautious with predictions as Ties disrupt pattern analysis.")
    if insights['confidence']['high_confidence_losses'] > 3:
        advice.append("Multiple high-confidence predictions resulted in losses. Re-evaluate betting on high-confidence predictions.")
    insights['advice'] = advice
    return insights

# --- RESULT INPUT WITH NATIVE STREAMLIT BUTTONS ---
st.subheader("Enter Result")
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
div.stButton > button[kind="tie_btn"] {
    background: linear-gradient(to bottom, #28a745, #1e7e34);
    border-color: #1e7e34;
    color: white;
}
div.stButton > button[kind="tie_btn"]:hover {
    background: linear-gradient(to bottom, #4caf50, #28a745);
}
div.stButton > button[kind="undo_btn"] {
    background: linear-gradient(to bottom, #6c757d, #545b62);
    border-color: #545b62;
    color: white;
}
div.stButton > button[kind="undo_btn"]:hover {
    background: linear-gradient(to bottom, #8e959c, #6c757d);
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
        if st.session_state.history and st.session_state.sequence:
            st.session_state.sequence.pop()
            last = st.session_state.history.pop()
            if last['Win']:
                st.session_state.wins -= 1
                st.session_state.bankroll -= last['Amount'] if last["Bet"] == 'P' else last['Amount'] * 0.95
                st.session_state.prediction_accuracy[last['Bet']] -= 1
                st.session_state.consecutive_losses = 0
            else:
                st.session_state.bankroll += last['Amount']
                st.session_state.losses -= 1
                st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
            st.session_state.prediction_accuracy['total'] -= 1
            st.session_state.t3_level = last['T3_Level']
            st.session_state.t3_results = last['T3_Results']
            st.session_state.pending_bet = None
            st.session_state.advice = "Last entry undone."
            st.session_state.last_was_tie = False

# --- DISPLAY SEQUENCE AS BEAD PLATE ---
st.subheader("Current Sequence (Bead Plate)")
sequence = st.session_state.sequence[-100:] if 'sequence' in st.session_state else []
grid = []
current_col = []
for result in sequence:
    if len(current_col) < 6:
        current_col.append(result)
    else:
        grid.append(current_col)
        current_col = [result]
if current_col:
    grid.append(current_col)
if grid and len(grid[-1]) < 6:
    grid[-1] += [''] * (6 - len(grid[-1]))
num_columns = len(grid)
bead_plate_html = "<div style='display: flex; flex-direction: row; gap: 5px; max-width: 120px; overflow-x: auto;'>"
for col in grid[:num_columns]:
    col_html = "<div style='display: flex; flex-direction: column; gap: 5px;'>"
    for result in col:
        if result == '':
            col_html += "<div style='width: 20px; height: 20px;'></div>"
        elif result == 'P':
            col_html += "<div style='width: 20px; height: 20px; background-color: blue; border-radius: 50%;'></div>"
        elif result == 'B':
            col_html += "<div style='width: 20px; height: 20px; background-color: red; border-radius: 50%;'></div>"
        elif result == 'T':
            col_html += "<div style='width: 20px; height: 20px; background-color: green; border-radius: 50%;'></div>"
    col_html += "</div>"
    bead_plate_html += col_html
bead_plate_html += "</div>"
st.markdown(bead_plate_html, unsafe_allow_html=True)

# --- PREDICTION DISPLAY ---
if st.session_state.pending_bet:
    amount, side = st.session_state.pending_bet
    color = 'blue' if side == 'P' else 'red'
    conf = st.session_state.advice.split('(')[-1].split('%')[0] if '(' in st.session_state.advice else '0'
    st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.0f} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
else:
    if not st.session_state.target_hit:
        st.info(st.session_state.advice)

# --- UNIT PROFIT ---
if st.session_state.base_bet > 0 and st.session_state.initial_bankroll > 0:
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    units_profit = profit / st.session_state.base_bet
    st.markdown(f"**Units Profit**: {units_profit:.2f} units (${profit:.2f})")
else:
    st.markdown("**Units Profit**: 0.00 units ($0.00)")

# --- STATUS ---
st.subheader("Status")
st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
st.markdown(f"**Betting Strategy**: {st.session_state.strategy} | T3 Level: {st.session_state.t3_level}")
st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

# --- PREDICTION ACCURACY ---
st.subheader("Prediction Accuracy")
total = st.session_state.prediction_accuracy['total']
if total > 0:
    p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
    b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
    st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
    st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

# --- PREDICTION INSIGHTS ---
st.subheader("Prediction Insights")
if st.session_state.history or st.session_state.sequence:
    insights = get_prediction_insights()
    st.markdown("**Basic Statistics**")
    st.write(f"Total Bets: {insights['basic_stats']['total_bets']}")
    st.write(f"Player Prediction Accuracy: {insights['basic_stats']['player_accuracy']:.1f}%")
    st.write(f"Banker Prediction Accuracy: {insights['basic_stats']['banker_accuracy']:.1f}%")
    st.write(f"Win Ratio: {insights['basic_stats']['win_ratio']:.1f}%")
    st.markdown("**Streak Analysis**")
    st.write(f"Longest Win Streak: {insights['streaks']['max_win_streak']}")
    st.write(f"Longest Loss Streak: {insights['streaks']['max_loss_streak']}")
    st.write(f"Current Consecutive Losses: {insights['streaks']['consecutive_losses']}")
    st.markdown("**Outcome Frequency**")
    st.write(f"Player Outcomes: {insights['outcome_freq']['player']:.1f}%")
    st.write(f"Banker Outcomes: {insights['outcome_freq']['banker']:.1f}%")
    st.write(f"Tie Outcomes: {insights['outcome_freq']['tie']:.1f}%")
    st.markdown("**Confidence Trends**")
    st.write(f"Average Prediction Confidence: {insights['confidence']['avg_confidence']:.1f}%")
    st.write(f"High-Confidence Losses (≥60%): {insights['confidence']['high_confidence_losses']}")
    st.markdown("**Bigram Transition Patterns**")
    bigram_data = []
    for bigram, transitions in insights['bigrams'].items():
        total = sum(transitions.values())
        if total > 0:
            bigram_data.append({
                'Bigram': f"{bigram[0]}{bigram[1]}",
                'To Player': f"{(transitions['P'] / total * 100):.1f}%",
                'To Banker': f"{(transitions['B'] / total * 100):.1f}%",
                'Occurrences': total
            })
    if bigram_data:
        st.dataframe(bigram_data)
    st.markdown("**Recommendations**")
    if insights['advice']:
        for advice in insights['advice']:
            st.warning(advice)
    else:
        st.info("No specific recommendations at this time. Continue with current strategy.")
    if insights['outcome_freq']['player'] + insights['outcome_freq']['banker'] + insights['outcome_freq']['tie'] > 0:
        fig = px.pie(
            values=[insights['outcome_freq']['player'], insights['outcome_freq']['banker'], insights['outcome_freq']['tie']],
            names=['Player', 'Banker', 'Tie'],
            title='Outcome Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    if len(st.session_state.history) > 0:
        confidence_data = [
            {'Bet': i+1, 'Confidence': float(h['confidence'])}
            for i, h in enumerate(st.session_state.loss_log[-10:])
        ]
        if confidence_data:
            fig = px.line(
                pd.DataFrame(confidence_data),
                x='Bet',
                y='Confidence',
                title='Recent Prediction Confidence Trend',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for insights yet. Place some bets to generate insights.")

# --- LOSS LOG ---
if st.session_state.loss_log:
    st.subheader("Recent Losses")
    st.dataframe([
        {
            "Sequence": ", ".join(log['sequence']),
            "Prediction": log['prediction'],
            "Result": log['result'],
            "Confidence": log['confidence'] + "%"
        }
        for log in st.session_state.loss_log[-5:]
    ])

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
            "T3_Level": h["T3_Level"]
        }
        for h in st.session_state.history[-n:]
    ])
