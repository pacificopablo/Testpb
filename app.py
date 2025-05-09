import streamlit as st
import pyrebase
import random
from collections import defaultdict
import json
import streamlit.components.v1 as components

# Firebase Configuration (Replace with your Firebase project credentials)
firebase_config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_PROJECT_ID.firebaseapp.com",
    "projectId": "YOUR_PROJECT interventionId",
    "storageBucket": "YOUR_PROJECT_ID.appspot.com",
    "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
    "appId": "YOUR_APP_ID",
    "databaseURL": "https://YOUR_PROJECT_ID.firebaseio.com",
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")

# --- SESSION STATE INIT ---
def initialize_session_state():
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
        st.session_state.user = None
        st.session_state.google_token = None

# Initialize session state
initialize_session_state()

# --- AUTHENTICATION UI ---
def save_user_data():
    if st.session_state.user:
        user_id = st.session_state.user['localId']
        user_data = {
            'bankroll': st.session_state.bankroll,
            'base_bet': st.session_state.base_bet,
            'sequence': st.session_state.sequence,
            'pending_bet': st.session_state.pending_bet,
            'strategy': st.session_state.strategy,
            't3_level': st.session_state.t3_level,
            't3_results': st.session_state.t3_results,
            'advice': st.session_state.advice,
            'history': st.session_state.history,
            'wins': st.session_state.wins,
            'losses': st.session_state.losses,
            'target_mode': st.session_state.target_mode,
            'target_value': st.session_state.target_value,
            'initial_bankroll': st.session_state.initial_bankroll,
            'target_hit': st.session_state.target_hit,
            'prediction_accuracy': st.session_state.prediction_accuracy,
            'consecutive_losses': st.session_state.consecutive_losses,
            'loss_log': st.session_state.loss_log,
            'last_was_tie': st.session_state.last_was_tie,
        }
        db.child("users").child(user_id).child("sessions").set(user_data)

def load_user_data():
    if st.session_state.user:
        user_id = st.session_state.user['localId']
        try:
            user_data = db.child("users").child(user_id).child("sessions").get().val()
            if user_data:
                for key, value in user_data.items():
                    st.session_state[key] = value
        except:
            pass

if 'user' not in st.session_state or not st.session_state.user:
    st.title("Welcome to MANG BACCARAT GROUP")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        # Email/Password Login
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.session_state.google_token = None
                    load_user_data()
                    st.success("Logged in successfully!")
                    st.rerun()
                except Exception as e:
                    st.error("Login failed. Check your credentials.")

        # Google Login
        st.markdown("**Or sign in with Google**")
        google_login_html = f"""
        <script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js"></script>
        <script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js"></script>
        <script>
        const firebaseConfig = {json.dumps(firebase_config)};
        firebase.initializeApp(firebaseConfig);
        const provider = new firebase.auth.GoogleAuthProvider();
        function signInWithGoogle() {{
            firebase.auth().signInWithPopup(provider)
                .then((result) => {{
                    result.user.getIdToken().then((idToken) => {{
                        window.location.href = window.location.pathname + '?google_token=' + encodeURIComponent(idToken);
                    }});
                }})
                .catch((error) => {{
                    window.parent.postMessage({{ type: 'google_login_error', error: error.message }}, '*');
                }});
        }}
        </script>
        <button onclick="signInWithGoogle()" style="background-color: #4285F4; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">
            Sign in with Google
        </button>
        """
        components.html(google_login_html, height=100)

        # Handle Google login response
        google_token = st.query_params.get("google_token", [None])[0]
        if google_token:
            try:
                user = auth.sign_in_with_idp({
                    'requestUri': 'http://localhost:8501',
                    'id_token': google_token
                })
                st.session_state.user = user
                st.session_state.google_token = google_token
                load_user_data()
                st.success("Logged in with Google successfully!")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Google login failed: {str(e)}")

    with tab2:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_button = st.form_submit_button("Sign Up")
            if signup_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user = auth.create_user_with_email_and_password(new_email, new_password)
                        st.session_state.user = user
                        st.session_state.google_token = None
                        initialize_session_state()
                        st.session_state.user = user
                        save_user_data()
                        st.success("Account created and logged in successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error("Sign-up failed. Email may already be in use.")

else:
    # --- MAIN APP ---
    st.title("MANG BACCARAT GROUP")
    st.markdown(f"Logged in as: {st.session_state.user['email']}")
    if st.button("Logout"):
        auth.current_user = None
        st.session_state.clear()
        st.rerun()

    # --- RESET BUTTON ---
    if st.button("Reset Session"):
        initialize_session_state()
        st.session_state.user = st.session_state.user
        st.session_state.google_token = st.session_state.google_token
        save_user_data()
        st.rerun()

    # --- SETUP FORM ---
    st.subheader("Setup")
    with st.form("setup_form"):
        bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
        base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0)
        betting_strategy = st.selectbox(
            "Choose Betting Strategy",
            ["T3", "Flatbet"],
            index=0 if st.session_state.strategy == "T3" else 1,
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
            st.session_state.t3_results = [] if betting_strategy == 'T3' else []
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
            if betting_strategy == 'Flatbet':
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
            save_user_data()
            st.success(f"Session started with {betting_strategy} strategy!")

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
        return ('P', prob_p) if prob_p > prob_b else ('B', prob_b)

    def check_target_hit():
        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
        else:
            unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.base_bet
            return unit_profit >= st.session_state.target_value

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
        save_user_data()

    def place_result(result):
        if st.session_state.target_hit:
            reset_session_auto()
            return
        st.session_state.last_was_tie = (result == 'T')
        bet_amount = 0
        if st.session_state.pending_bet and result != 'T':
            bet_amount, selection = st.session_state.pending_bet
            win = result == selection
            if win:
                st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_results.append('W')
                st.session_state.wins += 1
                st.session_state.prediction_accuracy[selection] += 1
                st.session_state.consecutive_losses = 0
            else:
                st.session_state.bankroll -= bet_amount
                if st.session_state.strategy == 'T3':
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
                "T3_Level": st.session_state.t3_level if st.session_state.strategy == 'T3' else 1,
                "T3_Results": st.session_state.t3_results.copy() if st.session_state.strategy == 'T3' else []
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
            save_user_data()
            return
        pred, conf = predict_next()
        if conf < 50.5:
            st.session_state.pending_bet = None
            st.session_state.advice = f"No bet (Confidence: {conf:.1f}% < 50.5%)"
        else:
            bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else st.session_state.base_bet * st.session_state.t3_level
            if bet_amount > st.session_state.bankroll:
                st.session_state.pending_bet = None
                st.session_state.advice = "No bet: Insufficient bankroll."
            else:
                st.session_state.pending_bet = (bet_amount, pred)
                st.session_state.advice = f"Next Bet: ${bet_amount:.0f} on {pred} ({conf:.1f}%)"
        if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins == 3:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif wins == 2 and losses == 1:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses == 2 and wins == 1:
                st.session_state.t3_level += 1
            elif losses == 3:
                st.session_state.t3_level += 2
            st.session_state.t3_results = []
        save_user_data()

    # --- RESULT INPUT ---
    st.subheader("Enter Result")
    st.markdown("""
    <style>
    div.stButton > button {
        width: 90px; height: 35px; font-size: 14px; font-weight: bold; border-radius: 6px;
        border: 1px solid; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); cursor: pointer;
        transition: all 0.15s ease; display: flex; align-items: center; justify-content: center;
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
                if st.session_state.strategy == 'T3':
                    st.session_state.t3_level = last['T3_Level']
                    st.session_state.t3_results = last['T3_Results']
                else:
                    st.session_state.t3_level = 1
                    st.session_state.t3_results = []
                st.session_state.pending_bet = None
                st.session_state.advice = "Last entry undone."
                st.session_state.last_was_tie = False
                save_user_data()

    # --- BEAD PLATE ---
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
    st.markdown(f"**Betting Strategy**: {st.session_state.strategy}" + 
                (f" | T3 Level: {st.session_state.t3_level}" if st.session_state.strategy == 'T3' else ""))
    st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")

    # --- PREDICTION ACCURACY ---
    st.subheader("Prediction Accuracy")
    total = st.session_state.prediction_accuracy['total']
    if total > 0:
        p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
        b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
        st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
        st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")

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
                "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-"
            }
            for h in st.session_state.history[-n:]
        ])
