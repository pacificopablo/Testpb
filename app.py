import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import random
import string
from collections import defaultdict

st.set_page_config(layout="centered", page_title="MANG BACCARAT GROUP")
st.title("MANG BACCARAT GROUP")

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('baccarat_tracker.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT,
            verified INTEGER DEFAULT 0,
            bankroll REAL DEFAULT 1500
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            hand_number INTEGER,
            bet_size REAL,
            outcome TEXT,
            profit_loss REAL,
            session_profit REAL,
            bankroll REAL,
            t3_level INTEGER,
            t3_results TEXT,
            sequence TEXT,
            wins INTEGER,
            losses INTEGER,
            prediction_accuracy TEXT,
            consecutive_losses INTEGER,
            loss_log TEXT,
            last_was_tie INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            code TEXT
        )
    ''')
    conn.commit()
    return conn

# --- AUTHENTICATION FUNCTIONS ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def generate_verification_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# --- SESSION STATE INIT ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'view' not in st.session_state:
    st.session_state.view = 'login'
if 'error' not in st.session_state:
    st.session_state.error = ''
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1500.0
    st.session_state.base_bet = 15.0  # 1% of default bankroll
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
    st.session_state.initial_bankroll = 1500.0
    st.session_state.target_hit = False
    st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
    st.session_state.consecutive_losses = 0
    st.session_state.loss_log = []
    st.session_state.last_was_tie = False
    st.session_state.session_profit = 0.0
    st.session_state.daily_loss = 0.0

# Validate strategy
if st.session_state.strategy not in ['T3', 'Flatbet']:
    st.session_state.strategy = 'T3'

# --- MAIN APP ---
def main():
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
        div.stButton > button[kind="banker_btn"] {
            background: linear-gradient(to bottom, #dc3545, #a71d2a);
            border-color: #a71d2a;
            color: white;
        }
        div.stButton > button[kind="tie_btn"] {
            background: linear-gradient(to bottom, #28a745, #1e7e34);
            border-color: #1e7e34;
            color: white;
        }
        div.stButton > button[kind="undo_btn"] {
            background: linear-gradient(to bottom, #6c757d, #545b62);
            border-color: #545b62;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    conn = init_db()
    c = conn.cursor()

    if st.session_state.error:
        st.error(st.session_state.error)

    # --- SIGNUP VIEW ---
    if st.session_state.view == 'signup':
        st.subheader("Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            try:
                hashed_password = hash_password(password)
                c.execute('INSERT INTO users (email, password, bankroll) VALUES (?, ?, ?)', 
                         (email, hashed_password, st.session_state.bankroll))
                code = generate_verification_code()
                c.execute('INSERT INTO verifications (email, code) VALUES (?, ?)', (email, code))
                conn.commit()
                st.session_state.error = f"Verification code sent to {email}. Code: {code} (Check console for demo)."
                st.session_state.view = 'verify'
            except sqlite3.IntegrityError:
                st.session_state.error = "Email already exists."
            except Exception:
                st.session_state.error = "Signup failed."
        if st.button("Back to Login"):
            st.session_state.view = 'login'
            st.session_state.error = ''

    # --- EMAIL VERIFICATION VIEW ---
    elif st.session_state.view == 'verify':
        st.subheader("Verify Email")
        email = st.text_input("Email")
        code = st.text_input("Verification Code")
        if st.button("Verify"):
            c.execute('SELECT code FROM verifications WHERE email = ?', (email,))
            row = c.fetchone()
            if row and row[0] == code:
                c.execute('UPDATE users SET verified = 1 WHERE email = ?', (email,))
                c.execute('DELETE FROM verifications WHERE email = ?', (email,))
                conn.commit()
                st.session_state.error = "Email verified. Please log in."
                st.session_state.view = 'login'
            else:
                st.session_state.error = "Invalid code."
        if st.button("Back to Login"):
            st.session_state.view = 'login'
            st.session_state.error = ''

    # --- LOGIN VIEW ---
    elif st.session_state.view == 'login':
        st.subheader("Log In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            if user and verify_password(password, user[2]) and user[3]:
                st.session_state.user = {'id': user[0], 'email': user[1], 'bankroll': user[4]}
                st.session_state.bankroll = user[4]
                st.session_state.base_bet = user[4] * 0.01  # 1% of bankroll
                st.session_state.initial_bankroll = user[4]
                # Load latest session data
                c.execute('SELECT * FROM sessions WHERE user_id = ? ORDER BY id DESC LIMIT 1', (user[0],))
                session = c.fetchone()
                if session:
                    st.session_state.sequence = eval(session[10]) if session[10] else []
                    st.session_state.t3_level = session[8]
                    st.session_state.t3_results = eval(session[9]) if session[9] else []
                    st.session_state.wins = session[11]
                    st.session_state.losses = session[12]
                    st.session_state.prediction_accuracy = eval(session[13]) if session[13] else {'P': 0, 'B': 0, 'total': 0}
                    st.session_state.consecutive_losses = session[14]
                    st.session_state.loss_log = eval(session[15]) if session[15] else []
                    st.session_state.last_was_tie = bool(session[16])
                    st.session_state.session_profit = session[6]
                st.session_state.view = 'tracker'
                st.session_state.error = ''
            elif user and not user[3]:
                st.session_state.error = "Email not verified."
            else:
                st.session_state.error = "Invalid credentials."
        if st.button("Sign Up Instead"):
            st.session_state.view = 'signup'
            st.session_state.error = ''

    # --- TRACKER VIEW ---
    elif st.session_state.view == 'tracker' and st.session_state.user:
        st.subheader(f"Welcome, {st.session_state.user['email']}")
        if st.button("Log Out"):
            st.session_state.user = None
            st.session_state.view = 'login'
            st.session_state.error = ''
            st.experimental_rerun()

        # --- RESET BUTTON ---
        if st.button("Reset Session"):
            st.session_state.bankroll = st.session_state.initial_bankroll
            st.session_state.base_bet = st.session_state.initial_bankroll * 0.01
            st.session_state.sequence = []
            st.session_state.pending_bet = None
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.advice = ""
            st.session_state.history = []
            st.session_state.wins = 0
            st.session_state.losses = 0
            st.session_state.session_profit = 0.0
            st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
            st.session_state.consecutive_losses = 0
            st.session_state.loss_log = []
            st.session_state.last_was_tie = False
            c.execute('DELETE FROM sessions WHERE user_id = ?', (st.session_state.user['id'],))
            c.execute('UPDATE users SET bankroll = ? WHERE id = ?', 
                     (st.session_state.initial_bankroll, st.session_state.user['id']))
            conn.commit()
            st.experimental_rerun()

        # --- SETUP FORM ---
        st.subheader("Setup")
        with st.form("setup_form"):
            bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
            base_bet = st.number_input("Enter Base Bet ($)", min_value=0.0, value=st.session_state.base_bet, step=1.0, 
                                      help="Recommended: 1% of bankroll")
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
                st.session_state.error = "Bankroll must be positive."
            elif base_bet <= 0:
                st.session_state.error = "Base bet must be positive."
            elif base_bet > bankroll:
                st.session_state.error = "Base bet cannot exceed bankroll."
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
                st.session_state.session_profit = 0.0
                st.session_state.prediction_accuracy = {'P': 0, 'B': 0, 'total': 0}
                st.session_state.consecutive_losses = 0
                st.session_state.loss_log = []
                st.session_state.last_was_tie = False
                c.execute('DELETE FROM sessions WHERE user_id = ?', (st.session_state.user['id'],))
                c.execute('UPDATE users SET bankroll = ? WHERE id = ?', (bankroll, st.session_state.user['id']))
                conn.commit()
                st.success(f"Session started with {betting_strategy} strategy!")

        # --- PROFIT LOCK & GROW LOGIC ---
        unit_size = st.session_state.base_bet
        bet_size = unit_size * 1.5 if st.session_state.session_profit >= 75 else unit_size
        if st.session_state.bankroll >= st.session_state.initial_bankroll * 1.1:  # +10%
            unit_size = st.session_state.bankroll * 0.01
            bet_size = unit_size * 1.5 if st.session_state.session_profit >= 75 else unit_size
            st.session_state.base_bet = unit_size
        if st.session_state.bankroll >= st.session_state.initial_bankroll * 1.2:  # +20%
            withdrawal = 0.5 * (st.session_state.bankroll - st.session_state.initial_bankroll)
            st.session_state.bankroll = st.session_state.initial_bankroll * 1.1
            st.session_state.base_bet = st.session_state.bankroll * 0.01
            unit_size = st.session_state.base_bet
            bet_size = unit_size
            c.execute('UPDATE users SET bankroll = ? WHERE id = ?', 
                     (st.session_state.bankroll, st.session_state.user['id']))
            conn.commit()
            st.session_state.error = f"Withdrew ${withdrawal:.2f}. New bankroll: ${st.session_state.bankroll:.2f}."
            st.experimental_rerun()

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
            st.session_state.base_bet = st.session_state.initial_bankroll * 0.01
            st.session_state.sequence = []
            st.session_state.pending_bet = None
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.advice = "Session reset: Target reached."
            st.session_state.history = []
            st.session_state.wins = 0
            st.session_state.losses = 0
            st.session_state.target_hit = False
            st.session_state.session_profit = 0.0
            st.session_state.consecutive_losses = 0
            st.session_state.loss_log = []
            st.session_state.last_was_tie = False
            c.execute('DELETE FROM sessions WHERE user_id = ?', (st.session_state.user['id'],))
            c.execute('UPDATE users SET bankroll = ? WHERE id = ?', 
                     (st.session_state.initial_bankroll, st.session_state.user['id']))
            conn.commit()

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
                profit_loss = 0
                if win:
                    profit_loss = bet_amount * 0.95 if selection == 'B' else bet_amount
                    st.session_state.bankroll += profit_loss
                    if st.session_state.strategy == 'T3':
                        st.session_state.t3_results.append('W')
                    st.session_state.wins += 1
                    st.session_state.prediction_accuracy[selection] += 1
                    st.session_state.consecutive_losses = 0
                else:
                    profit_loss = -bet_amount
                    st.session_state.bankroll += profit_loss
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
                st.session_state.session_profit += profit_loss
                st.session_state.daily_loss = min(st.session_state.daily_loss + profit_loss, 0)
                st.session_state.prediction_accuracy['total'] += 1

                # Stop-Loss/Stop-Win
                if st.session_state.session_profit <= -120:
                    st.session_state.error = "Stop-loss hit at -$120. Session ended."
                    reset_session_auto()
                    st.experimental_rerun()
                elif st.session_state.session_profit >= 225:
                    st.session_state.error = "Stop-win hit at +$225. Session ended."
                    reset_session_auto()
                    st.experimental_rerun()
                elif st.session_state.daily_loss <= -60:
                    st.session_state.error = "Daily loss cap hit at -$60. Stop for the day."
                    reset_session_auto()
                    st.experimental_rerun()

                st.session_state.history.append({
                    "Bet": selection,
                    "Result": result,
                    "Amount": bet_amount,
                    "Profit/Loss": profit_loss,
                    "Win": win,
                    "T3_Level": st.session_state.t3_level if st.session_state.strategy == 'T3' else 1,
                    "T3_Results": st.session_state.t3_results.copy() if st.session_state.strategy == 'T3' else []
                })
                if len(st.session_state.history) > 1000:
                    st.session_state.history = st.session_state.history[-1000:]

                # Save session data
                c.execute('''
                    INSERT INTO sessions (user_id, hand_number, bet_size, outcome, profit_loss, session_profit, bankroll, 
                                        t3_level, t3_results, sequence, wins, losses, prediction_accuracy, 
                                        consecutive_losses, loss_log, last_was_tie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (st.session_state.user['id'], len(st.session_state.history), bet_amount, result, 
                      profit_loss, st.session_state.session_profit, st.session_state.bankroll, 
                      st.session_state.t3_level, str(st.session_state.t3_results), str(st.session_state.sequence), 
                      st.session_state.wins, st.session_state.losses, str(st.session_state.prediction_accuracy), 
                      st.session_state.consecutive_losses, str(st.session_state.loss_log), 
                      int(st.session_state.last_was_tie)))
                c.execute('UPDATE users SET bankroll = ? WHERE id = ?', 
                         (st.session_state.bankroll, st.session_state.user['id']))
                conn.commit()

                st.session_state.pending_bet = None

            if not st.session_state.pending_bet and result != 'T':
                st.session_state.consecutive_losses = 0

            st.session_state.sequence.append(result)
            if len(st.session_state.sequence) > 100:
                st.session_state.sequence = st.session_state.sequence[-100:]

            if check_target_hit():
                st.session_state.target_hit = True
                return

            # Check confidence and bankroll
            pred, conf = predict_next()
            if conf < 50.5:
                st.session_state.pending_bet = None
                st.session_state.advice = f"No bet (Confidence: {conf:.1f}% < 50.5%)"
            else:
                bet_amount = st.session_state.base_bet if st.session_state.strategy == 'Flatbet' else \
                            st.session_state.base_bet * st.session_state.t3_level
                bet_amount = bet_size  # Override with Profit Lock & Grow bet size
                if bet_amount > st.session_state.bankroll:
                    st.session_state.pending_bet = None
                    st.session_state.advice = "No bet: Insufficient bankroll."
                else:
                    st.session_state.pending_bet = (bet_amount, pred)
                    st.session_state.advice = f"Next Bet: ${bet_amount:.2f} on {pred} ({conf:.1f}%)"

            # T3 Level Adjustment
            if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
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

        # --- RESULT INPUT ---
        st.subheader("Enter Result")
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
                        st.session_state.bankroll -= last['Profit/Loss']
                        st.session_state.prediction_accuracy[last['Bet']] -= 1
                        st.session_state.consecutive_losses = 0
                    else:
                        st.session_state.bankroll -= last['Profit/Loss']
                        st.session_state.losses -= 1
                        st.session_state.consecutive_losses = max(0, st.session_state.consecutive_losses - 1)
                        st.session_state.loss_log.pop()
                    st.session_state.session_profit -= last['Profit/Loss']
                    st.session_state.daily_loss = min(st.session_state.daily_loss - last['Profit/Loss'], 0)
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
                    c.execute('DELETE FROM sessions WHERE user_id = ? AND id = (SELECT MAX(id) FROM sessions WHERE user_id = ?)', 
                             (st.session_state.user['id'], st.session_state.user['id']))
                    c.execute('UPDATE users SET bankroll = ? WHERE id = ?', 
                             (st.session_state.bankroll, st.session_state.user['id']))
                    conn.commit()

        # --- DISPLAY SEQUENCE AS BEAD PLATE ---
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
            st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.2f} | Win Prob: {conf}%</h4>", unsafe_allow_html=True)
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
        st.markdown(f"**Base Bet (Unit Size)**: ${st.session_state.base_bet:.2f}")
        st.markdown(f"**Session Profit**: ${st.session_state.session_profit:.2f}")
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
                    "Amount": f"${h['Amount']:.2f}",
                    "Profit/Loss": f"${h['Profit/Loss']:.2f}",
                    "Outcome": "Win" if h["Win"] else "Loss",
                    "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-"
                }
                for h in st.session_state.history[-n:]
            ])

        # --- RULES ---
        st.subheader("Profit Lock & Grow Rules")
        st.markdown(f"- Bet 1 unit (${unit_size:.2f}) on predicted outcome (P/B).")
        st.markdown(f"- At session profit +$75, bet 1.5 units (${unit_size * 1.5:.2f}).")
        st.markdown("- Stop-loss: -$120. Stop-win: +$225. Daily cap: -$60.")
        st.markdown(f"- At bankroll ${(st.session_state.initial_bankroll * 1.1):.2f} (+10%), unit = ${(st.session_state.bankroll * 0.01 if st.session_state.bankroll >= st.session_state.initial_bankroll * 1.1 else st.session_state.base_bet):.2f}.")
        st.markdown(f"- At bankroll ${(st.session_state.initial_bankroll * 1.2):.2f} (+20%), withdraw ${(0.5 * (st.session_state.initial_bankroll * 1.2 - st.session_state.initial_bankroll)):.2f}, new bankroll ${(st.session_state.initial_bankroll * 1.1):.2f}.")

    conn.close()

if __name__ == "__main__":
    main()
