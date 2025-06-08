import streamlit as st
import random
import math

# Set page configuration
st.set_page_config(page_title="Baccarat Predictor - Enhanced Dominant Pairs System", layout="wide")

class BaccaratPredictor:
    def __init__(self):
        # Initialize session state to persist data
        if 'pair_types' not in st.session_state:
            st.session_state.pair_types = []
            st.session_state.next_prediction = "N/A"
            st.session_state.base_amount = 10.0
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.result_tracker = 0.0
            st.session_state.profit_lock = 0.0
            st.session_state.previous_result = None
            st.session_state.state_history = []
            st.session_state.current_dominance = "N/A"
            st.session_state.streak_type = None
            st.session_state.consecutive_wins = 0
            st.session_state.consecutive_losses = 0
            st.session_state.stats = {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}

    def set_base_amount(self, amount):
        try:
            amount = float(amount)
            if 1 <= amount <= 100:
                st.session_state.base_amount = amount
                st.session_state.bet_amount = st.session_state.base_amount
                self.update_display()
                st.success("Base amount updated successfully.")
            else:
                st.error("Base amount must be between $1 and $100.")
        except ValueError:
            st.error("Please enter a valid number.")

    def new_session(self):
        self.reset_all()
        st.success("New session started.")

    def reset_betting(self):
        if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
            st.warning("Stop-loss reached. Resetting to resume betting.")
        if st.session_state.result_tracker >= 0:
            st.session_state.result_tracker = 0.0
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.consecutive_wins = 0
        st.session_state.consecutive_losses = 0
        st.session_state.streak_type = None

        if len(st.session_state.pair_types) >= 5:
            recent_pairs = st.session_state.pair_types[-10:]
            odd_count = sum(1 for a, b in recent_pairs if a != b)
            even_count = sum(1 for a, b in recent_pairs if a == b)
            result = st.session_state.previous_result
            if abs(odd_count - even_count) < 2:
                st.session_state.current_dominance = "N/A"
                st.session_state.next_prediction = "Hold"
                st.session_state.bet_amount = 0
            elif odd_count > even_count:
                st.session_state.current_dominance = "Odd"
                st.session_state.next_prediction = "Player" if result == 'B' else "Banker"
                st.session_state.bet_amount = st.session_state.base_amount
            else:
                st.session_state.current_dominance = "Even"
                st.session_state.next_prediction = "Player" if result == 'P' else "Banker"
                st.session_state.bet_amount = st.session_state.base_amount
            last_three = [st.session_state.pair_types[-i][1] for i in range(1, min(4, len(st.session_state.pair_types)))]
            if len(last_three) >= 3 and all(r == last_three[0] for r in last_three):
                st.session_state.streak_type = last_three[0]
                st.session_state.next_prediction = "Player" if st.session_state.streak_type == 'P' else "Banker"
                st.session_state.current_dominance = f"Streak ({st.session_state.streak_type})"
                st.session_state.bet_amount = 2 * st.session_state.base_amount
        else:
            st.session_state.next_prediction = "N/A"
            st.session_state.current_dominance = "N/A"
            st.session_state.streak_type = None
            st.session_state.bet_amount = st.session_state.base_amount

        self.update_display()
        st.success("Betting reset.")

    def reset_all(self):
        st.session_state.pair_types = []
        st.session_state.result_tracker = 0.0
        st.session_state.profit_lock = 0.0
        st.session_state.bet_amount = st.session_state.base_amount
        st.session_state.base_amount = 10.0
        st.session_state.next_prediction = "N/A"
        st.session_state.previous_result = None
        st.session_state.state_history = []
        st.session_state.current_dominance = "N/A"
        st.session_state.consecutive_wins = 0
        st.session_state.consecutive_losses = 0
        st.session_state.streak_type = None
        st.session_state.stats = {'wins': 0, 'losses': 0, 'ties': 0, 'streaks': [], 'odd_pairs': 0, 'even_pairs': 0}
        self.update_display()
        st.success("All session data reset, profit lock reset.")

    def record_result(self, result):
        state = {
            'pair_types': st.session_state.pair_types.copy(),
            'previous_result': st.session_state.previous_result,
            'result_tracker': st.session_state.result_tracker,
            'profit_lock': st.session_state.profit_lock,
            'bet_amount': st.session_state.bet_amount,
            'current_dominance': st.session_state.current_dominance,
            'next_prediction': st.session_state.next_prediction,
            'consecutive_wins': st.session_state.consecutive_wins,
            'consecutive_losses': st.session_state.consecutive_losses,
            'streak_type': st.session_state.streak_type,
            'stats': st.session_state.stats.copy()
        }
        st.session_state.state_history.append(state)

        if result == 'T':
            st.session_state.stats['ties'] += 1
            self.update_display()
            return

        if st.session_state.previous_result is None:
            st.session_state.previous_result = result
            st.session_state.next_prediction = "N/A"
            st.session_state.bet_amount = st.session_state.base_amount
            self.update_display()
            return

        if st.session_state.previous_result != 'T':
            pair = (st.session_state.previous_result, result)
            st.session_state.pair_types.append(pair)
            pair_type = "Even" if pair[0] == pair[1] else "Odd"
            st.session_state.stats['odd_pairs' if pair_type == "Odd" else 'even_pairs'] += 1

        last_three = [st.session_state.pair_types[-i][1] for i in range(1, min(4, len(st.session_state.pair_types)))]
        if len(last_three) >= 3 and all(r == result for r in last_three):
            st.session_state.streak_type = result
            st.session_state.stats['streaks'].append(len(last_three))
        else:
            st.session_state.streak_type = None

        if len(st.session_state.pair_types) >= 5:
            recent_pairs = st.session_state.pair_types[-10:]
            odd_count = sum(1 for a, b in recent_pairs if a != b)
            even_count = sum(1 for a, b in recent_pairs if a == b)

            if st.session_state.streak_type:
                st.session_state.next_prediction = "Player" if st.session_state.streak_type == 'P' else "Banker"
                st.session_state.current_dominance = f"Streak ({st.session_state.streak_type})"
                st.session_state.bet_amount = 2 * st.session_state.base_amount
            elif abs(odd_count - even_count) < 2:
                st.session_state.current_dominance = "N/A"
                st.session_state.next_prediction = "Hold"
                st.session_state.bet_amount = 0
            elif odd_count > even_count:
                st.session_state.current_dominance = "Odd"
                st.session_state.next_prediction = "Player" if result == 'B' else "Banker"
                st.session_state.bet_amount = st.session_state.base_amount
            else:
                st.session_state.current_dominance = "Even"
                st.session_state.next_prediction = "Player" if result == 'P' else "Banker"
                st.session_state.bet_amount = st.session_state.base_amount

            if st.session_state.bet_amount == 0:
                st.session_state.bet_amount = st.session_state.base_amount

            if len(st.session_state.pair_types) >= 6 and st.session_state.state_history[-1]['next_prediction'] != "Hold":
                previous_prediction = st.session_state.state_history[-1]['next_prediction']
                effective_bet = min(5 * st.session_state.base_amount, st.session_state.bet_amount)
                if (previous_prediction == "Player" and result == 'P'):
                    st.session_state.result_tracker += effective_bet
                    st.session_state.stats['wins'] += 1
                    st.session_state.consecutive_wins += 1
                    st.session_state.consecutive_losses = 0
                    if st.session_state.result_tracker > st.session_state.profit_lock:
                        st.session_state.profit_lock = st.session_state.result_tracker
                        st.session_state.result_tracker = 0.0
                        st.session_state.bet_amount = st.session_state.base_amount
                        st.success(f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.")
                        self.update_display()
                        return
                    if st.session_state.consecutive_wins >= 2:
                        st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
                elif (previous_prediction == "Banker" and result == 'B'):
                    st.session_state.result_tracker += effective_bet * 0.95
                    st.session_state.stats['wins'] += 1
                    st.session_state.consecutive_wins += 1
                    st.session_state.consecutive_losses = 0
                    if st.session_state.result_tracker > st.session_state.profit_lock:
                        st.session_state.profit_lock = st.session_state.result_tracker
                        st.session_state.result_tracker = 0.0
                        st.session_state.bet_amount = st.session_state.base_amount
                        st.success(f"New profit lock achieved: ${st.session_state.profit_lock:.2f}! Bankroll reset.")
                        self.update_display()
                        return
                    if st.session_state.consecutive_wins >= 2:
                        st.session_state.bet_amount = max(st.session_state.base_amount, st.session_state.bet_amount - st.session_state.base_amount)
                else:
                    st.session_state.result_tracker -= effective_bet
                    st.session_state.stats['losses'] += 1
                    st.session_state.consecutive_losses += 1
                    st.session_state.consecutive_wins = 0
                    if st.session_state.consecutive_losses >= 3:
                        st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount * 2)
                    elif st.session_state.streak_type:
                        st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)
                    else:
                        st.session_state.bet_amount = min(5 * st.session_state.base_amount, st.session_state.bet_amount + st.session_state.base_amount)

        if st.session_state.result_tracker <= -10 * st.session_state.base_amount:
            st.warning("Loss limit reached. Resetting to resume betting.")
            st.session_state.bet_amount = st.session_state.base_amount
            st.session_state.next_prediction = "Player" if result == 'B' else "Banker" if result == 'P' else random.choice(["Player", "Banker"])
            self.update_display()
            return

        st.session_state.previous_result = result
        self.update_display()

    def undo(self):
        if not st.session_state.state_history:
            st.error("No actions to undo.")
            return

        last_state = st.session_state.state_history.pop()
        st.session_state.pair_types = last_state['pair_types']
        st.session_state.previous_result = last_state['previous_result']
        st.session_state.result_tracker = last_state['result_tracker']
        st.session_state.profit_lock = last_state['profit_lock']
        st.session_state.bet_amount = last_state['bet_amount']
        st.session_state.current_dominance = last_state['current_dominance']
        st.session_state.next_prediction = last_state['next_prediction']
        st.session_state.consecutive_wins = last_state['consecutive_wins']
        st.session_state.consecutive_losses = last_state['consecutive_losses']
        st.session_state.streak_type = last_state['streak_type']
        st.session_state.stats = last_state['stats']
        self.update_display()
        st.success("Last action undone.")

    def update_display(self):
        # Update the displayed values using session state
        st.session_state.unit_info = f"Bet Amount: {'No Bet' if st.session_state.bet_amount == 0 else f'${st.session_state.bet_amount:.2f}'}"
        st.session_state.profit = f"Bankroll: ${st.session_state.result_tracker:.2f}"
        st.session_state.profit_lock = f"Profit Lock: ${st.session_state.profit_lock:.2f}"
        st.session_state.prediction = f"Bet: {st.session_state.next_prediction}"
        st.session_state.streak = f"Streak: {st.session_state.streak_type if st.session_state.streak_type else 'None'}"

        total_games = st.session_state.stats['wins'] + st.session_state.stats['losses']
        win_rate = (st.session_state.stats['wins'] / total_games * 100) if total_games > 0 else 0
        avg_streak = sum(st.session_state.stats['streaks']) / len(st.session_state.stats['streaks']) if st.session_state.stats['streaks'] else 0
        st.session_state.stats_display = f"Win Rate: {win_rate:.1f}% | Avg Streak: {avg_streak:.1f} | Patterns: Odd: {st.session_state.stats['odd_pairs']}, Even: {st.session_state.stats['even_pairs']}"

        # Update deal history
        history_text = ""
        for i, pair in enumerate(st.session_state.pair_types[-100:], 1):
            pair_type = "Even" if pair[0] == pair[1] else "Odd"
            history_text += f"{pair} ({pair_type})\n"
        st.session_state.history_text = history_text

    def simulate_games(self, num_games=100):
        outcomes = ['P', 'B', 'T']
        weights = [0.446, 0.458, 0.096]
        for _ in range(num_games):
            result = random.choices(outcomes, weights)[0]
            self.record_result(result)
        st.success(f"Simulated {num_games} games. Check stats for results.")

def main():
    app = BaccaratPredictor()

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #2C2F33;
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        background-color: #7289DA;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #99AAB5;
    }
    .stTextInput>div>input {
        background-color: #23272A;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px;
    }
    .stTextArea textarea {
        background-color: #23272A;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 14px;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title">Enhanced Dominant Pairs Baccarat Predictor</div>', unsafe_allow_html=True)

    # Base amount input
    col1, col2 = st.columns([2, 1])
    with col1:
        # Use a unique key and rely on session state for persistence
        base_amount = st.text_input("Base Amount ($1-$100)", value="", placeholder=str(st.session_state.base_amount), key="base_amount_input")
    with col2:
        if st.button("Set Amount"):
            if base_amount:
                app.set_base_amount(base_amount)
            else:
                st.error("Please enter a value for the base amount.")

    # Info display
    st.markdown(f"**{st.session_state.get('unit_info', 'Bet Amount: $10.00')}**")
    st.markdown(f"**{st.session_state.get('profit', 'Bankroll: $0.00')}**")
    st.markdown(f"**{st.session_state.get('profit_lock', 'Profit Lock: $0.00')}**")
    st.markdown(f"**{st.session_state.get('prediction', 'Bet: N/A')}**")
    st.markdown(f"**{st.session_state.get('streak', 'Streak: None')}**")

    # Stats display
    st.markdown(f"**{st.session_state.get('stats_display', 'Win Rate: 0% | Avg Streak: 0 | Patterns: Odd: 0, Even: 0')}**")

    # Action buttons
    st.markdown('<div class="section-title">Record Result</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Player"):
            app.record_result('P')
    with col2:
        if st.button("Banker"):
            app.record_result('B')
    with col3:
        if st.button("Tie"):
            app.record_result('T')
    with col4:
        if st.button("Undo"):
            app.undo()

    # Deal history
    st.markdown('<div class="section-title">Deal History</div>', unsafe_allow_html=True)
    st.text_area("", value=st.session_state.get('history_text', ''), height=200, key="history", disabled=True)

    # Session controls
    st.markdown('<div class="section-title">Session Controls</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Reset Bet"):
            app.reset_betting()
    with col2:
        if st.button("Reset Session"):
            app.reset_all()
    with col3:
        if st.button("New Session"):
            app.new_session()
    with col4:
        if st.button("Simulate (100 games)"):
            app.simulate_games(100)

if __name__ == "__main__":
    main()
