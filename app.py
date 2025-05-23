import streamlit as st
from collections import Counter

class BaccaratAI:
    def __init__(self, initial_bankroll, base_bet):
        self.shoe_history = []  # Store outcomes: 'P' (Player), 'B' (Banker), 'T' (Tie)
        self.max_shoe_size = 80  # Typical shoe size
        self.bankroll = initial_bankroll
        self.base_bet = base_bet
        self.bead_road = [[] for _ in range(6)]  # 6-row bead road grid

    def add_outcome(self, outcome):
        """Add a hand outcome to the shoe history and bead road."""
        if outcome not in ['P', 'B', 'T']:
            raise ValueError("Outcome must be 'P' (Player), 'B' (Banker), or 'T' (Tie)")
        self.shoe_history.append(outcome)
        
        # Add to bead road (fill column by column, top to bottom)
        col = len(self.shoe_history) - 1
        row = col % 6
        col //= 6
        while len(self.bead_road[row]) <= col:
            self.bead_road[row].append(' ')
        self.bead_road[row][col] = outcome

    def display_bead_road(self):
        """Return the bead road as a string."""
        output = "Bead Road:\n"
        max_cols = max(len(row) for row in self.bead_road)
        for row in self.bead_road:
            output += "| " + " ".join(row + [' '] * (max_cols - len(row))) + " |\n"
        return output

    def analyze_shoe(self):
        """Analyze the shoe for patterns and statistics."""
        if not self.shoe_history:
            return {"player_count": 0, "banker_count": 0, "tie_count": 0, "recent_pattern": []}

        counts = Counter(self.shoe_history)
        total_hands = len(self.shoe_history)
        recent_pattern = self.shoe_history[-5:] if len(self.shoe_history) >= 5 else self.shoe_history

        return {
            "player_count": counts.get('P', 0) / total_hands,
            "banker_count": counts.get('B', 0) / total_hands,
            "tie_count": counts.get('T', 0) / total_hands,
            "recent_pattern": recent_pattern
        }

    def suggest_bet(self):
        """Suggest the next bet based on shoe analysis."""
        analysis = self.analyze_shoe()
        
        if not self.shoe_history:
            return "B"  # Default to Banker for first hand

        recent_counts = Counter(analysis["recent_pattern"])
        most_common_recent = max(recent_counts, key=recent_counts.get, default="B")

        if len(analysis["recent_pattern"]) >= 3 and len(set(analysis["recent_pattern"][-3:])) == 1:
            return analysis["recent_pattern"][-1]

        if abs(analysis["player_count"] - analysis["banker_count"]) < 0.1:
            return "B"
        return "P" if analysis["player_count"] > analysis["banker_count"] else "B"

    def update_bankroll(self, bet, outcome):
        """Update bankroll based on bet and outcome."""
        if bet == outcome:
            if bet == 'P':
                self.bankroll += self.base_bet
            elif bet == 'B':
                self.bankroll += self.base_bet * 0.95  # 5% commission on Banker
        elif outcome != 'T':
            self.bankroll -= self.base_bet
        return self.bankroll

def main():
    st.title("Baccarat AI Bet Selector")

    # Initialize session state
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
        st.session_state.baccarat = None
        st.session_state.game_over = False
        st.session_state.suggested_bet = ''

    # Input form for bankroll and base bet
    if not st.session_state.game_started and not st.session_state.game_over:
        bankroll = st.number_input("Initial Bankroll ($)", min_value=0.0, step=10.0, value=1000.0)
        base_bet = st.number_input("Base Bet ($)", min_value=0.0, step=1.0, value=10.0)
        
        if st.button("Start Game"):
            if base_bet <= 0 or bankroll <= 0 or base_bet > bankroll:
                st.error("Bankroll and base bet must be positive, and base bet must not exceed bankroll.")
            else:
                st.session_state.baccarat = BaccaratAI(bankroll, base_bet)
                st.session_state.game_started = True
                st.session_state.suggested_bet = st.session_state.baccarat.suggest_bet()
                st.session_state.game_over = False

    # Game interface
    if st.session_state.game_started and not st.session_state.game_over:
        baccarat = st.session_state.baccarat
        st.write(f"**Bankroll**: ${baccarat.bankroll:.2f}")
        st.write(f"**Base Bet**: ${baccarat.base_bet:.2f}")
        st.write(f"**Hand**: {len(baccarat.shoe_history) + 1}")
        st.write(f"**AI Suggests**: {'Player' if st.session_state.suggested_bet == 'P' else 'Banker'}")

        # Display bead road
        st.markdown("### Bead Road")
        st.text(baccarat.display_bead_road())

        # Outcome buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Player"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'P')
                baccarat.add_outcome('P')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet = baccarat.suggest_bet()
        with col2:
            if st.button("Banker"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'B')
                baccarat.add_outcome('B')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet = baccarat.suggest_bet()
        with col3:
            if st.button("Tie"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'T')
                baccarat.add_outcome('T')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet = baccarat.suggest_bet()
        with col4:
            if st.button("Quit"):
                st.session_state.game_over = True

    # Game over screen
    if st.session_state.game_over:
        st.markdown("## Game Over!")
        baccarat = st.session_state.baccarat
        st.write(f"**Final Bankroll**: ${baccarat.bankroll:.2f}")
        st.markdown("### Final Shoe Analysis")
        analysis = baccarat.analyze_shoe()
        st.write(f"Player Frequency: {analysis['player_count']:.2%}")
        st.write(f"Banker Frequency: {analysis['banker_count']:.2%}")
        st.write(f"Tie Frequency: {analysis['tie_count']:.2%}")
        st.text(baccarat.display_bead_road())
        
        if st.button("Start New Game"):
            st.session_state.game_started = False
            st.session_state.baccarat = None
            st.session_state.game_over = False
            st.session_state.suggested_bet = ''
            st.experimental_rerun()

if __name__ == "__main__":
    main()
