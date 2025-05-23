import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class BaccaratAI:
    def __init__(self, initial_bankroll, base_bet):
        self.shoe_history = []  # Store outcomes: 'P' (Player), 'B' (Banker), 'T' (Tie)
        self.bet_history = []   # Store AI bets and outcomes
        self.max_shoe_size = 80  # Typical shoe size
        self.bankroll = initial_bankroll
        self.base_bet = base_bet
        self.current_bet = base_bet
        self.max_bet = initial_bankroll * 0.1  # Max bet is 10% of initial bankroll
        self.bead_road = [[] for _ in range(6)]  # 6-row bead road grid
        self.bankroll_history = [initial_bankroll]  # Track bankroll over time
        self.loss_streak = 0  # Track consecutive losses for progressive betting

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
        """Return the bead road as a string with color-coded outcomes."""
        output = "Bead Road:\n"
        max_cols = max(len(row) for row in self.bead_road)
        for row in self.bead_road:
            row_str = "| "
            for outcome in row + [' '] * (max_cols - len(row)):
                if outcome == 'P':
                    row_str += "🟢 "  # Green for Player
                elif outcome == 'B':
                    row_str += "🔴 "  # Red for Banker
                elif outcome == 'T':
                    row_str += "🟡 "  # Yellow for Tie
                else:
                    row_str += "  "
            row_str += "|"
            output += row_str + "\n"
        return output

    def analyze_shoe(self):
        """Analyze the shoe for patterns and statistics."""
        if not self.shoe_history:
            return {
                "player_count": 0,
                "banker_count": 0,
                "tie_count": 0,
                "recent_pattern": [],
                "streak_length": 0,
                "is_streak": False,
                "is_chop": False
            }

        counts = Counter(self.shoe_history)
        total_hands = len(self.shoe_history)
        recent_pattern = self.shoe_history[-5:] if len(self.shoe_history) >= 5 else self.shoe_history

        # Detect streaks and chops
        streak_length = 1
        is_streak = False
        is_chop = False
        if len(self.shoe_history) >= 3:
            last_three = self.shoe_history[-3:]
            if len(set(last_three)) == 1 and last_three[0] in ['P', 'B']:
                is_streak = True
                # Count streak length
                for i in range(len(self.shoe_history) - 4, -1, -1):
                    if self.shoe_history[i] == last_three[0]:
                        streak_length += 1
                    else:
                        break
            elif all(self.shoe_history[i] != self.shoe_history[i-1] for i in range(-1, -3, -1)):
                is_chop = True

        return {
            "player_count": counts.get('P', 0) / total_hands,
            "banker_count": counts.get('B', 0) / total_hands,
            "tie_count": counts.get('T', 0) / total_hands,
            "recent_pattern": recent_pattern,
            "streak_length": streak_length,
            "is_streak": is_streak,
            "is_chop": is_chop
        }

    def suggest_bet(self):
        """Suggest the next bet with confidence score."""
        analysis = self.analyze_shoe()
        
        if not self.shoe_history:
            return "B", 0.5  # Default to Banker with neutral confidence

        confidence = 0.5  # Baseline confidence
        recent_counts = Counter(analysis["recent_pattern"])
        most_common_recent = max(recent_counts, key=recent_counts.get, default="B")

        if analysis["is_streak"] and analysis["streak_length"] >= 3:
            # Follow the streak
            confidence = min(0.8, 0.5 + 0.1 * analysis["streak_length"])
            return analysis["recent_pattern"][-1], confidence
        elif analysis["is_chop"]:
            # Bet opposite of the last outcome for chop pattern
            confidence = 0.65
            return "P" if self.shoe_history[-1] == 'B' else "B", confidence
        elif abs(analysis["player_count"] - analysis["banker_count"]) < 0.1:
            # Close frequencies, default to Banker due to lower house edge
            return "B", 0.55
        else:
            # Bet on the dominant side
            confidence = 0.6
            return "P" if analysis["player_count"] > analysis["banker_count"] else "B", confidence

    def update_bankroll(self, bet, outcome):
        """Update bankroll based on bet and outcome, and adjust bet size."""
        win = bet == outcome
        if win:
            if bet == 'P':
                self.bankroll += self.current_bet
            elif bet == 'B':
                self.bankroll += self.current_bet * 0.95  # 5% commission on Banker
            self.loss_streak = 0
            self.current_bet = self.base_bet  # Reset bet after win
        elif outcome != 'T':
            self.bankroll -= self.current_bet
            self.loss_streak += 1
            # Double bet after loss, up to max_bet
            self.current_bet = min(self.current_bet * 2, self.max_bet)
            if self.current_bet > self.bankroll:
                self.current_bet = self.bankroll  # Cap at available bankroll
        self.bankroll_history.append(self.bankroll)
        self.bet_history.append((bet, outcome, win, self.current_bet))
        return self.bankroll

    def plot_bankroll(self):
        """Generate a plot of bankroll over time."""
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(self.bankroll_history)), self.bankroll_history, 'b-', label='Bankroll')
        plt.title('Bankroll Over Time')
        plt.xlabel('Hand Number')
        plt.ylabel('Bankroll ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig('bankroll_plot.png')
        plt.close()

def main():
    st.title("Baccarat AI Bet Selector")

    # Initialize session state
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
        st.session_state.baccarat = None
        st.session_state.game_over = False
        st.session_state.suggested_bet = ''
        st.session_state.confidence = 0.5

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
                st.session_state.suggested_bet, st.session_state.confidence = st.session_state.baccarat.suggest_bet()
                st.session_state.game_over = False

    # Game interface
    if st.session_state.game_started and not st.session_state.game_over:
        baccarat = st.session_state.baccarat
        st.write(f"**Bankroll**: ${baccarat.bankroll:.2f}")
        st.write(f"**Current Bet**: ${baccarat.current_bet:.2f}")
        st.write(f"**Hand**: {len(baccarat.shoe_history) + 1}")
        st.write(f"**AI Suggests**: {'Player' if st.session_state.suggested_bet == 'P' else 'Banker'} "
                 f"(Confidence: {st.session_state.confidence:.2%})")

        # Display bead road
        st.markdown("### Bead Road")
        st.text(baccarat.display_bead_road())

        # Plot bankroll
        baccarat.plot_bankroll()
        st.image('bankroll_plot.png', caption='Bankroll Trend')

        # Display recent bet history
        st.markdown("### Recent Bet History")
        if baccarat.bet_history:
            for i, (bet, outcome, win, bet_amount) in enumerate(baccarat.bet_history[-5:], 1):
                result = "Win" if win else "Loss" if outcome != 'T' else "Tie"
                st.write(f"Hand {len(baccarat.shoe_history) - len(baccarat.bet_history) + i}: "
                        f"Bet {'Player' if bet == 'P' else 'Banker'} (${bet_amount:.2f}), "
                        f"Outcome: {'Player' if outcome == 'P' else 'Banker' if outcome == 'B' else 'Tie'}, "
                        f"Result: {result}")

        # Outcome buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Player"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'P')
                baccarat.add_outcome('P')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet, st.session_state.confidence = baccarat.suggest_bet()
        with col2:
            if st.button("Banker"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'B')
                baccarat.add_outcome('B')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet, st.session_state.confidence = baccarat.suggest_bet()
        with col3:
            if st.button("Tie"):
                new_bankroll = baccarat.update_bankroll(st.session_state.suggested_bet, 'T')
                baccarat.add_outcome('T')
                if new_bankroll < baccarat.base_bet or len(baccarat.shoe_history) >= baccarat.max_shoe_size:
                    st.session_state.game_over = True
                else:
                    st.session_state.suggested_bet, st.session_state.confidence = baccarat.suggest_bet()
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
        st.image('bankroll_plot.png', caption='Final Bankroll Trend')
        
        if st.button("Start New Game"):
            st.session_state.game_started = False
            st.session_state.baccarat = None
            st.session_state.game_over = False
            st.session_state.suggested_bet = ''
            st.session_state.confidence = 0.5
            st.experimental_rerun()

if __name__ == "__main__":
    main()
