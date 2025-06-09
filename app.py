import random
from collections import deque
from copy import deepcopy

class BaccaratMoneyManager:
    def __init__(self):
        self.base_amount = 10.0  # Base betting unit in dollars
        self.bet_amount = self.base_amount  # Current bet amount
        self.result_tracker = 0.0  # Current bankroll
        self.profit_lock = 0.0  # Secured profit
        self.previous_result = None  # Last game result
        self.state_history = deque(maxlen=100)  # Store states for undo
        self.stats = {'wins': 0, 'losses': 0, 'bet_history': []}  # Track outcomes
        self.consecutive_wins = 0  # Track win streak
        self.consecutive_losses = 0  # Track loss streak
        self.is_streak = False  # Streak detection flag
        self.next_prediction = "N/A"  # Next bet (Player, Banker)
        print(f"Baccarat Money Manager initialized. Base amount: ${self.base_amount:.2f}")

    def set_base_amount(self, amount):
        """Set the base betting amount ($1-$100)."""
        try:
            amount = float(amount)
            if 1 <= amount <= 100:
                self.base_amount = amount
                self.bet_amount = self.base_amount
                self.update_display()
                print(f"Base amount set to ${self.base_amount:.2f}")
            else:
                print("Base amount must be between $1 and $100.")
        except (ValueError, TypeError):
            print("Please enter a valid number.")

    def reset_betting(self):
        """Reset betting progression to initial state."""
        if self.result_tracker <= -10 * self.base_amount:
            print("Stop-loss reached. Resetting to resume betting.")
        self.result_tracker = 0.0 if self.result_tracker >= 0 else self.result_tracker
        self.bet_amount = self.base_amount
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.is_streak = False
        self.next_prediction = "Player" if self.previous_result == 'B' else "Banker" if self.previous_result == 'P' else "N/A"
        self.update_display()
        print("Betting reset.")

    def reset_all(self):
        """Reset all session data to initial values."""
        self.base_amount = 10.0
        self.bet_amount = self.base_amount
        self.result_tracker = 0.0
        self.profit_lock = 0.0
        self.previous_result = None
        self.state_history.clear()
        self.stats = {'wins': 0, 'losses': 0, 'bet_history': []}
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.is_streak = False
        self.next_prediction = "N/A"
        self.update_display()
        print("All session data reset, profit lock reset.")

    def record_result(self, result):
        """Record a game result and update money management."""
        # Save current state for undo
        state = {
            'base_amount': self.base_amount,
            'bet_amount': self.bet_amount,
            'result_tracker': self.result_tracker,
            'profit_lock': self.profit_lock,
            'previous_result': self.previous_result,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'is_streak': self.is_streak,
            'next_prediction': self.next_prediction,
            'stats': deepcopy(self.stats)
        }
        self.state_history.append(state)

        # Handle first result
        if self.previous_result is None:
            self.previous_result = result
            self.next_prediction = "N/A"
            self.bet_amount = self.base_amount
            self.update_display()
            print("First result recorded. Waiting for more results.")
            return

        # Update streak and prediction
        if self.previous_result == result:
            self.is_streak = True
            self.next_prediction = "Player" if result == 'P' else "Banker"
            self.bet_amount = 2 * self.base_amount  # Double bet during streak
        else:
            self.is_streak = False
            self.next_prediction = "Player" if result == 'B' else "Banker"
            self.bet_amount = self.base_amount

        # Evaluate bet outcome after 5 results
        if len(self.stats['bet_history']) >= 5 and self.next_prediction != "N/A":
            effective_bet = min(5 * self.base_amount, self.bet_amount)
            outcome = ""
            if self.next_prediction == result:
                win_amount = effective_bet if result == 'P' else effective_bet * 0.95  # 5% Banker commission
                self.result_tracker += win_amount
                self.stats['wins'] += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                outcome = f"Won ${win_amount:.2f}"
                print(f"Bet won: +${win_amount:.2f}")
                if self.result_tracker > self.profit_lock:
                    self.profit_lock = self.result_tracker
                    self.result_tracker = 0.0
                    self.bet_amount = self.base_amount
                    print(f"New profit lock achieved: ${self.profit_lock:.2f}! Bankroll reset.")
                    self.update_display()
                    return
                self.bet_amount = self.base_amount  # Reset after any win
            else:
                self.result_tracker -= effective_bet
                self.stats['losses'] += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                outcome = f"Lost ${effective_bet:.2f}"
                print(f"Bet lost: -${effective_bet:.2f}")
                if self.consecutive_losses >= 2:
                    self.bet_amount = self.base_amount  # Reset after two losses
                else:
                    self.bet_amount = min(5 * self.base_amount, self.bet_amount * 2)

            self.stats['bet_history'].append({
                'prediction': self.next_prediction,
                'result': result,
                'bet_amount': effective_bet,
                'outcome': outcome
            })

        # Check stop-loss
        if self.result_tracker <= -10 * self.base_amount:
            print("Loss limit reached. Resetting to resume betting.")
            self.bet_amount = self.base_amount
            self.next_prediction = "Player" if result == 'B' else "Banker"
            self.update_display()
            return

        self.previous_result = result
        self.update_display()

    def undo(self):
        """Undo the last action by restoring the previous state."""
        try:
            if not self.state_history:
                print("No actions to undo.")
                return
            last_state = self.state_history.pop()
            self.base_amount = last_state['base_amount']
            self.bet_amount = last_state['bet_amount']
            self.result_tracker = last_state['result_tracker']
            self.profit_lock = last_state['profit_lock']
            self.previous_result = last_state['previous_result']
            self.consecutive_wins = last_state['consecutive_wins']
            self.consecutive_losses = last_state['consecutive_losses']
            self.is_streak = last_state['is_streak']
            self.next_prediction = last_state['next_prediction']
            self.stats = last_state['stats']
            self.update_display()
            print("Last action undone.")
        except Exception as e:
            print(f"Error during undo: {e}")

    def update_display(self):
        """Display current state of the game."""
        total_games = self.stats['wins'] + self.stats['losses']
        win_rate = self.stats['wins'] / total_games * 100 if total_games > 0 else 0.0
        print("\n=== Current State ===")
        print(f"Bet Amount: {'No Bet' if self.bet_amount == 0 else f'${self.bet_amount:.2f}'}")
        print(f"Bankroll: ${self.result_tracker:.2f}")
        print(f"Profit Lock: ${self.profit_lock:.2f}")
        print(f"Next Bet: {self.next_prediction}")
        print(f"Stats: Wins: {self.stats['wins']}, Losses: {self.stats['losses']}, Win Rate: {win_rate:.1f}%")
        if self.stats['bet_history']:
            print("Recent Bets:")
            for bet in self.stats['bet_history'][-3:]:
                print(f"  Bet: {bet['prediction']}, Result: {bet['result']}, Amount: ${bet['bet_amount']:.2f}, Outcome: {bet['outcome']}")
        print("==================\n")

    def simulate_games(self, num_games=100):
        """Simulate games to test money management."""
        if not isinstance(num_games, int) or num_games <= 0:
            print("Number of games must be a positive integer.")
            return
        outcomes = ['P', 'B']
        weights = [0.493362, 0.506638]  # Normalized probabilities
        for _ in range(num_games):
            result = random.choices(outcomes, weights)[0]
            self.record_result(result)
        print(f"Simulated {num_games} games. Check stats for results.")

def main():
    manager = BaccaratMoneyManager()
    while True:
        print("\nOptions:")
        print("1. Record Result (P/B)")
        print("2. Set Base Amount")
        print("3. Reset Betting")
        print("4. Reset Session")
        print("5. Undo Last Action")
        print("6. Simulate 100 Games")
        print("7. Exit")
        choice = input("Enter choice (1-7): ").strip()
        if not choice:
            print("Input cannot be empty. Try again.")
            continue

        if choice == '1':
            result = input("Enter result (P/B): ").strip().upper()
            if not result:
                print("Input cannot be empty. Use P or B.")
            elif result in ['P', 'B']:
                manager.record_result(result)
            else:
                print("Invalid result. Use P or B.")
        elif choice == '2':
            amount = input("Enter base amount ($1-$100): ").strip()
            manager.set_base_amount(amount)
        elif choice == '3':
            manager.reset_betting()
        elif choice == '4':
            manager.reset_all()
        elif choice == '5':
            manager.undo()
        elif choice == '6':
            manager.simulate_games()
        elif choice == '7':
            print("Exiting Baccarat Money Manager.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
