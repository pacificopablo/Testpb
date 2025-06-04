
from collections import deque
import tkinter as tk
from tkinter import messagebox, scrolledtext

def get_prediction(history, full_history, bet_history):
    if len(full_history) < 2:
        return "Default: Bet on Banker (Confidence: 50.00%)"

    # Initialize transition counts
    transitions = {'B': {'B': 0.1, 'P': 0.1}, 'P': {'B': 0.1, 'P': 0.1}}  # Pseudocounts for smoothing

    # Count transitions from full history
    for i in range(len(full_history) - 1):
        current = full_history[i]
        next_outcome = full_history[i + 1]
        transitions[current][next_outcome] += 1

    # Get last outcome from history (limited to 5)
    last_outcome = list(history)[-1] if history else full_history[-1]

    # Calculate transition probabilities
    total = transitions[last_outcome]['B'] + transitions[last_outcome]['P']
    prob_b = transitions[last_outcome]['B'] / total
    prob_p = transitions[last_outcome]['P'] / total

    # Select bet with confidence threshold
    confidence_threshold = 0.1
    if prob_b > prob_p + confidence_threshold:
        return f"Bet: Banker (Confidence: {prob_b:.2%})"
    elif prob_p > prob_b + confidence_threshold:
        return f"Bet: Player (Confidence: {prob_p:.2%})"
    return f"Default: Bet on Banker (Confidence: {max(prob_b, prob_p):.2%})"

def add_result(state, result, text_area):
    bet_amount = 0
    bet_selection = None
    bet_outcome = None
    transition_probs = {'B': 0.0, 'P': 0.0}

    # Process pending bet outcome
    if state['pending_bet']:
        bet_amount, bet_selection = state['pending_bet']
        state['bets_placed'] += 1
        if result == bet_selection:
            state['bankroll'] += bet_amount * (0.95 if bet_selection == 'B' else 1.0)
            state['bets_won'] += 1
            bet_outcome = 'win'
            state['t3_results'].append('W')
            state['t5_results'].append('W')
        else:
            state['bankroll'] -= bet_amount
            bet_outcome = 'loss'
            state['t3_results'].append('L')
            state['t5_results'].append('L')

        # Evaluate T3 level after three rounds
        if len(state['t3_results']) == 3:
            wins = state['t3_results'].count('W')
            losses = state['t3_results'].count('L')
            if wins > losses:
                state['t3_level'] += 1  # More wins: move forward
            elif losses > wins:
                state['t3_level'] -= 1  # More losses: move back
            state['t3_results'] = []  # Reset for next three rounds

        # Evaluate T5 level after five rounds
        if len(state['t5_results']) == 5:
            wins = state['t5_results'].count('W')
            losses = state['t5_results'].count('L')
            if wins > losses:
                state['t5_level'] -= 1  # More wins: go back
            elif losses > wins:
                state['t5_level'] += 1  # More losses: go forward
            state['t5_results'] = []  # Reset for next five rounds

        state['pending_bet'] = None

    state['history'].append(result)
    state['full_history'].append(result)

    # Generate new prediction and bet
    if len(state['history']) >= 5:
        prediction = get_prediction(state['history'], state['full_history'], state['bet_history'])
        # Extract probabilities for bet_history
        if "Confidence" in prediction:
            conf = float(prediction.split("Confidence: ")[1].strip("%)")) / 100
            if "Banker" in prediction:
                transition_probs['B'] = conf
                transition_probs['P'] = 1.0 - conf
            elif "Player" in prediction:
                transition_probs['P'] = conf
                transition_probs['B'] = 1.0 - conf
        bet_selection = 'B' if "Banker" in prediction else 'P' if "Player" in prediction else None
        if bet_selection:
            bet_amount = state['base_bet'] * abs(state['t3_level'])  # Use T3 level for wager
            if bet_amount > state['bankroll']:
                state['pending_bet'] = None
                state['prediction'] = f"{prediction} | Skip betting (bet ${bet_amount:.2f} exceeds bankroll)."
            else:
                state['pending_bet'] = (bet_amount, bet_selection)
                state['prediction'] = f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {state['t3_level']}) | {prediction}"
        else:
            state['pending_bet'] = None
            state['prediction'] = "No valid bet selection."
    else:
        state['pending_bet'] = None
        state['prediction'] = f"Need {5 - len(state['history'])} more results for prediction."

    state['bet_history'].append((result, bet_amount, bet_selection, bet_outcome, state['t3_level'], state['t3_results'][:], state['t5_level'], state['t5_results'][:], transition_probs))
    update_display(state, text_area)

def update_display(state, text_area):
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, f"=== Baccarat Predictor ===\n")
    text_area.insert(tk.END, f"History: {''.join(state['history']) if state['history'] else 'No results yet'}\n")
    text_area.insert(tk.END, f"Bankroll: ${state['bankroll']:.2f}\n")
    text_area.insert(tk.END, f"Base Bet: ${state['base_bet']:.2f}\n")
    text_area.insert(tk.END, f"Session: {state['bets_placed']} bets, {state['bets_won']} wins\n")
    text_area.insert(tk.END, f"T3 Status: Level {state['t3_level']}, Results: {state['t3_results']}\n")
    text_area.insert(tk.END, f"T5 Status: Level {state['t5_level']}, Results: {state['t5_results']}\n")
    text_area.insert(tk.END, f"Prediction: {state['prediction']}\n")
    text_area.insert(tk.END, "\nDebug State:\n")
    text_area.insert(tk.END, f"Full History: {state['full_history']}\n")
    text_area.insert(tk.END, f"Bet History: {state['bet_history']}\n")
    text_area.insert(tk.END, f"Pending Bet: {state['pending_bet']}\n")

def initialize_state():
    return {
        'history': deque(maxlen=5),
        'full_history': [],  # Store all outcomes
        'prediction': "",
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_bankroll': 0.0,
        't3_level': 1,
        't3_results': [],
        't5_level': 1,  # Initialize T5 level
        't5_results': [],  # Track T5 outcomes
        'bet_history': [],
        'pending_bet': None,
        'bets_placed': 0,
        'bets_won': 0,
        'session_active': False
    }

def start_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame):
    try:
        bankroll = float(bankroll_entry.get())
        base_bet = float(base_bet_entry.get())
        if bankroll <= 0 or base_bet <= 0:
            messagebox.showerror("Error", "Bankroll and base bet must be positive numbers.")
        elif base_bet > bankroll * 0.05:
            messagebox.showerror("Error", "Base bet cannot exceed 5% of bankroll.")
        else:
            state.update({
                'bankroll': bankroll,
                'base_bet': base_bet,
                'initial_bankroll': bankroll,
                'history': deque(maxlen=5),
                'full_history': [],
                't3_results': [],
                't5_results': [],
                'bet_history': [],
                'pending_bet': None,
                'bets_placed': 0,
                'bets_won': 0,
                't3_level': 1,
                't5_level': 1,
                'prediction': "",
                'session_active': True
            })
            bankroll_entry.config(state='disabled')
            base_bet_entry.config(state='disabled')
            update_display(state, text_area)
            enable_game_buttons(button_frame)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

def reset_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame):
    state.update(initialize_state())
    bankroll_entry.config(state='normal')
    bankroll_entry.delete(0, tk.END)
    base_bet_entry.config(state='normal')
    base_bet_entry.delete(0, tk.END)
    update_display(state, text_area)
    disable_game_buttons(button_frame)

def undo_action(state, text_area):
    if not state['history']:
        messagebox.showwarning("Warning", "No results to undo.")
    else:
        state['history'].pop()
        state['full_history'].pop()
        if state['bet_history']:
            last_bet = state['bet_history'].pop()
            result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results, t5_level, t5_results, _ = last_bet
            if bet_amount > 0:
                state['bets_placed'] -= 1
                if bet_outcome == 'win':
                    state['bankroll'] -= bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                    state['bets_won'] -= 1
                elif bet_outcome == 'loss':
                    state['bankroll'] += bet_amount
                state['t3_level'] = t3_level
                state['t3_results'] = t3_results[:]
                state['t5_level'] = t5_level
                state['t5_results'] = t5_results[:]
        state['pending_bet'] = None
        state['prediction'] = ""
        update_display(state, text_area)

def check_session_end(state, text_area, root, button_frame):
    if state['session_active'] and state['bankroll'] < state['initial_bankroll'] * 0.8:
        messagebox.showerror("Session Ended", "Bankroll below 80% of initial. Session ended.")
        state['session_active'] = False
        reset_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame)
    elif state['session_active'] and state['bankroll'] >= state['initial_bankroll'] * 1.5:
        messagebox.showinfo("Session Ended", "Bankroll above 150% of initial. Session ended.")
        state['session_active'] = False
        reset_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame)

def enable_game_buttons(button_frame):
    for widget in button_frame.winfo_children():
        widget.config(state='normal')

def disable_game_buttons(button_frame):
    for widget in button_frame.winfo_children():
        widget.config(state='disabled')

def main():
    root = tk.Tk()
    root.title("Baccarat Predictor with T3 and T5")
    root.geometry("600x400")  # Adjusted for mobile screens

    state = initialize_state()

    # Setup frame
    setup_frame = tk.Frame(root)
    setup_frame.pack(pady=10)

    tk.Label(setup_frame, text="Initial Bankroll ($):").grid(row=0, column=0, padx=5)
    bankroll_entry = tk.Entry(setup_frame)
    bankroll_entry.grid(row=0, column=1, padx=5)

    tk.Label(setup_frame, text="Base Bet ($):").grid(row=1, column=0, padx=5)
    base_bet_entry = tk.Entry(setup_frame)
    base_bet_entry.grid(row=1, column=1, padx=5)

    start_button = tk.Button(setup_frame, text="Start Session", command=lambda: start_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame))
    start_button.grid(row=2, column=0, columnspan=2, pady=5)

    # Game buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Banker (B)", command=lambda: [add_result(state, 'B', text_area), check_session_end(state, text_area, root, button_frame)], state='disabled').grid(row=0, column=0, padx=5)
    tk.Button(button_frame, text="Player (P)", command=lambda: [add_result(state, 'P', text_area), check_session_end(state, text_area, root, button_frame)], state='disabled').grid(row=0, column=1, padx=5)
    tk.Button(button_frame, text="Undo (U)", command=lambda: undo_action(state, text_area), state='disabled').grid(row=0, column=2, padx=5)
    tk.Button(button_frame, text="Reset (R)", command=lambda: reset_session(state, bankroll_entry, base_bet_entry, text_area, root, button_frame), state='disabled').grid(row=0, column=3, padx=5)
    tk.Button(button_frame, text="Quit (Q)", command=root.quit).grid(row=0, column=4, padx=5)

    # Text display
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15)
    text_area.pack(pady=10)
    update_display(state, text_area)

    root.mainloop()

if __name__ == "__main__":
    main()
