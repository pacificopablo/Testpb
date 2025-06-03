import streamlit as st

def predict_bet(history):
    """
    Predict the next bet based on the Smart Fusion 5 Baccarat System rules.
    Input: List of up to 5 outcomes ('B' for Banker, 'P' for Player).
    Output: Tuple of (result message, predicted bet or None).
    """
    # Validate input: filter valid outcomes (B or P)
    history = [x.upper() for x in history if x.upper() in ['B', 'P']]
    if len(history) < 2:
        return "Please enter at least 2 valid outcomes (B or P).", None

    # Rule 5: 2-in-a-Row Catcher
    if len(history) >= 2 and history[-1] == history[-2] and (len(history) < 3 or history[-1] != history[-3]):
        return f"Bet: {history[-1]} (2-in-a-Row Catcher)", history[-1]

    # Rule 4: Dragon Slayer (3 or more in a row)
    if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
        bet = 'P' if history[-1] == 'B' else 'B'
        return f"Bet: {bet} (Dragon Slayer - 3+ streak)", bet

    # Rule 3: Zigzag Pattern Break
    if len(history) >= 4 and history[-1] != history[-2] and history[-2] != history[-3] and history[-3] != history[-4]:
        return f"Bet: {history[-1]} (Zigzag Break)", history[-1]

    # Rule 2: OTB4L Check
    if len(history) >= 2 and history[-1] != history[-2]:
        bet = 'P' if history[-2] == 'B' else 'B'
        return f"Bet: {bet} (OTB4L)", bet

    # Rule 1: Banker Bias (Default)
    return "Bet: B (Banker Bias - Default)", 'B'

# Streamlit app layout
st.title("Smart Fusion 5 Baccarat Predictor")
st.markdown("""
Enter the last 5 outcomes from your live dealer Baccarat game (B for Banker, P for Player, ignore Ties).
The system will predict the next bet using the Smart Fusion 5 rules. Use flat betting (1 unit) and avoid Tie bets.
""")

# Input fields for the last 5 outcomes
st.subheader("Enter Last 5 Outcomes")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    input1 = st.text_input("1", max_chars=1, key="input1", placeholder="B/P")
with col2:
    input2 = st.text_input("2", max_chars=1, key="input2", placeholder="B/P")
with col3:
    input3 = st.text_input("3", max_chars=1, key="input3", placeholder="B/P")
with col4:
    input4 = st.text_input("4", max_chars=1, key="input4", placeholder="B/P")
with col5:
    input5 = st.text_input("5", max_chars=1, key="input5", placeholder="B/P")

# Collect inputs
history = [input1, input2, input3, input4, input5]

# Button to predict
if st.button("Predict Next Bet"):
    result, _ = predict_bet(history)
    st.markdown(f"**{result}**")
    valid_history = [x.upper() for x in history if x.upper() in ['B', 'P']]
    if valid_history:
        st.write(f"**Current Sequence**: {'-'.join(valid_history)}")
    else:
        st.write("**Current Sequence**: None")

# Additional instructions
st.markdown("""
### Betting Guidelines
- **Flat Bet**: Always bet 1 unit.
- **Stop-Loss**: Stop if down 6 units.
- **Win Goal**: Optional exit after +5 units.
- **Do Not**: Bet on Tie or use Martingale progression.
- **Unsure?**: Sit out the hand if the sequence is unclear.
""")
