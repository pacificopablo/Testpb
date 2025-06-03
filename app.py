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
        return "Please enter at least 2 valid outcomes (B or P) by pressing Player or Banker.", None

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

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit app layout
st.title("Smart Fusion 5 Baccarat Predictor")
st.markdown("""
Press the **Player** or **Banker** button to record each Baccarat game outcome (ignore Ties).
The system will predict the next bet using the Smart Fusion 5 rules, based on the last 5 outcomes.
Use flat betting (1 unit) and avoid Tie bets.
""")

# Buttons to input outcomes
st.subheader("Record Latest Outcome")
col1, col2 = st.columns(2)
with col1:
    if st.button("Player"):
        st.session_state.history.append('P')
        # Keep only the last 5 outcomes
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]
        # Predict next bet
        result, _ = predict_bet(st.session_state.history)
        st.session_state.result = result
with col2:
    if st.button("Banker"):
        st.session_state.history.append('B')
        # Keep only the last 5 outcomes
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]
        # Predict next bet
        result, _ = predict_bet(st.session_state.history)
        st.session_state.result = result

# Display current sequence and prediction
st.subheader("Current Status")
if st.session_state.history:
    st.write(f"**Current Sequence**: {'-'.join(st.session_state.history)}")
else:
    st.write("**Current Sequence**: None (press Player or Banker to start)")

if 'result' in st.session_state and st.session_state.result:
    st.markdown(f"**{st.session_state.result}**")
else:
    st.write("**Prediction**: None (need at least 2 outcomes)")

# Clear history button
if st.button("Clear History"):
    st.session_state.history = []
    if 'result' in st.session_state:
        del st.session_state.result
    st.write("**History cleared**. Start recording outcomes.")

# Additional instructions
st.markdown("""
### Betting Guidelines
- **Flat Bet**: Always bet 1 unit.
- **Stop-Loss**: Stop if down 6 units.
- **Win Goal**: Optional exit after +5 units.
- **Do Not**: Bet on Tie or use Martingale progression.
- **Unsure?**: Sit out the hand if the sequence is unclear.
""")
