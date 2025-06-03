import streamlit as st

def get_prediction(history):
    last5 = history[-5:]
    count = {'B': 0, 'P': 0}
    for r in last5:
        if r in ['B', 'P']:
            count[r] += 1
    
    if count['B'] >= 3:
        return "Banker (Bias)"
    if count['P'] >= 3:
        return "Player (Bias)"
    if ''.join(last5) == "BPBPB" or ''.join(last5) == "PBPBP":
        return f"Zigzag Breaker → Bet {last5[-1]}"
    if len(last5) >= 3 and all(v == last5[-1] for v in last5[-3:]):
        return f"Dragon Slayer → Bet {'Player' if last5[-1] == 'B' else 'Banker'}"
    if len(last5) >= 3:
        second_last = last5[-2]
        return f"OTB4L → Bet {'Player' if second_last == 'B' else 'Banker'}"
    
    return "Default: Bet Banker"

def main():
    st.title("Baccarat Predictor")
    
    # Initialize session state for history and prediction
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ""
    
    # Display current history
    st.markdown("**Current History:** " + ("".join(st.session_state.history) if st.session_state.history else "No results yet"))
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    # Banker button
    with col1:
        if st.button("Banker (B)"):
            st.session_state.history.append("B")
            st.session_state.prediction = get_prediction(st.session_state.history)
    
    # Player button
    with col2:
        if st.button("Player (P)"):
            st.session_state.history.append("P")
            st.session_state.prediction = get_prediction(st.session_state.history)
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.prediction = ""
    
    # Display prediction
    if st.session_state.prediction:
        st.markdown(f"**Prediction:** {st.session_state.prediction}")

if __name__ == "__main__":
    main()
