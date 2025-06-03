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
    
    # Initialize session state for history if not exists
    if 'history' not in st.session_state:
        st.session_state.history = ""
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ""
    
    # Input field for history
    history_input = st.text_input("Enter last results (e.g., BPPBB)", 
                                 value=st.session_state.history,
                                 key="history_input")
    
    # Update history in session state
    st.session_state.history = history_input
    
    # Predict button
    if st.button("Get Prediction"):
        clean_history = history_input.upper().replace("[^BP]", "").split()
        clean_history = list(''.join(clean_history))
        if clean_history:
            st.session_state.prediction = get_prediction(clean_history)
        else:
            st.session_state.prediction = "Please enter valid history (B or P only)"
    
    # Display prediction
    if st.session_state.prediction:
        st.markdown(f"**Prediction:** {st.session_state.prediction}")

if __name__ == "__main__":
    main()
