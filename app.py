import streamlit as st

class App:
    def __init__(self):
        # Initialize session state variables
        if 'profit_lock' not in st.session_state:
            st.session_state.profit_lock = 0.0  # Numeric value
        if 'profit_lock_display' not in st.session_state:
            st.session_state.profit_lock_display = "Profit Lock: $0.00"  # Display string
        if 'results' not in st.session_state:
            st.session_state.results = []  # Store results for demo purposes

    def record_result(self, result):
        """
        Update profit_lock based on the result.
        For demo: 'P' adds $10.0, anything else subtracts $5.0.
        """
        if result == 'P':
            st.session_state.profit_lock += 10.0
        else:
            st.session_state.profit_lock -= 5.0
        # Ensure profit_lock doesn't go negative (optional, based on your logic)
        st.session_state.profit_lock = max(0.0, st.session_state.profit_lock)
        # Store result for display (optional)
        st.session_state.results.append(result)
        # Update the display
        self.update_display()

    def update_display(self):
        """Update the UI with the formatted profit_lock value."""
        # Format the display string without overwriting the numeric profit_lock
        st.session_state.profit_lock_display = f"Profit Lock: ${st.session_state.profit_lock:.2f}"
        # Update Streamlit UI
        st.markdown(f"**{st.session_state.profit_lock_display}**")
        # Optionally display results history
        if st.session_state.results:
            st.write("Results History:", ", ".join(st.session_state.results))

def main():
    """Main function to run the Streamlit app."""
    st.title("Profit Lock Tracker")
    
    # Create an instance of the App
    app = App()
    
    # Add a button to simulate recording a result
    if st.button("Record Positive Result (P)"):
        app.record_result('P')
    
    # Add another button for a negative result (for demo)
    if st.button("Record Negative Result (N)"):
        app.record_result('N')
    
    # Display current state
    app.update_display()

if __name__ == "__main__":
    main()
