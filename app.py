import streamlit as st

def get_prediction(history): if len(history) < 5: return "Need at least 5 rounds of history."

last_5 = history[-5:]

# Rule 4: Dragon Slayer
if len(set(last_5[-3:])) == 1:
    return "Player" if last_5[-1] == "B" else "Banker"

# Rule 3: Zigzag Breaker
if last_5[-4:] in [["B", "P", "B", "P"], ["P", "B", "P", "B"]]:
    return last_5[-1]

# Rule 2: OTB4L
if last_5[-1] != last_5[-3]:
    return "Player" if last_5[-3] == "B" else "Banker"

# Rule 5: 2-in-a-row Catcher
if last_5[-2] == last_5[-1] and last_5[-3] != last_5[-2]:
    return last_5[-1]

# Rule 1: Default Banker
return "Banker"

st.title("Smart Fusion 5 - Baccarat Predictor")

st.markdown(""" Enter the last 5+ Baccarat outcomes using P for Player and B for Banker. Ties should be ignored. Examples:

B P B B P

P P B B B """)


input_string = st.text_input("Enter last outcomes (space-separated):", "")

if input_string: entries = input_string.upper().split() valid_entries = [e for e in entries if e in ["B", "P"]]

if len(valid_entries) < 5:
    st.warning("Please enter at least 5 outcomes (B or P only). Ties should be excluded.")
else:
    prediction = get_prediction(valid_entries)
    st.success(f"Next recommended bet: **{prediction}**")

