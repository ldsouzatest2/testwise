import streamlit as st
import json

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>TestWise</h1>", unsafe_allow_html=True)
st.markdown("---")  # horizontal line

# Initialize session state
if "steps" not in st.session_state:
    st.session_state.steps = []

if "title" not in st.session_state:
    st.session_state.title = ""

# Input for test case title
st.session_state.title = st.text_input("Test Case Title", st.session_state.title)

# Step form
with st.form("step_form"):
    step_number = len(st.session_state.steps) + 1
    action = st.text_input("Action")
    target = st.text_input("Target")
    locator = st.text_input("Locator (optional)")
    expected = st.text_input("Expected Result")
    submitted = st.form_submit_button("Add Step")
    if submitted:
        st.session_state.steps.append({
            "step": f"Step {step_number}",
            "action": action,
            "target": target,
            "locator": locator,
            "expected": expected
        })

# Show current steps
st.subheader("Steps Added")
st.json(st.session_state.steps)

# Export JSON
if st.button("Download JSON"):
    test_case = {
        "title": st.session_state.title,
        "steps": st.session_state.steps
    }
    st.download_button(
        label="Download Test Case JSON",
        data=json.dumps(test_case, indent=2),
        file_name="test_case.json",
        mime="application/json"
    )
