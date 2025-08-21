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

if "test_data" not in st.session_state:
    st.session_state.test_data = []

if "test_data_labels" not in st.session_state:
    st.session_state.test_data_labels = {"key_label": "Username", "value_label": "Password"}

# Input for test case title
st.session_state.title = st.text_input("Test Case Title", st.session_state.title)

# Configurable test data labels
st.subheader("Configure Test Data Field Names")
key_label = st.text_input("Test Data Key Field Label", st.session_state.test_data_labels["key_label"])
value_label = st.text_input("Test Data Value Field Label", st.session_state.test_data_labels["value_label"])
st.session_state.test_data_labels["key_label"] = key_label
st.session_state.test_data_labels["value_label"] = value_label

# --- Test Data form ---
with st.form("test_data_form", clear_on_submit=True):
    key_input = st.text_input(f"{key_label} (comma-separated for multiple values, e.g., a1,a2,a3)")
    value_input = st.text_input(f"{value_label} (comma-separated for multiple values, e.g., b1,b2,b3)")
    submitted_data = st.form_submit_button("Add Test Data")
    if submitted_data and key_input:
        key_list = [key_label]  # Use the label as the key
        value_list = [v.strip() for v in value_input.split(",")]
        key_values_dict = {key_label: ",".join([v for v in key_input.split(",")]), 
                           value_label: ",".join(value_list)}
        st.session_state.test_data.append(key_values_dict)

# --- Show test data with delete buttons ---
st.subheader("Test Data")
for idx, data_set in enumerate(st.session_state.test_data):
    cols = st.columns([4, 1])
    cols[0].json(data_set)
    if cols[1].button("Delete", key=f"delete_data_{idx}"):
        st.session_state.test_data.pop(idx)
        st.experimental_rerun()

# --- Keyword actions ---
keyword_actions = [
    "Open Browser",
    "Click",
    "Set Text",
    "Verify Element",
    "Select",
    "Check",
    "Uncheck",
    "Scroll",
    "Hover"
]

# --- Step form ---
with st.form("step_form", clear_on_submit=True):
    step_number = len(st.session_state.steps) + 1
    action = st.selectbox("Action", keyword_actions)
    target = st.text_input("Target (field name)")
    locator = st.text_input("Locator (optional)")
    value = st.text_input("Value")
    expected = st.text_input("Expected Result")
    submitted = st.form_submit_button("Add Step")
    if submitted:
        st.session_state.steps.append({
            "step": f"Step {step_number}",
            "action": action,
            "target": target,
            "locator": locator,
            "value": value,
            "expected": expected
        })

# --- Show steps with delete buttons ---
st.subheader("Steps Added")
for idx, step in enumerate(st.session_state.steps):
    cols = st.columns([4, 1])
    cols[0].json(step)
    if cols[1].button("Delete", key=f"delete_step_{idx}"):
        st.session_state.steps.pop(idx)
        st.experimental_rerun()

# --- Export JSON ---
if st.button("Download JSON"):
    test_case = {
        "title": st.session_state.title,
        "test_data": st.session_state.test_data,
        "steps": st.session_state.steps
    }
    st.download_button(
        label="Download Test Case JSON",
        data=json.dumps(test_case, indent=2),
        file_name="test_case.json",
        mime="application/json"
    )
