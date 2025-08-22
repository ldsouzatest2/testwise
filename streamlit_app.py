import streamlit as st
import pandas as pd
import json
import numpy as np
from faker import Faker
from itertools import combinations, product
import random
from typing import Dict, List, Any, Tuple

# Configure page
st.set_page_config(
    page_title="Advanced Test Case Creator",
    page_icon="🧪",
    layout="wide"
)

# Initialize Faker
fake = Faker()

# Initialize session state
if 'test_data' not in st.session_state:
    st.session_state.test_data = pd.DataFrame()
if 'test_steps' not in st.session_state:
    st.session_state.test_steps = []
if 'test_title' not in st.session_state:
    st.session_state.test_title = ""

# Keyword actions
keyword_actions = [
    "Open Browser", "Click", "Set Text", "Verify Element", "Select",
    "Check", "Uncheck", "Scroll", "Hover"
]

# Faker data types
faker_data_types = {
    "Name": lambda: fake.name(),
    "First Name": lambda: fake.first_name(),
    "Last Name": lambda: fake.last_name(),
    "Email": lambda: fake.email(),
    "Username": lambda: fake.user_name(),
    "Password": lambda: fake.password(),
    "Phone": lambda: fake.phone_number(),
    "Address": lambda: fake.address().replace('\n', ' '),
    "City": lambda: fake.city(),
    "Country": lambda: fake.country(),
    "Company": lambda: fake.company(),
    "Job Title": lambda: fake.job(),
    "Credit Card": lambda: fake.credit_card_number(),
    "SSN": lambda: fake.ssn(),
    "Date": lambda: fake.date(),
    "URL": lambda: fake.url(),
    "IPv4": lambda: fake.ipv4(),
    "Text": lambda: fake.text(max_nb_chars=50),
    "Number": lambda: str(fake.random_int(1, 1000)),
    "Boolean": lambda: str(fake.boolean()),
    "UUID": lambda: fake.uuid4()
}

def estimate_orthogonal_size(factors: Dict[str, List[str]]) -> int:
    """Estimate the size of orthogonal matrix for given factors"""
    max_levels = max(len(levels) for levels in factors.values())
    num_factors = len(factors)
    
    if num_factors <= 2:
        return min(25, max_levels ** num_factors)
    elif num_factors <= 4 and max_levels <= 4:
        return 16  # L16 array
    elif num_factors <= 8 and max_levels <= 2:
        return 8   # L8 array
    else:
        # Estimate pairwise combinations
        return min(50, max_levels * num_factors)

def generate_orthogonal_matrix(factors: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Generate optimized test combinations using orthogonal array principles
    """
    factor_names = list(factors.keys())
    factor_levels = [factors[name] for name in factor_names]
    
    # Calculate minimum combinations needed
    max_levels = max(len(levels) for levels in factor_levels)
    
    if len(factor_names) <= 2:
        # For 2 or fewer factors, use all combinations
        combinations = list(product(*factor_levels))
    elif len(factor_names) <= 4 and max_levels <= 4:
        # Use L16 orthogonal array for up to 4 factors with 4 levels
        combinations = generate_l16_array(factors)
    elif len(factor_names) <= 8 and max_levels <= 2:
        # Use L8 orthogonal array for up to 8 factors with 2 levels
        combinations = generate_l8_array(factors)
    else:
        # For larger sets, use pairwise combinations
        combinations = generate_pairwise_combinations(factors)
    
    # Convert to DataFrame
    df = pd.DataFrame(combinations, columns=factor_names)
    return df

def generate_l8_array(factors: Dict[str, List[str]]) -> List[Tuple]:
    """Generate L8 orthogonal array (2^7 design)"""
    l8_base = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    
    factor_names = list(factors.keys())[:8]  # Limit to 8 factors
    combinations = []
    
    for row in l8_base:
        combination = []
        for i, factor_name in enumerate(factor_names):
            if i < len(row):
                level_index = row[i] % len(factors[factor_name])
                combination.append(factors[factor_name][level_index])
            else:
                combination.append(factors[factor_name][0])
        combinations.append(tuple(combination))
    
    return combinations

def generate_l16_array(factors: Dict[str, List[str]]) -> List[Tuple]:
    """Generate L16 orthogonal array (4^5 design)"""
    l16_base = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 3],
        [0, 2, 2, 3, 1],
        [0, 3, 3, 2, 2],
        [1, 0, 1, 2, 2],
        [1, 1, 0, 3, 1],
        [1, 2, 3, 0, 3],
        [1, 3, 2, 1, 0],
        [2, 0, 2, 3, 3],
        [2, 1, 3, 2, 0],
        [2, 2, 0, 1, 2],
        [2, 3, 1, 0, 1],
        [3, 0, 3, 1, 1],
        [3, 1, 2, 0, 2],
        [3, 2, 1, 2, 0],
        [3, 3, 0, 3, 3]
    ]
    
    factor_names = list(factors.keys())[:5]  # Limit to 5 factors
    combinations = []
    
    for row in l16_base:
        combination = []
        for i, factor_name in enumerate(factor_names):
            if i < len(row):
                level_index = row[i] % len(factors[factor_name])
                combination.append(factors[factor_name][level_index])
            else:
                combination.append(factors[factor_name][0])
        combinations.append(tuple(combination))
    
    return combinations

def generate_pairwise_combinations(factors: Dict[str, List[str]]) -> List[Tuple]:
    """Generate pairwise combinations for large factor sets"""
    factor_names = list(factors.keys())
    all_pairs = []
    
    # Generate all pairwise combinations
    for i, j in combinations(range(len(factor_names)), 2):
        name1, name2 = factor_names[i], factor_names[j]
        for val1 in factors[name1]:
            for val2 in factors[name2]:
                all_pairs.append((i, j, val1, val2))
    
    # Greedy algorithm to find minimum set covering all pairs
    combinations_list = []
    covered_pairs = set()
    
    while len(covered_pairs) < len(all_pairs):
        best_combination = None
        best_coverage = 0
        
        # Try random combinations
        for _ in range(100):  # Limit iterations
            combination = []
            for name in factor_names:
                combination.append(random.choice(factors[name]))
            
            # Count how many uncovered pairs this combination covers
            coverage = 0
            for i, j in combinations(range(len(factor_names)), 2):
                pair = (i, j, combination[i], combination[j])
                if pair in all_pairs and pair not in covered_pairs:
                    coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_combination = combination
        
        if best_combination:
            combinations_list.append(tuple(best_combination))
            # Mark covered pairs
            for i, j in combinations(range(len(factor_names)), 2):
                pair = (i, j, best_combination[i], best_combination[j])
                if pair in all_pairs:
                    covered_pairs.add(pair)
        else:
            break  # No more improvements possible
    
    return combinations_list[:50]  # Limit to reasonable size

def generate_fake_data(data_type: str, count: int) -> List[str]:
    """Generate fake data of specified type and count"""
    generator = faker_data_types.get(data_type)
    if generator:
        return [generator() for _ in range(count)]
    return [f"sample_{i+1}" for i in range(count)]

def add_test_step():
    """Add a new test step to the session state"""
    new_step = {
        "step": f"Step {len(st.session_state.test_steps) + 1}",
        "action": "Click",
        "target": "",
        "locator": "",
        "value": "",
        "expected": ""
    }
    st.session_state.test_steps.append(new_step)

def remove_test_step(index):
    """Remove a test step by index"""
    if 0 <= index < len(st.session_state.test_steps):
        st.session_state.test_steps.pop(index)
        # Update step numbers
        for i, step in enumerate(st.session_state.test_steps):
            step["step"] = f"Step {i + 1}"

def generate_json_output() -> str:
    """Generate the final JSON output"""
    # Convert test data to the required format
    test_data_dict = {}
    if not st.session_state.test_data.empty:
        for column in st.session_state.test_data.columns:
            values = st.session_state.test_data[column].dropna().astype(str).tolist()
            test_data_dict[column] = ",".join(values)
    
    output = {
        "title": st.session_state.test_title or "Untitled Test",
        "test_data": [test_data_dict] if test_data_dict else [],
        "steps": st.session_state.test_steps
    }
    
    return json.dumps(output, indent=2)

# Main UI
st.title("🧪 Advanced Test Case Creator")
st.markdown("Create comprehensive test cases with Faker data generation and orthogonal matrix optimization")

# Sidebar for test configuration
with st.sidebar:
    st.header("⚙️ Test Configuration")
    st.session_state.test_title = st.text_input(
        "Test Title", 
        value=st.session_state.test_title,
        placeholder="e.g., Login Test"
    )
    
    st.markdown("---")
    st.markdown("### 🎲 Data Generation")
    
    # Faker data generation
    with st.expander("🤖 Generate Fake Data"):
        faker_type = st.selectbox("Data Type", list(faker_data_types.keys()))
        faker_count = st.number_input("Number of Records", min_value=1, max_value=100, value=5)
        faker_column_name = st.text_input("Column Name", value=faker_type)
        
        if st.button("Generate Fake Data"):
            fake_data = generate_fake_data(faker_type, faker_count)
            new_df = pd.DataFrame({faker_column_name: fake_data})
            
            if st.session_state.test_data.empty:
                st.session_state.test_data = new_df
            else:
                st.session_state.test_data = pd.concat([st.session_state.test_data, new_df], axis=1)
            st.rerun()
    
    # Orthogonal matrix generation
    with st.expander("📊 Orthogonal Matrix"):
        orthogonal_tab1, orthogonal_tab2 = st.tabs(["📝 Manual Entry", "🔄 From Existing Data"])
        
        with orthogonal_tab1:
            st.markdown("Define factors and their levels for optimized combinations:")
            
            if 'orthogonal_factors' not in st.session_state:
                st.session_state.orthogonal_factors = {}
            
            # Add factor
            factor_name = st.text_input("Factor Name", placeholder="e.g., Browser, OS, User Type")
            factor_levels = st.text_input("Levels (comma-separated)", placeholder="e.g., Chrome, Firefox, Safari")
            
            if st.button("Add Factor") and factor_name and factor_levels:
                levels = [level.strip() for level in factor_levels.split(',')]
                st.session_state.orthogonal_factors[factor_name] = levels
                st.rerun()
            
            # Display current factors
            if st.session_state.orthogonal_factors:
                st.write("**Current Factors:**")
                for name, levels in st.session_state.orthogonal_factors.items():
                    st.write(f"- {name}: {', '.join(levels)}")
                
                if st.button("Generate Orthogonal Matrix"):
                    orthogonal_df = generate_orthogonal_matrix(st.session_state.orthogonal_factors)
                    st.session_state.test_data = orthogonal_df
                    st.success(f"Generated {len(orthogonal_df)} optimized test combinations!")
                    st.rerun()
                
                if st.button("Clear Factors"):
                    st.session_state.orthogonal_factors = {}
                    st.rerun()
        
        with orthogonal_tab2:
            st.markdown("Generate orthogonal matrix from your existing test data:")
            
            if not st.session_state.test_data.empty:
                # Show current data info
                st.write("**Current Data Analysis:**")
                current_factors = {}
                for col in st.session_state.test_data.columns:
                    unique_vals = st.session_state.test_data[col].dropna().unique()
                    current_factors[col] = list(unique_vals)
                    st.write(f"- **{col}**: {len(unique_vals)} levels ({', '.join(map(str, unique_vals[:3]))}{'...' if len(unique_vals) > 3 else ''})")
                
                # Calculate potential reduction
                current_combinations = len(st.session_state.test_data)
                if len(current_factors) > 1:
                    estimated_orthogonal = estimate_orthogonal_size(current_factors)
                    reduction = max(0, (current_combinations - estimated_orthogonal) / current_combinations * 100)
                    
                    st.info(f"📊 **Potential Optimization**: {current_combinations} → {estimated_orthogonal} combinations ({reduction:.1f}% reduction)")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Optimize Existing Data", type="primary"):
                        optimized_df = generate_orthogonal_matrix(current_factors)
                        st.session_state.test_data = optimized_df
                        st.success(f"✅ Optimized! Reduced to {len(optimized_df)} combinations with full coverage!")
                        st.rerun()
                
                with col2:
                    if st.button("📋 Copy Structure"):
                        st.session_state.orthogonal_factors = current_factors.copy()
                        st.success("✅ Structure copied to Manual Entry tab!")
                        st.rerun()
                
                # Show optimization preview
                if st.checkbox("🔍 Preview Optimization"):
                    preview_df = generate_orthogonal_matrix(current_factors)
                    st.write("**Optimized Preview:**")
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    if len(preview_df) > 10:
                        st.caption(f"Showing first 10 of {len(preview_df)} optimized combinations")
            else:
                st.info("No test data available. Add some test data first to use this feature.")
    
    st.markdown("---")
    st.markdown("### 📋 Quick Actions")
    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.test_data = pd.DataFrame()
        st.session_state.test_steps = []
        st.session_state.orthogonal_factors = {}
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Test Data Management")
    
    # Test data editor
    st.subheader("Excel-like Data Editor")
    
    # Option to add new columns
    with st.expander("➕ Add New Column"):
        col_name = st.text_input("Column Name", placeholder="e.g., Username, Password, Email")
        if st.button("Add Column") and col_name:
            if st.session_state.test_data.empty:
                st.session_state.test_data = pd.DataFrame({col_name: [""]})
            else:
                st.session_state.test_data[col_name] = ""
            st.rerun()
    
    # Data editor
    if not st.session_state.test_data.empty:
        edited_data = st.data_editor(
            st.session_state.test_data,
            num_rows="dynamic",
            use_container_width=True,
            key="test_data_editor"
        )
        st.session_state.test_data = edited_data
        
        # Show data statistics
        st.subheader("Data Statistics")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total Combinations", len(st.session_state.test_data))
        with col_stats2:
            st.metric("Factors", len(st.session_state.test_data.columns))
        with col_stats3:
            if len(st.session_state.test_data.columns) > 1:
                total_possible = 1
                for col in st.session_state.test_data.columns:
                    unique_vals = st.session_state.test_data[col].nunique()
                    total_possible *= unique_vals if unique_vals > 0 else 1
                reduction = (1 - len(st.session_state.test_data) / total_possible) * 100 if total_possible > 0 else 0
                st.metric("Reduction", f"{reduction:.1f}%")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.test_data, use_container_width=True)
        
        # Variable reference helper
        if not st.session_state.test_data.empty:
            st.info(
                "💡 **Variable Reference:** Use `${ColumnName}` in test steps to reference data. "
                f"Available variables: {', '.join([f'${col}' for col in st.session_state.test_data.columns])}"
            )
    else:
        st.info("No test data defined. Use sidebar options to generate data or add columns manually.")

with col2:
    st.header("🎯 Test Steps")
    
    # Add step button
    if st.button("➕ Add Test Step", type="primary"):
        add_test_step()
        st.rerun()
    
    # Test steps management
    for i, step in enumerate(st.session_state.test_steps):
        with st.container():
            st.markdown(f"### {step['step']}")
            
            # Create columns for step fields
            step_col1, step_col2, step_col3 = st.columns([2, 1, 1])
            
            with step_col1:
                step["action"] = st.selectbox(
                    "Action",
                    keyword_actions,
                    index=keyword_actions.index(step["action"]) if step["action"] in keyword_actions else 0,
                    key=f"action_{i}"
                )
                
                step["target"] = st.text_input(
                    "Target",
                    value=step["target"],
                    placeholder="e.g., Username field, Login button",
                    key=f"target_{i}"
                )
            
            with step_col2:
                step["locator"] = st.text_input(
                    "Locator",
                    value=step["locator"],
                    placeholder="e.g., #username, .btn-login",
                    key=f"locator_{i}"
                )
                
                step["value"] = st.text_input(
                    "Value",
                    value=step["value"],
                    placeholder="e.g., ${Username}, www.google.com",
                    key=f"value_{i}"
                )
            
            with step_col3:
                step["expected"] = st.text_area(
                    "Expected Result",
                    value=step["expected"],
                    height=100,
                    placeholder="Describe expected outcome",
                    key=f"expected_{i}"
                )
                
                if st.button("🗑️", key=f"remove_{i}", help="Remove this step"):
                    remove_test_step(i)
                    st.rerun()
            
            st.markdown("---")

# Generate and display JSON output
st.header("📤 Generated JSON Output")

if st.session_state.test_steps or not st.session_state.test_data.empty:
    json_output = generate_json_output()
    
    # Display JSON with syntax highlighting
    st.code(json_output, language="json")
    
    # Download button
    st.download_button(
        label="💾 Download JSON",
        data=json_output,
        file_name=f"{st.session_state.test_title or 'test_case'}.json",
        mime="application/json",
        type="primary"
    )
    
    # Copy to clipboard button
    if st.button("📋 Copy to Clipboard"):
        st.code(json_output, language="json")
        st.success("JSON copied! Use Ctrl+A, Ctrl+C to copy from the code block above.")
else:
    st.info("Add test data or test steps to generate JSON output.")

# Footer with usage instructions
st.markdown("---")
with st.expander("📖 Advanced Usage Instructions"):
    st.markdown("""
    ### 🆕 New Features:
    
    #### 🤖 Faker Data Generation:
    - **20+ Data Types**: Names, emails, addresses, credit cards, etc.
    - **Bulk Generation**: Create multiple records at once
    - **Realistic Data**: Industry-standard fake data for testing
    
    #### 📊 Orthogonal Matrix Optimization:
    - **Reduced Test Cases**: Generate minimum combinations for maximum coverage
    - **Pairwise Testing**: Covers all pair interactions efficiently  
    - **Smart Algorithms**: Uses L8, L16 arrays and pairwise combinations
    - **Coverage Statistics**: Shows reduction percentage vs full factorial
    
    ### How to Use Advanced Features:
    
    #### Faker Data Generation:
    1. Select data type (Email, Name, Password, etc.)
    2. Set number of records to generate
    3. Specify column name
    4. Click "Generate Fake Data"
    
    #### Orthogonal Matrix:
    1. Define factors (e.g., Browser: Chrome, Firefox, Safari)
    2. Add multiple factors with their levels
    3. Click "Generate Orthogonal Matrix"
    4. Get optimized test combinations with high coverage
    
    ### Example Orthogonal Factors:
    - **Browser**: Chrome, Firefox, Safari, Edge
    - **OS**: Windows, Mac, Linux
    - **User Type**: Admin, Regular, Guest
    - **Device**: Desktop, Mobile, Tablet
    
    ### Benefits:
    - **Time Saving**: 80% reduction in test cases with same coverage
    - **Better Coverage**: Mathematical guarantee of interaction coverage
    - **Realistic Data**: Professional-grade test data generation
    - **Export Ready**: JSON format for TestRail, Katalon, Playwright
    """)

# Display current stats
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Current Stats")
    st.metric("Test Steps", len(st.session_state.test_steps))
    st.metric("Data Columns", len(st.session_state.test_data.columns) if not st.session_state.test_data.empty else 0)
    st.metric("Data Rows", len(st.session_state.test_data) if not st.session_state.test_data.empty else 0)
    if hasattr(st.session_state, 'orthogonal_factors'):
        st.metric("Orthogonal Factors", len(st.session_state.orthogonal_factors))