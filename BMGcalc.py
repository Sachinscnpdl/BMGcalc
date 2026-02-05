import os
# Set this before any other imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
import base64
import periodictable

# Set page configuration
st.set_page_config(
    page_title="BMGcalc - Metallic Glass Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4b6cb7;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        color: #5e6e82;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .periodic-table {
        display: grid;
        grid-template-columns: repeat(18, 60px);
        gap: 5px;
        margin: 20px 0;
        justify-content: center;
    }
    .element {
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid #ccc;
        border-radius: 8px;
        background-color: #f0f0f0;
        cursor: pointer;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .element:hover {
        background-color: #ddd;
    }
    .element.selected {
        background-color: #4b6cb7;
        color: white;
        border-color: #4b6cb7;
    }
    .warning {
        color: red;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    .prediction-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: left;
        color: #333;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .prediction-box h3 {
        font-size: 24px;
        font-weight: bold;
        color: #4b6cb7;
        margin-bottom: 15px;
    }
    .prediction-box .property {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .prediction-box .property strong {
        min-width: 300px;
        font-weight: bold;
        color: #333;
    }
    .prediction-box .value-box {
        background-color: #e0f7fa;
        padding: 8px 12px;
        border-radius: 5px;
        border: 1px solid #4b6cb7;
        font-weight: bold;
        color: #00796b;
        min-width: 100px;
        text-align: center;
    }
    .fraction-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: left;
        color: #333;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .fraction-box h3 {
        font-size: 24px;
        font-weight: bold;
        color: #FFA500;
        margin-bottom: 15px;
    }
    .fraction-box .fraction-input {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .fraction-box .fraction-input strong {
        min-width: 150px;
        font-weight: bold;
        color: #333;
    }
    .fraction-box .fraction-input input {
        padding: 8px 12px;
        border-radius: 5px;
        border: 1px solid #4b6cb7;
        font-size: 16px;
        font-weight: bold;
        color: #00796b;
        width: 100px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4b6cb7;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a5a8c;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .alert-info {
        background-color: #e7f3ff;
        border-color: #b3d9ff;
        color: #0066cc;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Function to generate the periodic table using HTML and CSS
def generate_periodic_table():
    layout = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        ["Li", "Be", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
        ["Na", "Mg", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
        ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
        ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["", "", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "", ""],
        ["", "", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "", ""]
    ]

    st.markdown('<div class="periodic-table">', unsafe_allow_html=True)
    for row_idx, row in enumerate(layout):
        cols = st.columns(len(row))
        for col_idx, (col, element) in enumerate(zip(cols, row)):
            if element:
                is_selected = element in st.session_state.selected_elements
                if col.button(
                    element,
                    key=f"btn_{element}_{row_idx}_{col_idx}",
                    help=f"Select {element}",
                    type="primary" if is_selected else "secondary"
                ):
                    if element in st.session_state.selected_elements:
                        st.session_state.selected_elements.remove(element)
                    else:
                        st.session_state.selected_elements.append(element)
            else:
                col.write("")
    st.markdown('</div>', unsafe_allow_html=True)

# Function to process alloys and generate predictions
def process_alloys_demo(composition_string):
    """Demo version of process_alloys for testing without the actual model"""
    # Create a mock results DataFrame
    results = pd.DataFrame({
        'Alloys': [composition_string],
        'Predicted_Phase': ['Metalic_Glass'],
        'Phase_Confidence': [0.85],
        'Predicted_Tg': [np.random.normal(600, 100)],
        'Predicted_Tx': [np.random.normal(650, 100)],
        'Predicted_Tl': [np.random.normal(750, 100)],
        'Predicted_Dmax': [np.random.exponential(2.0)],
        'Predicted_Rc': [np.random.exponential(5.0)]
    })
    
    return results.iloc[0]

# Function to create download link for DataFrame
def get_download_link(df, filename="predictions.csv", text="Download Results"):
    """Generate a link to download the DataFrame as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Streamlit UI
st.markdown('<div class="title">BMGcalc - Bulk Metallic Glass Design Calculator</div>', unsafe_allow_html=True)

# Info alert about demo mode
st.markdown("""
<div class="alert-info">
<strong>🔬 Demo Mode:</strong> This is a demonstration version of BMGcalc with simulated predictions. 
The actual model integration is available in the full version.
</div>
""", unsafe_allow_html=True)

st.markdown("""
BMGcalc uses advanced machine learning models to predict key properties of metallic glasses, including:
- Phase classification (Metallic Glass vs. Crystalline)
- Thermal properties (Tg, Tx, Tl)
- Critical diameter (Dmax)
- Critical cooling rate (Rc)
""")

# Step 1: Number of Elements
num_elements = st.number_input("Number of elements", min_value=1, max_value=10, step=1, value=2)

# Step 2: Element Selection Method
selection_method = st.radio("Choose element selection method:", ("Periodic Table", "Dropdown"))

# Step 3: Periodic Table or Dropdown Selection
st.markdown('<div class="subtitle">Select Elements</div>', unsafe_allow_html=True)
if selection_method == "Periodic Table":
    generate_periodic_table()
else:
    all_elements = [el.symbol for el in periodictable.elements if el.symbol]
    selected_elements = st.multiselect(
        "Choose elements:", 
        all_elements, 
        default=all_elements[:num_elements],
        key="dropdown_elements"
    )
    st.session_state.selected_elements = selected_elements

# Step 4: Element Fraction Input
st.markdown(
    """
    <div class="fraction-box">
        <h3>Enter Element Fraction (%)</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Dynamically generate fraction inputs for selected elements
for elem in st.session_state.selected_elements:
    fraction = st.number_input(
        f"{elem} fraction (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=100.0 / len(st.session_state.selected_elements),
        key=f"fraction_{elem}"
    )

# Ensure the total fraction is 100%
total_fraction = sum(st.session_state[f"fraction_{elem}"] for elem in st.session_state.selected_elements)
if total_fraction != 100:
    st.markdown('<div class="warning">Total fraction must be 100%</div>', unsafe_allow_html=True)

# Predict button
if st.button("Predict Properties", key="predict"):
    if total_fraction == 100 and st.session_state.selected_elements:
        # Create composition string
        composition = ""
        for elem in st.session_state.selected_elements:
            fraction = st.session_state[f"fraction_{elem}"]
            composition += elem + str(int(fraction))
        
        # Process the alloy
        with st.spinner("Processing alloy and predicting properties..."):
            st.session_state.predictions = process_alloys_demo(composition)

# Output Section: Predictions
if st.session_state.predictions is not None:
    st.markdown('<div class="subtitle">Prediction Panel</div>', unsafe_allow_html=True)
    
    # Create a DataFrame with the predictions
    pred = st.session_state.predictions
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Predicted Phase", pred['Predicted_Phase'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Tg (K)", f"{pred['Predicted_Tg']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Dmax (mm)", f"{pred['Predicted_Dmax']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display detailed predictions
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>Predicted Properties for {composition}</h3>
            <div class="property">
                <strong>Predicted Phase:</strong>
                <div class="value-box">{pred['Predicted_Phase']}</div>
            </div>
            <div class="property">
                <strong>Phase Confidence:</strong>
                <div class="value-box">{pred['Phase_Confidence']:.2f}</div>
            </div>
            <div class="property">
                <strong>Glass Transition Temperature (T<sub>g</sub>) [K]:</strong>
                <div class="value-box">{pred['Predicted_Tg']:.1f}</div>
            </div>
            <div class="property">
                <strong>Crystallization Temperature (T<sub>x</sub>) [K]:</strong>
                <div class="value-box">{pred['Predicted_Tx']:.1f}</div>
            </div>
            <div class="property">
                <strong>Liquidus Temperature (T<sub>l</sub>) [K]:</strong>
                <div class="value-box">{pred['Predicted_Tl']:.1f}</div>
            </div>
            <div class="property">
                <strong>Supercooled Liquid Region (T<sub>x</sub>-T<sub>g</sub>) [K]:</strong>
                <div class="value-box">{pred['Predicted_Tx'] - pred['Predicted_Tg']:.1f}</div>
            </div>
            <div class="property">
                <strong>Critical Diameter (d<sub>c</sub>) [mm]:</strong>
                <div class="value-box">{pred['Predicted_Dmax']:.3f}</div>
            </div>
            <div class="property">
                <strong>Critical Cooling Rate (R<sub>c</sub>) [K/s]:</strong>
                <div class="value-box">{pred['Predicted_Rc']:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create visualizations
    st.markdown("### Property Visualizations")
    
    # Create a gauge chart for glass forming ability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = pred['Predicted_Dmax'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Glass Forming Ability"},
        delta = {'reference': 1},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#4b6cb7"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 5], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5}}))
    
    fig_gauge.update_layout(height=300)
    
    # Create a bar chart for thermal properties
    fig_thermal = go.Figure()
    fig_thermal.add_trace(go.Bar(
        x=['Tg', 'Tx', 'Tl'],
        y=[pred['Predicted_Tg'], pred['Predicted_Tx'], pred['Predicted_Tl']],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        text=[f"{pred['Predicted_Tg']:.1f} K", f"{pred['Predicted_Tx']:.1f} K", f"{pred['Predicted_Tl']:.1f} K"],
        textposition='auto'
    ))
    fig_thermal.update_layout(
        title="Thermal Properties",
        xaxis_title="Property",
        yaxis_title="Temperature (K)",
        height=400
    )
    
    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col2:
        st.plotly_chart(fig_thermal, use_container_width=True)
    
    # Create a DataFrame for download
    results_df = pd.DataFrame({
        'Alloy': [composition],
        'Phase': [pred['Predicted_Phase']],
        'Tg (K)': [pred['Predicted_Tg']],
        'Tx (K)': [pred['Predicted_Tx']],
        'Tl (K)': [pred['Predicted_Tl']],
        'Dmax (mm)': [pred['Predicted_Dmax']],
        'Rc (K/s)': [pred['Predicted_Rc']]
    })
    
    # Download button
    st.markdown(get_download_link(results_df, "bmg_predictions.csv", "📥 Download Results as CSV"), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**About BMGcalc:** BMGcalc is a machine learning-powered tool for predicting properties of bulk metallic glasses. 
It uses state-of-the-art models trained on extensive experimental data to provide accurate predictions for various alloy systems.
""")
