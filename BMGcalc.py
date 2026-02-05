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
    page_title="BMGcalc",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with gradient background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        padding: 20px;
    }
    
    .main-container {
        background: white;
        border-radius: 24px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1400px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        font-size: 24px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .element-grid {
        display: grid;
        grid-template-columns: repeat(18, 45px);
        gap: 4px;
        margin: 20px 0;
        justify-content: center;
    }
    
    .element-cell {
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid #e6e8f0;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a;
        transition: all 0.2s ease;
    }
    
    .element-cell:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .element-cell.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid #e6e8f0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
    }
    
    .property-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #f0f2f6;
    }
    
    .property-row:last-child {
        border-bottom: none;
    }
    
    .property-name {
        font-size: 16px;
        font-weight: 500;
        color: #4a5568;
    }
    
    .property-value {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        background: #f8f9ff;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        min-width: 120px;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e6e8f0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #667eea;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 14px 32px;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-size: 16px;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .fraction-input {
        background: #f8f9ff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 16px;
        font-weight: 500;
        color: #1a1a1a;
        width: 100%;
        text-align: center;
    }
    
    .tab-container {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #e6e8f0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Warning message */
    .warning-message {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        color: #c53030;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'element_fractions' not in st.session_state:
    st.session_state.element_fractions = {}

# Modern periodic table generator
def generate_modern_periodic_table():
    layout = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
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

    st.markdown('<div class="element-grid">', unsafe_allow_html=True)
    for row_idx, row in enumerate(layout):
        for col_idx, element in enumerate(row):
            if element:
                is_selected = element in st.session_state.selected_elements
                button_style = "selected" if is_selected else ""
                
                # Create a button using st.button with custom styling
                if st.button(
                    element,
                    key=f"btn_{element}_{row_idx}_{col_idx}",
                    help=f"Click to select {element}",
                ):
                    if element in st.session_state.selected_elements:
                        st.session_state.selected_elements.remove(element)
                    else:
                        st.session_state.selected_elements.append(element)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Mock prediction function
def process_alloys_demo(composition_string):
    results = pd.DataFrame({
        'Alloys': [composition_string],
        'Predicted_Phase': ['Metallic Glass'],
        'Phase_Confidence': [np.random.uniform(0.7, 0.95)],
        'Predicted_Tg': [np.random.normal(600, 50)],
        'Predicted_Tx': [np.random.normal(650, 50)],
        'Predicted_Tl': [np.random.normal(750, 50)],
        'Predicted_Dmax': [np.random.exponential(1.5) + 1],
        'Predicted_Rc': [np.random.exponential(3) + 1]
    })
    return results.iloc[0]

# Download link generator
def get_download_link(df, filename="bmg_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block;">📥 Download Results</a>'
    return href

# Main App UI
st.markdown('<h1 class="main-header">BMGcalc</h1>', unsafe_allow_html=True)

# Main container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Input Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-title">Alloy Composition</div>', unsafe_allow_html=True)
        
        # Number of elements
        num_elements = st.selectbox(
            "Number of elements",
            options=[2, 3, 4, 5, 6],
            index=0,
            help="Select number of elements in your alloy"
        )
        
        # Element selection method
        method = st.radio(
            "Selection method",
            ["Periodic Table", "Dropdown List"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Element selection
        if method == "Periodic Table":
            generate_modern_periodic_table()
        else:
            all_elements = [el.symbol for el in periodictable.elements if el.symbol]
            selected = st.multiselect(
                "Select elements",
                all_elements,
                default=["Cu", "Zr"],
                max_selections=num_elements,
                key="dropdown_elements"
            )
            st.session_state.selected_elements = selected[:num_elements]
        
        # Update selected elements to match number limit
        if len(st.session_state.selected_elements) > num_elements:
            st.session_state.selected_elements = st.session_state.selected_elements[:num_elements]
    
    with col2:
        st.markdown('<div class="section-title">Element Fractions</div>', unsafe_allow_html=True)
        
        if st.session_state.selected_elements:
            cols = st.columns(len(st.session_state.selected_elements))
            total = 0
            
            for idx, elem in enumerate(st.session_state.selected_elements):
                with cols[idx]:
                    default_value = 100 / len(st.session_state.selected_elements)
                    fraction = st.number_input(
                        f"{elem}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_value),
                        step=0.1,
                        format="%.1f",
                        key=f"frac_{elem}"
                    )
                    st.session_state.element_fractions[elem] = fraction
                    total += fraction
            
            st.progress(total/100, text=f"Total: {total:.1f}%")
            
            if abs(total - 100) > 0.1:
                st.markdown('<div class="warning-message">Total must be 100%</div>', unsafe_allow_html=True)
        else:
            st.info("Select elements first")
    
    # Prediction button
    if st.button("🔬 Predict Properties", use_container_width=True):
        if st.session_state.selected_elements and abs(total - 100) < 0.1:
            composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                 for elem in st.session_state.selected_elements])
            with st.spinner("Analyzing alloy composition..."):
                st.session_state.predictions = process_alloys_demo(composition)
    
    # Results Section
    if st.session_state.predictions is not None:
        st.markdown("---")
        pred = st.session_state.predictions
        
        # Key Metrics
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Phase</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred["Predicted_Phase"]}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{pred['Phase_Confidence']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Glass Forming</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred["Predicted_Dmax"]:.2f} mm</div>', unsafe_allow_html=True)
            st.metric("Critical Cooling", f"{pred['Predicted_Rc']:.1f} K/s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Tg</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred["Predicted_Tg"]:.0f} K</div>', unsafe_allow_html=True)
            st.metric("ΔT", f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.0f} K")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Tx</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred["Predicted_Tx"]:.0f} K</div>', unsafe_allow_html=True)
            st.metric("Tl", f"{pred['Predicted_Tl']:.0f} K")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Properties
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 18px; font-weight: 600; color: #1a1a1a; margin-bottom: 20px;">Property Details</div>', unsafe_allow_html=True)
        
        properties = [
            ("Phase Classification", f"{pred['Predicted_Phase']} ({pred['Phase_Confidence']:.1%})"),
            ("Glass Transition (Tg)", f"{pred['Predicted_Tg']:.1f} K"),
            ("Crystallization (Tx)", f"{pred['Predicted_Tx']:.1f} K"),
            ("Liquidus (Tl)", f"{pred['Predicted_Tl']:.1f} K"),
            ("Supercooled Region", f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.1f} K"),
            ("Critical Diameter", f"{pred['Predicted_Dmax']:.3f} mm"),
            ("Critical Cooling Rate", f"{pred['Predicted_Rc']:.2f} K/s"),
        ]
        
        for name, value in properties:
            st.markdown(f'''
            <div class="property-row">
                <span class="property-name">{name}</span>
                <span class="property-value">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        tab1, tab2 = st.tabs(["📈 Thermal Profile", "📊 GFA Indicator"])
        
        with tab1:
            fig_temp = go.Figure()
            temperatures = [pred['Predicted_Tg'], pred['Predicted_Tx'], pred['Predicted_Tl']]
            labels = ['Tg', 'Tx', 'Tl']
            colors = ['#667eea', '#764ba2', '#4c51bf']
            
            for i, (temp, label, color) in enumerate(zip(temperatures, labels, colors)):
                fig_temp.add_trace(go.Bar(
                    x=[label],
                    y=[temp],
                    name=label,
                    marker_color=color,
                    text=f"{temp:.0f} K",
                    textposition='auto',
                    hovertemplate=f"{label}: {temp:.0f} K<extra></extra>"
                ))
            
            fig_temp.update_layout(
                title="Thermal Properties",
                yaxis_title="Temperature (K)",
                showlegend=False,
                plot_bgcolor='white',
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred['Predicted_Dmax'],
                title={'text': "Glass Forming Ability", 'font': {'size': 20}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1},
                    'bar': {'color': "#667eea"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 1], 'color': '#fed7d7'},
                        {'range': [1, 3], 'color': '#feebc8'},
                        {'range': [3, 10], 'color': '#c6f6d5'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': pred['Predicted_Dmax']}
                }
            ))
            
            fig_gauge.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Download button
        results_df = pd.DataFrame({
            'Alloy': [composition],
            'Phase': [pred['Predicted_Phase']],
            'Confidence': [pred['Phase_Confidence']],
            'Tg_K': [pred['Predicted_Tg']],
            'Tx_K': [pred['Predicted_Tx']],
            'Tl_K': [pred['Predicted_Tl']],
            'Delta_T': [pred['Predicted_Tx'] - pred['Predicted_Tg']],
            'Dmax_mm': [pred['Predicted_Dmax']],
            'Rc_Ks': [pred['Predicted_Rc']]
        })
        
        st.markdown(get_download_link(results_df), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 20px; font-size: 14px;">
    BMGcalc • Advanced Metallic Glass Prediction • Demo Version
</div>
""", unsafe_allow_html=True)
