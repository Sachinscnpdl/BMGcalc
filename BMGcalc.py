import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import periodictable
import base64

# Set page configuration - NO SPACE ABOVE
st.set_page_config(
    page_title="BMGcalc",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove all default padding and margins
st.markdown("""
<style>
    /* Remove all default padding and margins */
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        background: #0A192F;
        min-height: 100vh;
    }
    
    /* Remove Streamlit header and footer */
    header {visibility: hidden !important; height: 0 !important;}
    footer {visibility: hidden !important; height: 0 !important;}
    
    /* Remove Streamlit spacing */
    .block-container {padding: 0 !important; max-width: 100% !important;}
    .main > div {padding: 0 !important;}
    
    /* Custom fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        padding: 1.5rem 2rem;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }
    
    /* Main container */
    .main-container {
        padding: 2rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Section title */
    .section-title {
        color: #00B4DB;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 180, 219, 0.3);
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Element grid */
    .element-grid {
        display: grid;
        grid-template-columns: repeat(18, 1fr);
        gap: 4px;
        margin: 1rem 0;
    }
    
    .element-box {
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        background: rgba(30, 41, 59, 0.7);
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 600;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    
    .element-box:hover {
        background: rgba(0, 180, 219, 0.2);
        border-color: #00B4DB;
    }
    
    .element-box.selected {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        border-color: #00B4DB;
    }
    
    /* Property display */
    .property-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        margin: 0.25rem 0;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
    
    .property-label {
        color: #94A3B8;
        font-size: 0.9rem;
    }
    
    .property-value {
        color: #00B4DB;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Metric card */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(0, 180, 219, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00B4DB;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    
    /* Remove Streamlit default elements */
    .st-emotion-cache-10trblm {padding: 0 !important;}
    .st-emotion-cache-1dp5vir {display: none !important;}
    
    /* Gauge container */
    .gauge-container {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 180, 219, 0.3);
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
if 'normalize_mode' not in st.session_state:
    st.session_state.normalize_mode = True

# Accurate periodic table layout
PERIODIC_TABLE = {
    1: ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
    2: ["Li", "Be", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    3: ["Na", "Mg", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
    4: ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    5: ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
    6: ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    7: ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
    8: ["", "", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "", ""],
    9: ["", "", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "", ""]
}

def create_simple_gauge(dmax_value):
    """Create a simple gauge chart that won't cause errors"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dmax_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Glass Forming Ability", 'font': {'size': 24, 'color': '#00B4DB'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00B4DB"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239, 68, 68, 0.6)'},
                {'range': [1, 3], 'color': 'rgba(245, 158, 11, 0.6)'},
                {'range': [3, 5], 'color': 'rgba(34, 197, 94, 0.6)'},
                {'range': [5, 10], 'color': 'rgba(59, 130, 246, 0.6)'}],
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"}
    )
    
    return fig

def create_composition_pie(elements, fractions):
    """Create composition pie chart"""
    colors = ['#00B4DB', '#0083B0', '#006994', '#005073', '#003752']
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{elem}" for elem in elements],
        values=fractions,
        hole=.5,
        marker_colors=colors[:len(elements)],
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    )])
    
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def normalize_fractions():
    """Normalize fractions to sum to 100%"""
    if st.session_state.selected_elements and st.session_state.element_fractions:
        total = sum(st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements)
        if total > 0:
            for elem in st.session_state.selected_elements:
                current = st.session_state.element_fractions.get(elem, 0)
                st.session_state.element_fractions[elem] = (current / total) * 100

def process_alloys_demo(composition_string):
    """Demo prediction function"""
    results = pd.DataFrame({
        'Alloys': [composition_string],
        'Predicted_Phase': ['Metallic Glass'],
        'Phase_Confidence': [np.random.uniform(0.85, 0.98)],
        'Predicted_Tg': [np.random.normal(620, 30)],
        'Predicted_Tx': [np.random.normal(700, 30)],
        'Predicted_Tl': [np.random.normal(800, 30)],
        'Predicted_Dmax': [np.random.exponential(1.2) + 2],
        'Predicted_Rc': [np.random.exponential(2) + 1]
    })
    return results.iloc[0]

def get_download_link(df, filename="bmg_predictions.csv"):
    """Generate download link"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background: #10B981; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; text-align: center; width: 100%;">📥 Download Results</a>'
    return href

# MAIN APP
st.markdown('<div class="main-header">⚗️ BMGcalc - Metallic Glass Predictor</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Select Elements</div>', unsafe_allow_html=True)
        
        # Number of elements
        num_elements = st.select_slider(
            "Number of elements",
            options=[2, 3, 4, 5, 6],
            value=3,
            key="num_elements"
        )
        
        # Periodic Table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Generate periodic table buttons
        for row_idx, row in PERIODIC_TABLE.items():
            cols = st.columns(len(row))
            for col_idx, element in enumerate(row):
                if element:
                    with cols[col_idx]:
                        is_selected = element in st.session_state.selected_elements
                        if st.button(
                            element,
                            key=f"btn_{element}_{row_idx}_{col_idx}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True
                        ):
                            if element in st.session_state.selected_elements:
                                st.session_state.selected_elements.remove(element)
                            else:
                                if len(st.session_state.selected_elements) < num_elements:
                                    st.session_state.selected_elements.append(element)
                                    # Initialize fraction
                                    if element not in st.session_state.element_fractions:
                                        st.session_state.element_fractions[element] = 100 / (len(st.session_state.selected_elements))
                                else:
                                    st.warning(f"Maximum {num_elements} elements allowed")
                            st.rerun()
                else:
                    with cols[col_idx]:
                        st.write("")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Composition Input
        if st.session_state.selected_elements:
            st.markdown('<div class="section-title">Set Composition (%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Auto-normalize toggle
            st.session_state.normalize_mode = st.toggle(
                "Auto-normalize to 100%",
                value=True,
                help="Automatically adjust fractions to sum to 100%"
            )
            
            # Calculate total
            fractions = []
            total = 0
            
            for elem in st.session_state.selected_elements:
                # Get current value
                current_val = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                
                # Create slider
                new_val = st.slider(
                    elem,
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_val),
                    step=0.5,
                    key=f"slider_{elem}"
                )
                
                fractions.append(new_val)
                total += new_val
            
            # Auto-normalize if enabled
            if st.session_state.normalize_mode and abs(total - 100) > 0.1 and total > 0:
                scale_factor = 100 / total
                for i, elem in enumerate(st.session_state.selected_elements):
                    st.session_state.element_fractions[elem] = fractions[i] * scale_factor
            else:
                for elem, frac in zip(st.session_state.selected_elements, fractions):
                    st.session_state.element_fractions[elem] = frac
            
            # Display total
            st.progress(total/100, text=f"Total: {total:.1f}%")
            
            if abs(total - 100) > 0.1:
                st.warning(f"Total should be 100% (Current: {total:.1f}%)")
            
            # Composition Pie Chart
            if len(st.session_state.selected_elements) > 1:
                fig_pie = create_composition_pie(
                    st.session_state.selected_elements,
                    [st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements]
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Predict Button
            if st.button("🚀 Predict Properties", use_container_width=True):
                composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                     for elem in st.session_state.selected_elements])
                with st.spinner("Analyzing alloy composition..."):
                    st.session_state.predictions = process_alloys_demo(composition)
    
    with col2:
        # Results Display
        if st.session_state.predictions is not None:
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            pred = st.session_state.predictions
            
            # Key Metrics
            cols = st.columns(3)
            with cols[0]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">GFA Score</div>
                    <div class="metric-value">{:.2f}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Dmax (mm)</div>
                </div>
                """.format(pred['Predicted_Dmax']), unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">ΔT Range</div>
                    <div class="metric-value">{:.0f}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">K</div>
                </div>
                """.format(pred['Predicted_Tx'] - pred['Predicted_Tg']), unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Phase</div>
                    <div class="metric-value">{}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Confidence: {:.1%}</div>
                </div>
                """.format(pred['Predicted_Phase'], pred['Phase_Confidence']), unsafe_allow_html=True)
            
            # Property Details
            properties = [
                ("Glass Transition (Tg)", f"{pred['Predicted_Tg']:.1f} K"),
                ("Crystallization (Tx)", f"{pred['Predicted_Tx']:.1f} K"),
                ("Liquidus (Tl)", f"{pred['Predicted_Tl']:.1f} K"),
                ("Critical Diameter", f"{pred['Predicted_Dmax']:.4f} mm"),
                ("Critical Cooling Rate", f"{pred['Predicted_Rc']:.3f} K/s"),
            ]
            
            for name, value in properties:
                st.markdown(f"""
                <div class="property-row">
                    <div class="property-label">{name}</div>
                    <div class="property-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Glass Forming Ability Gauge
            st.markdown('<div class="section-title">Glass Forming Ability</div>', unsafe_allow_html=True)
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            fig_gauge = create_simple_gauge(pred['Predicted_Dmax'])
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Results
            composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                 for elem in st.session_state.selected_elements])
            results_df = pd.DataFrame({
                'Alloy': [composition],
                'Phase': [pred['Predicted_Phase']],
                'Confidence': [pred['Phase_Confidence']],
                'Tg_K': [pred['Predicted_Tg']],
                'Tx_K': [pred['Predicted_Tx']],
                'Tl_K': [pred['Predicted_Tl']],
                'Delta_T_K': [pred['Predicted_Tx'] - pred['Predicted_Tg']],
                'Dmax_mm': [pred['Predicted_Dmax']],
                'Rc_Ks': [pred['Predicted_Rc']]
            })
            
            st.markdown(get_download_link(results_df), unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; color: #334155; margin-bottom: 1rem;">⚗️</div>
                <div style="color: #00B4DB; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
                    Ready for Analysis
                </div>
                <div style="color: #94A3B8;">
                    Select elements and set composition to get predictions
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div style="color: #64748B; font-size: 0.9rem;">
        BMGcalc v2.0 • Bulk Metallic Glass Design Platform
    </div>
</div>
""", unsafe_allow_html=True)
