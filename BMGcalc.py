import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import periodictable
import base64

# Set page configuration
st.set_page_config(
    page_title="BMGcalc",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, high-contrast CSS
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        padding: 1.2rem 2rem;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Main container */
    .main-container {
        padding: 1.5rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Section title - HIGH CONTRAST */
    .section-title {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #00B4DB;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(0, 180, 219, 0.3);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    
    /* Element grid - COMPACT */
    .element-grid {
        display: grid;
        grid-template-columns: repeat(18, 1fr);
        gap: 2px;
        margin: 0.5rem 0;
    }
    
    .element-box {
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        background: rgba(30, 41, 59, 0.9);
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 700;
        font-size: 0.75rem;
        color: #FFFFFF;
        padding: 0 !important;
        min-height: 26px;
        height: 26px;
    }
    
    .element-box:hover {
        background: rgba(0, 180, 219, 0.3);
        border-color: #00B4DB;
        transform: scale(1.05);
    }
    
    .element-box.selected {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        color: #FFFFFF;
        border-color: #00B4DB;
        box-shadow: 0 0 8px rgba(0, 180, 219, 0.5);
    }
    
    /* Property display - HIGH CONTRAST */
    .property-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem;
        margin: 0.2rem 0;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .property-label {
        color: #E2E8F0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .property-value {
        color: #00B4DB;
        font-size: 1.1rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric card - HIGH CONTRAST */
    .metric-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid rgba(0, 180, 219, 0.4);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00B4DB;
        margin: 0.3rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-label {
        color: #CBD5E1;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #0093B3 0%, #007296 100%);
        transform: translateY(-1px);
    }
    
    /* Secondary button for show/hide */
    .secondary-button {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid #00B4DB !important;
        color: #00B4DB !important;
    }
    
    /* Gauge container */
    .gauge-container {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(0, 180, 219, 0.4);
    }
    
    /* Composition display */
    .composition-display {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(0, 180, 219, 0.2);
    }
    
    .element-tag {
        background: rgba(0, 180, 219, 0.15);
        color: #00B4DB;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(0, 180, 219, 0.3);
    }
    
    /* Remove Streamlit default elements */
    .st-emotion-cache-10trblm {padding: 0 !important;}
    .st-emotion-cache-1dp5vir {display: none !important;}
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00B4DB, #0083B0);
    }
    
    /* Warning text */
    .warning-text {
        color: #F87171 !important;
        font-weight: 600 !important;
    }
    
    /* Success text */
    .success-text {
        color: #4ADE80 !important;
        font-weight: 600 !important;
    }
    
    /* Slider labels - HIGH CONTRAST */
    .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Selected elements display */
    .selected-elements-box {
        background: rgba(0, 180, 219, 0.1);
        border: 1px solid rgba(0, 180, 219, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.8rem 0;
    }
    
    /* Composition controls */
    .composition-controls {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Fix composition toggle */
    .fix-composition-toggle {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.5);
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        color: #CBD5E1;
        font-size: 0.8rem;
        cursor: pointer;
    }
    
    .fix-composition-toggle.active {
        background: rgba(0, 180, 219, 0.2);
        border-color: #00B4DB;
        color: #00B4DB;
    }
    
    /* Compact slider container */
    .compact-slider {
        margin: 0.3rem 0;
    }
    
    /* Auto-normalize info */
    .normalize-info {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
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
if 'show_periodic_table' not in st.session_state:
    st.session_state.show_periodic_table = True
if 'normalize_mode' not in st.session_state:
    st.session_state.normalize_mode = "auto"  # "auto", "manual", "fixed"
if 'locked_elements' not in st.session_state:
    st.session_state.locked_elements = []

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
    """Create a clean gauge chart for glass forming ability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dmax_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "GLASS FORMING ABILITY",
            'font': {'size': 20, 'color': '#00B4DB', 'family': 'Space Grotesk'}
        },
        number={
            'font': {'size': 36, 'color': '#FFFFFF', 'family': 'Space Grotesk'},
            'suffix': " mm"
        },
        gauge={
            'axis': {
                'range': [0, 10],
                'tickwidth': 1,
                'tickcolor': '#CBD5E1',
                'tickfont': {'color': '#CBD5E1', 'size': 10}
            },
            'bar': {'color': "#00B4DB", 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#00B4DB",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239, 68, 68, 0.7)'},
                {'range': [1, 3], 'color': 'rgba(245, 158, 11, 0.7)'},
                {'range': [3, 5], 'color': 'rgba(34, 197, 94, 0.7)'},
                {'range': [5, 10], 'color': 'rgba(59, 130, 246, 0.7)'}],
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#CBD5E1", 'family': "Inter"}
    )
    
    return fig

def create_composition_pie(elements, fractions):
    """Create composition pie chart"""
    colors = ['#00B4DB', '#0083B0', '#006994', '#005073', '#003752', '#001F3F']
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{elem}" for elem in elements],
        values=fractions,
        hole=.4,
        marker_colors=colors[:len(elements)],
        textinfo='label+percent',
        textfont=dict(size=12, color='white', family='Inter'),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        marker=dict(line=dict(color='rgba(255,255,255,0.3)', width=1))
    )])
    
    fig.update_layout(
        showlegend=False,
        height=220,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def normalize_composition():
    """Automatically normalize composition to 100%"""
    if not st.session_state.selected_elements:
        return
    
    # Calculate current total
    current_fractions = [st.session_state.element_fractions.get(elem, 0) 
                        for elem in st.session_state.selected_elements]
    total = sum(current_fractions)
    
    # If total is 0, set equal distribution
    if total == 0:
        equal_val = 100 / len(st.session_state.selected_elements)
        for elem in st.session_state.selected_elements:
            st.session_state.element_fractions[elem] = equal_val
        return
    
    # Normalize to 100%
    scale_factor = 100 / total
    for elem in st.session_state.selected_elements:
        st.session_state.element_fractions[elem] *= scale_factor

def adjust_composition_for_lock(locked_elements, changed_element, changed_value):
    """Adjust composition when some elements are locked"""
    if not locked_elements:
        normalize_composition()
        return
    
    # Calculate remaining percentage to distribute
    locked_total = sum(st.session_state.element_fractions.get(elem, 0) 
                      for elem in locked_elements if elem != changed_element)
    
    # Add the changed value if it's locked
    if changed_element in locked_elements:
        locked_total += changed_value
    
    remaining = 100 - locked_total
    
    # Distribute remaining among unlocked elements (excluding the changed one if unlocked)
    unlocked = [elem for elem in st.session_state.selected_elements 
                if elem not in locked_elements and elem != changed_element]
    
    if unlocked:
        equal_share = remaining / len(unlocked)
        for elem in unlocked:
            st.session_state.element_fractions[elem] = equal_share

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
    href = f'''
    <a href="data:file/csv;base64,{b64}" download="{filename}" 
       style="background: linear-gradient(90deg, #10B981 0%, #059669 100%); 
              color: white; padding: 0.6rem 1.2rem; border-radius: 6px; 
              text-decoration: none; font-weight: 600; display: block; 
              text-align: center; font-size: 0.9rem;">
        📥 Download Results
    </a>
    '''
    return href

# MAIN APP
st.markdown('<div class="main-header">⚗️ BMGcalc - Metallic Glass Predictor</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Element Selection</div>', unsafe_allow_html=True)
        
        # Number of elements
        num_elements = st.select_slider(
            "Number of elements",
            options=[2, 3, 4, 5, 6],
            value=3,
            key="num_elements"
        )
        
        # Show/Hide periodic table button
        show_hide_col1, show_hide_col2 = st.columns([3, 1])
        with show_hide_col2:
            button_label = "🔽 Hide Table" if st.session_state.show_periodic_table else "🔼 Show Table"
            button_type = "secondary" if st.session_state.show_periodic_table else "primary"
            
            if st.button(button_label, key="toggle_table", type=button_type):
                st.session_state.show_periodic_table = not st.session_state.show_periodic_table
                st.rerun()
        
        # Periodic Table (shown/hidden based on state)
        if st.session_state.show_periodic_table:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Generate periodic table buttons - COMPACT VERSION
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
                                    if element in st.session_state.locked_elements:
                                        st.session_state.locked_elements.remove(element)
                                else:
                                    if len(st.session_state.selected_elements) < num_elements:
                                        st.session_state.selected_elements.append(element)
                                        if element not in st.session_state.element_fractions:
                                            # Initialize with equal distribution
                                            equal_share = 100 / (len(st.session_state.selected_elements))
                                            for elem in st.session_state.selected_elements:
                                                st.session_state.element_fractions[elem] = equal_share
                                        normalize_composition()
                                    else:
                                        st.warning(f"Maximum {num_elements} elements allowed")
                                st.rerun()
                    else:
                        with cols[col_idx]:
                            st.write("")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Selected Elements Display
        if st.session_state.selected_elements:
            st.markdown('<div class="selected-elements-box">', unsafe_allow_html=True)
            st.markdown(f'''
            <div style="color: #00B4DB; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                Selected Elements ({len(st.session_state.selected_elements)}/{num_elements})
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.3rem;">
                {''.join([f'<span class="element-tag">{elem}</span>' for elem in st.session_state.selected_elements])}
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Composition Input
        if st.session_state.selected_elements:
            st.markdown('<div class="section-title">Set Composition (%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Composition control mode
            mode_col1, mode_col2, mode_col3 = st.columns(3)
            with mode_col1:
                if st.button("🔓 Auto-Adjust", 
                           key="mode_auto",
                           help="Automatically adjust all elements to sum 100%",
                           type="primary" if st.session_state.normalize_mode == "auto" else "secondary"):
                    st.session_state.normalize_mode = "auto"
                    normalize_composition()
                    st.rerun()
            
            with mode_col2:
                if st.button("🔒 Fixed Values", 
                           key="mode_manual",
                           help="Manually set each value independently",
                           type="primary" if st.session_state.normalize_mode == "manual" else "secondary"):
                    st.session_state.normalize_mode = "manual"
                    st.rerun()
            
            with mode_col3:
                if st.button("⚡ Smart Lock", 
                           key="mode_fixed",
                           help="Lock specific elements, auto-adjust others",
                           type="primary" if st.session_state.normalize_mode == "fixed" else "secondary"):
                    st.session_state.normalize_mode = "fixed"
                    st.rerun()
            
            # Mode description
            if st.session_state.normalize_mode == "auto":
                st.markdown('<div class="normalize-info">All values auto-adjusted to sum 100%</div>', unsafe_allow_html=True)
            elif st.session_state.normalize_mode == "manual":
                st.markdown('<div class="normalize-info">Set each value independently (Total may not be 100%)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="normalize-info">Lock specific elements, others auto-adjusted</div>', unsafe_allow_html=True)
            
            # Calculate total and adjust based on mode
            total = 0
            fractions = []
            
            for elem in st.session_state.selected_elements:
                # Get current value
                current_val = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                
                # Create columns for slider and lock button
                col_left, col_mid, col_right = st.columns([3, 1, 1])
                
                with col_left:
                    # Create slider
                    new_val = st.slider(
                        elem,
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_val),
                        step=0.5,
                        key=f"slider_{elem}",
                        disabled=(st.session_state.normalize_mode == "fixed" and 
                                 elem in st.session_state.locked_elements)
                    )
                
                with col_mid:
                    st.markdown(f'<div style="color: #00B4DB; font-weight: 700; padding-top: 0.5rem;">{new_val:.1f}%</div>', unsafe_allow_html=True)
                
                with col_right:
                    if st.session_state.normalize_mode == "fixed":
                        is_locked = elem in st.session_state.locked_elements
                        lock_label = "🔓" if is_locked else "🔒"
                        if st.button(lock_label, key=f"lock_{elem}", 
                                   help="Lock/unlock this element",
                                   type="primary" if is_locked else "secondary"):
                            if elem in st.session_state.locked_elements:
                                st.session_state.locked_elements.remove(elem)
                            else:
                                st.session_state.locked_elements.append(elem)
                            st.rerun()
                
                # Update fraction
                st.session_state.element_fractions[elem] = new_val
                fractions.append(new_val)
                total += new_val
            
            # Auto-adjust based on mode
            if st.session_state.normalize_mode == "auto":
                if abs(total - 100) > 0.1:
                    scale_factor = 100 / total if total > 0 else 1
                    for elem in st.session_state.selected_elements:
                        st.session_state.element_fractions[elem] *= scale_factor
                    total = 100
            elif st.session_state.normalize_mode == "fixed" and st.session_state.locked_elements:
                # Adjust unlocked elements to sum to 100%
                locked_total = sum(st.session_state.element_fractions.get(elem, 0) 
                                  for elem in st.session_state.locked_elements)
                unlocked = [elem for elem in st.session_state.selected_elements 
                           if elem not in st.session_state.locked_elements]
                
                if unlocked:
                    remaining = 100 - locked_total
                    equal_share = remaining / len(unlocked) if remaining > 0 else 0
                    for elem in unlocked:
                        st.session_state.element_fractions[elem] = equal_share
                    total = 100
            
            # Display total with color coding
            st.progress(total/100, text=f"Total: {total:.1f}%")
            
            if abs(total - 100) > 0.1:
                st.markdown(f'<div class="warning-text">⚠️ Total should be 100% (Current: {total:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-text">✓ Composition is valid (100%)</div>', unsafe_allow_html=True)
            
            # Composition Pie Chart
            if len(st.session_state.selected_elements) > 1:
                st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                fig_pie = create_composition_pie(
                    st.session_state.selected_elements,
                    [st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements]
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Predict Button - hides periodic table when clicked
            if st.button("🚀 Predict Properties", use_container_width=True, type="primary"):
                composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                     for elem in st.session_state.selected_elements])
                with st.spinner("Analyzing alloy composition..."):
                    st.session_state.predictions = process_alloys_demo(composition)
                    st.session_state.show_periodic_table = False  # Hide periodic table after prediction
                    st.rerun()
    
    with col2:
        # Results Display
        if st.session_state.predictions is not None:
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            
            # Show/Hide periodic table button in results panel too
            if not st.session_state.show_periodic_table:
                if st.button("📋 Show Periodic Table", key="show_table_results", type="secondary"):
                    st.session_state.show_periodic_table = True
                    st.rerun()
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            pred = st.session_state.predictions
            
            # Key Metrics - HIGH CONTRAST
            cols = st.columns(3)
            with cols[0]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">GFA SCORE</div>
                    <div class="metric-value">{:.2f}</div>
                    <div style="color: #CBD5E1; font-size: 0.7rem;">Dmax (mm)</div>
                </div>
                """.format(pred['Predicted_Dmax']), unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">ΔT RANGE</div>
                    <div class="metric-value">{:.0f}</div>
                    <div style="color: #CBD5E1; font-size: 0.7rem;">Tx - Tg (K)</div>
                </div>
                """.format(pred['Predicted_Tx'] - pred['Predicted_Tg']), unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">PHASE</div>
                    <div class="metric-value">{}</div>
                    <div style="color: #CBD5E1; font-size: 0.7rem;">Confidence: {:.1%}</div>
                </div>
                """.format(pred['Predicted_Phase'], pred['Phase_Confidence']), unsafe_allow_html=True)
            
            # Property Details - HIGH CONTRAST
            properties = [
                ("Glass Transition (Tg)", f"{pred['Predicted_Tg']:.1f} K"),
                ("Crystallization (Tx)", f"{pred['Predicted_Tx']:.1f} K"),
                ("Liquidus (Tl)", f"{pred['Predicted_Tl']:.1f} K"),
                ("Critical Diameter", f"{pred['Predicted_Dmax']:.4f} mm"),
                ("Critical Cooling Rate", f"{pred['Predicted_Rc']:.3f} K/s"),
            ]
            
            for name, value in properties:
                st.markdown(f'''
                <div class="property-row">
                    <div class="property-label">{name}</div>
                    <div class="property-value">{value}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Glass Forming Ability Gauge
            st.markdown('<div class="section-title">Glass Forming Ability</div>', unsafe_allow_html=True)
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            fig_gauge = create_simple_gauge(pred['Predicted_Dmax'])
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
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
            <div class="glass-card" style="text-align: center; padding: 2rem 1.5rem;">
                <div style="font-size: 2.5rem; color: #00B4DB; margin-bottom: 0.8rem;">⚗️</div>
                <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; font-family: 'Space Grotesk', sans-serif;">
                    READY FOR ANALYSIS
                </div>
                <div style="color: #CBD5E1; font-size: 0.9rem;">
                    Select elements and set composition to get predictions
                </div>
                <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 0.5rem; font-size: 0.8rem; color: #94A3B8;">
                    <div style="text-align: center;">
                        <div style="color: #00B4DB; font-weight: 600;">🔓</div>
                        <div>Auto-Adjust</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #00B4DB; font-weight: 600;">🔒</div>
                        <div>Fixed Values</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #00B4DB; font-weight: 600;">⚡</div>
                        <div>Smart Lock</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1.5rem; margin-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 500;">
        BMGcalc v2.0 • Bulk Metallic Glass Design Platform
    </div>
</div>
""", unsafe_allow_html=True)
