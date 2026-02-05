import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import periodictable
from streamlit.components.v1 import html

# Set page configuration - NO SPACE ABOVE
st.set_page_config(
    page_title="BMGcalc",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove default Streamlit padding and margins
st.markdown("""
<style>
    /* Remove all default padding and margins */
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        background: linear-gradient(135deg, #0A192F 0%, #112240 50%, #0A192F 100%);
        min-height: 100vh;
    }
    
    /* Remove Streamlit header */
    header {visibility: hidden !important; height: 0 !important;}
    
    /* Remove Streamlit footer */
    footer {visibility: hidden !important; height: 0 !important;}
    
    /* Remove extra spacing */
    .main > div {padding: 0 !important;}
    
    /* Remove Streamlit spacing */
    .block-container {padding: 0 !important; max-width: 100% !important;}
    
    /* Custom fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Modern header */
    .main-header {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        padding: 1.5rem 0;
        text-align: center;
        margin-bottom: 0 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 180, 219, 0.3);
    }
    
    /* Main container */
    .main-container {
        padding: 2rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Section styling */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #00B4DB;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 180, 219, 0.3);
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Modern button */
    .modern-button {
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
        margin: 1rem 0;
    }
    
    .modern-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 180, 219, 0.4);
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
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
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
        transform: scale(1.05);
    }
    
    .element-box.selected {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        border-color: #00B4DB;
        box-shadow: 0 0 15px rgba(0, 180, 219, 0.5);
    }
    
    /* Property display */
    .property-display {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .property-label {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .property-value {
        color: #00B4DB;
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Composition slider */
    .composition-slider {
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
    }
    
    /* Warning message */
    .warning-message {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #F87171;
        font-weight: 500;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #4ADE80;
        font-weight: 500;
    }
    
    /* Metric display */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 180, 219, 0.3);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00B4DB;
        font-family: 'Space Grotesk', sans-serif;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0;
        border: none;
        cursor: pointer;
        text-align: center;
        width: 100%;
    }
    
    .download-btn:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
    }
    
    /* Remove all Streamlit default elements */
    .st-emotion-cache-10trblm {padding: 0 !important;}
    .st-emotion-cache-1dp5vir {display: none !important;}
    .st-emotion-cache-1kyxreq {justify-content: center;}
    
    /* Custom tabs */
    .custom-tab {
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem 1.5rem;
        border-radius: 12px 12px 0 0;
        color: #94A3B8;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .custom-tab.active {
        background: rgba(0, 180, 219, 0.2);
        color: #00B4DB;
        border-bottom: 2px solid #00B4DB;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Gauge container */
    .gauge-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(0, 180, 219, 0.3);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
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

def create_glowing_gauge(predictions):
    """Create beautiful glowing gauge chart for glass forming ability"""
    dmax = predictions['Predicted_Dmax']
    
    fig = go.Figure()
    
    # Add gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dmax,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "GLASS FORMING ABILITY",
            'font': {'size': 26, 'color': '#00B4DB', 'family': 'Space Grotesk, sans-serif'}
        },
        number={
            'font': {'size': 48, 'color': '#FFFFFF', 'family': 'Space Grotesk, sans-serif'},
            'suffix': " mm",
            'valueformat': '.3f'
        },
        gauge={
            'axis': {
                'range': [0, 10],
                'tickwidth': 2,
                'tickcolor': 'white',
                'dtick': 1,
                'tickfont': {'size': 14, 'color': '#94A3B8'}
            },
            'bar': {
                'color': '#00B4DB',
                'thickness': 0.3
            },
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239, 68, 68, 0.6)', 'name': 'Poor'},
                {'range': [1, 3], 'color': 'rgba(245, 158, 11, 0.6)', 'name': 'Fair'},
                {'range': [3, 5], 'color': 'rgba(34, 197, 94, 0.6)', 'name': 'Good'},
                {'range': [5, 10], 'color': 'rgba(59, 130, 246, 0.6)', 'name': 'Excellent'}
            ],
            'threshold': {
                'line': {
                    'color': "#FFFFFF",
                    'width': 4,
                    'dash': 'dash'
                },
                'thickness': 0.85,
                'value': dmax
            }
        }
    ))
    
    # Add custom annotations for gauge zones
    fig.add_annotation(
        x=0.5, y=0.2,
        text="POOR",
        showarrow=False,
        font=dict(size=14, color="rgba(239, 68, 68, 0.8)", family="Space Grotesk"),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.5, y=0.3,
        text="FAIR",
        showarrow=False,
        font=dict(size=14, color="rgba(245, 158, 11, 0.8)", family="Space Grotesk"),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.5, y=0.4,
        text="GOOD",
        showarrow=False,
        font=dict(size=14, color="rgba(34, 197, 94, 0.8)", family="Space Grotesk"),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text="EXCELLENT",
        showarrow=False,
        font=dict(size=14, color="rgba(59, 130, 246, 0.8)", family="Space Grotesk"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(t=100, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter, sans-serif"},
        hovermode=False
    )
    
    return fig

def create_composition_pie(elements, fractions):
    """Create beautiful composition pie chart"""
    colors = ['#00B4DB', '#0083B0', '#006994', '#005073', '#003752']
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{elem}<br>{frac}%" for elem, frac in zip(elements, fractions)],
        values=fractions,
        hole=.5,
        marker_colors=colors[:len(elements)],
        textinfo='label',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate="<b>%{label}</b><extra></extra>",
        pull=[0.1 if i == 0 else 0 for i in range(len(elements))]
    )])
    
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text='COMPOSITION',
            x=0.5, y=0.5,
            font=dict(size=16, color='#94A3B8', family='Space Grotesk'),
            showarrow=False
        )]
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
    href = f"""
    <a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">
        <button class="download-btn">
            📥 Download Results
        </button>
    </a>
    """
    return href

# MAIN APP - NO SPACE ABOVE
st.markdown('<div class="main-header">⚗️ BMGcalc - Metallic Glass Predictor</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Select Elements</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Number of elements selector
        num_elements = st.select_slider(
            "Number of elements",
            options=[2, 3, 4, 5, 6],
            value=3,
            key="num_elements"
        )
        
        # Periodic Table
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); padding: 1rem; border-radius: 12px; margin: 1rem 0;">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem; text-align: center;">
                Click elements to select (Max: {})
            </div>
            <div class="element-grid">
        """.format(num_elements), unsafe_allow_html=True)
        
        # Generate periodic table buttons
        elements_per_row = {}
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
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Display selected elements
        if st.session_state.selected_elements:
            st.markdown(f"""
            <div style="margin: 1rem 0; padding: 1rem; background: rgba(0, 180, 219, 0.1); border-radius: 12px; border: 1px solid rgba(0, 180, 219, 0.3);">
                <div style="color: #00B4DB; font-weight: 600; margin-bottom: 0.5rem;">Selected Elements:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    {''.join([f'<span style="background: rgba(0, 180, 219, 0.2); padding: 0.5rem 1rem; border-radius: 8px; color: #00B4DB; font-weight: 600;">{elem}</span>' for elem in st.session_state.selected_elements])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Composition Input with Auto-adjust
        if st.session_state.selected_elements:
            st.markdown('<div class="section-title">Set Composition (%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Normalize mode toggle
            st.session_state.normalize_mode = st.toggle(
                "Auto-normalize to 100%",
                value=True,
                help="When enabled, adjusting one element automatically adjusts others to sum to 100%"
            )
            
            # Calculate total
            fractions = []
            total = 0
            
            for elem in st.session_state.selected_elements:
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    # Get current value
                    current_val = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                    
                    # Create slider
                    new_val = st.slider(
                        elem,
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_val),
                        step=0.5,
                        key=f"slider_{elem}",
                        label_visibility="visible"
                    )
                
                with col_right:
                    # Display current value
                    st.metric("", f"{new_val:.1f}%", delta=None)
                
                fractions.append(new_val)
                total += new_val
            
            # Auto-normalize if enabled
            if st.session_state.normalize_mode and abs(total - 100) > 0.1:
                if total > 0:
                    scale_factor = 100 / total
                    for i, elem in enumerate(st.session_state.selected_elements):
                        st.session_state.element_fractions[elem] = fractions[i] * scale_factor
            
            # Update fractions in session state
            for elem, frac in zip(st.session_state.selected_elements, fractions):
                st.session_state.element_fractions[elem] = frac
            
            # Display total
            col_total1, col_total2 = st.columns([2, 1])
            with col_total1:
                st.progress(total/100, text=f"Total: {total:.1f}%")
            with col_total2:
                if abs(total - 100) <= 0.1:
                    st.success("✓ Valid")
                else:
                    st.warning(f"Needs: {100-total:.1f}%")
            
            # Composition Pie Chart
            if len(st.session_state.selected_elements) > 1:
                st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
                fig_pie = create_composition_pie(
                    st.session_state.selected_elements,
                    [st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements]
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Predict Button
            if st.button("🚀 Predict Properties", use_container_width=True, type="primary"):
                composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                     for elem in st.session_state.selected_elements])
                with st.spinner("🔬 Analyzing alloy composition..."):
                    st.session_state.predictions = process_alloys_demo(composition)
    
    with col2:
        # Results Display
        if st.session_state.predictions is not None:
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            pred = st.session_state.predictions
            
            # Key Metrics
            cols_metrics = st.columns(3)
            with cols_metrics[0]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">GFA Score</div>
                    <div class="metric-value">{:.2f}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Dmax (mm)</div>
                </div>
                """.format(pred['Predicted_Dmax']), unsafe_allow_html=True)
            
            with cols_metrics[1]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">ΔT Range</div>
                    <div class="metric-value">{:.0f}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Tx - Tg (K)</div>
                </div>
                """.format(pred['Predicted_Tx'] - pred['Predicted_Tg']), unsafe_allow_html=True)
            
            with cols_metrics[2]:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Rc</div>
                    <div class="metric-value">{:.1f}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">K/s</div>
                </div>
                """.format(pred['Predicted_Rc']), unsafe_allow_html=True)
            
            # Property Details
            st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
            
            properties = [
                ("Phase", pred['Predicted_Phase'], f"({pred['Phase_Confidence']:.1%})"),
                ("Glass Transition (Tg)", f"{pred['Predicted_Tg']:.1f} K", "onset"),
                ("Crystallization (Tx)", f"{pred['Predicted_Tx']:.1f} K", "peak"),
                ("Liquidus (Tl)", f"{pred['Predicted_Tl']:.1f} K", "complete"),
                ("Supercooled Region", f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.1f} K", "ΔT"),
                ("Critical Diameter", f"{pred['Predicted_Dmax']:.4f} mm", "Dmax"),
                ("Critical Cooling Rate", f"{pred['Predicted_Rc']:.3f} K/s", "Rc"),
            ]
            
            for name, value, note in properties:
                st.markdown(f"""
                <div class="property-display">
                    <div>
                        <div class="property-label">{name}</div>
                        <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.25rem;">{note}</div>
                    </div>
                    <div class="property-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Glass Forming Ability Gauge
            st.markdown('<div class="section-title">Glass Forming Ability</div>', unsafe_allow_html=True)
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            fig_gauge = create_glowing_gauge(pred)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Results
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
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; color: #334155; margin-bottom: 1rem;">⚗️</div>
                <div style="color: #00B4DB; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; font-family: 'Space Grotesk', sans-serif;">
                    Ready for Analysis
                </div>
                <div style="color: #94A3B8; margin-bottom: 2rem;">
                    Select elements and set composition to get predictions
                </div>
                <div style="display: flex; justify-content: center; gap: 1rem; color: #64748B; font-size: 0.9rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔬</div>
                        <div>Phase Prediction</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🌡️</div>
                        <div>Thermal Properties</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                        <div>GFA Analysis</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;">
        BMGcalc v2.0 • Bulk Metallic Glass Design Platform
    </div>
    <div style="color: #475569; font-size: 0.8rem;">
        Powered by Machine Learning • For Research & Development
    </div>
</div>
""", unsafe_allow_html=True)
