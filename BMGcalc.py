import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import periodictable

# Set page configuration
st.set_page_config(
    page_title="BMGcalc",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with elegant gradient and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF6347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        padding: 1rem;
        text-shadow: 0 2px 20px rgba(255, 165, 0, 0.3);
        animation: shimmer 3s infinite linear;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 28px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1600px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.4);
    }
    
    .section-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFD700;
        margin-bottom: 25px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .periodic-table-container {
        background: rgba(10, 10, 40, 0.7);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .periodic-grid {
        display: grid;
        grid-template-columns: repeat(18, minmax(40px, 1fr));
        gap: 6px;
        margin: 20px 0;
    }
    
    .element-box {
        aspect-ratio: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px solid;
        border-radius: 12px;
        background: rgba(20, 20, 60, 0.8);
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .element-box:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }
    
    .element-box:hover:before {
        left: 100%;
    }
    
    .element-box:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
        z-index: 10;
    }
    
    .element-box.selected {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0f0c29;
        border-color: #FFD700;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.5);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 25px rgba(255, 215, 0, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.8); }
        100% { box-shadow: 0 0 25px rgba(255, 215, 0, 0.5); }
    }
    
    .element-symbol {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .element-number {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 2px;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(30, 30, 70, 0.9), rgba(20, 20, 50, 0.9));
        border-radius: 20px;
        padding: 35px;
        margin: 25px 0;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FFD700, #FF6347, #9D4EDD);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    .metric-tile {
        background: linear-gradient(135deg, rgba(40, 40, 80, 0.8), rgba(30, 30, 60, 0.8));
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-tile:hover {
        transform: translateY(-6px);
        border-color: rgba(255, 215, 0, 0.4);
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.2);
    }
    
    .metric-tile:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FFD700, #FFA500);
    }
    
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFD700;
        margin: 10px 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #A0A0D0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .property-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        transition: background 0.3s ease;
    }
    
    .property-row:hover {
        background: rgba(255, 255, 255, 0.03);
    }
    
    .property-name {
        font-size: 1rem;
        font-weight: 500;
        color: #E0E0FF;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .property-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFD700;
        background: rgba(255, 215, 0, 0.1);
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        min-width: 140px;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #0f0c29;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        padding: 18px 40px;
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.5px;
        text-transform: uppercase;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover:before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(255, 165, 0, 0.4);
    }
    
    .fraction-input-container {
        background: rgba(30, 30, 60, 0.8);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .element-fraction {
        display: flex;
        align-items: center;
        gap: 20px;
        margin: 15px 0;
        padding: 15px;
        background: rgba(40, 40, 80, 0.5);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .element-fraction:hover {
        background: rgba(50, 50, 100, 0.7);
        border-color: rgba(255, 215, 0, 0.3);
    }
    
    .fraction-slider {
        flex: 1;
    }
    
    .fraction-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #FFD700;
        min-width: 60px;
        text-align: center;
    }
    
    .tab-container {
        background: rgba(20, 20, 50, 0.8);
        border-radius: 20px;
        margin-top: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Remove Streamlit default elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 20, 40, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FFD700, #FFA500);
        border-radius: 10px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FFD700, #FFA500);
    }
    
    /* Alert boxes */
    .alert-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 107, 107, 0.05));
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #FF6B6B;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(100, 149, 237, 0.1), rgba(100, 149, 237, 0.05));
        border: 1px solid rgba(100, 149, 237, 0.3);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #6495ED;
        font-weight: 500;
    }
    
    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1a1a3a;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(255, 215, 0, 0.3);
        font-size: 0.9rem;
        font-weight: 400;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(135deg, #9D4EDD 0%, #7B2CBF 100%);
        color: white;
        padding: 15px 35px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        font-family: 'Montserrat', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(157, 78, 221, 0.4);
        color: white;
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

# Accurate periodic table layout (18 groups, 7 periods + lanthanides/actinides)
PERIODIC_TABLE_LAYOUT = {
    # Row: [elements in order with empty strings for gaps]
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

# Element group colors for better visualization
GROUP_COLORS = {
    "alkali": "#FF6B6B",
    "alkaline_earth": "#FFA500",
    "transition": "#4ECDC4",
    "post_transition": "#45B7D1",
    "metalloids": "#96CEB4",
    "nonmetals": "#FFEAA7",
    "halogens": "#DDA0DD",
    "noble_gases": "#98D8C8",
    "lanthanides": "#F7DC6F",
    "actinides": "#F1948A"
}

def get_element_group(element):
    """Get group for element color coding"""
    groups = {
        "alkali": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
        "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
        "transition": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                      "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                      "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                      "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"],
        "lanthanides": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
        "actinides": ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"],
        "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te", "Po"],
        "halogens": ["F", "Cl", "Br", "I", "At"],
        "noble_gases": ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
        "post_transition": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"],
        "nonmetals": ["H", "C", "N", "O", "P", "S", "Se"]
    }
    
    for group, elements in groups.items():
        if element in elements:
            return group
    return "other"

def generate_accurate_periodic_table():
    """Generate accurate periodic table with proper layout"""
    st.markdown('<div class="periodic-table-container">', unsafe_allow_html=True)
    
    for row in range(1, 10):
        cols = st.columns(18)
        elements = PERIODIC_TABLE_LAYOUT.get(row, [""]*18)
        
        for col_idx, (col, element) in enumerate(zip(cols, elements)):
            if element:
                is_selected = element in st.session_state.selected_elements
                group = get_element_group(element)
                color = GROUP_COLORS.get(group, "#6666CC")
                
                # Create element box with proper styling
                element_html = f"""
                <div class="element-box {'selected' if is_selected else ''}" 
                     style="border-color: {color if not is_selected else '#FFD700'};
                            background: {'linear-gradient(135deg, #FFD700, #FFA500)' if is_selected else f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'}"
                     onclick="this.dispatchEvent(new Event('click'))">
                    <div class="element-symbol">{element}</div>
                    <div class="element-number">{periodictable.elements.symbol(element).number if hasattr(periodictable.elements, 'symbol') else ''}</div>
                </div>
                """
                
                with col:
                    if st.button(element, key=f"pt_{row}_{col}_{element}", 
                               help=f"Click to select {element}",
                               type="primary" if is_selected else "secondary"):
                        if element in st.session_state.selected_elements:
                            st.session_state.selected_elements.remove(element)
                        else:
                            if len(st.session_state.selected_elements) < 10:  # Limit to 10 elements
                                st.session_state.selected_elements.append(element)
                        st.rerun()
            else:
                with col:
                    st.write("")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_beautiful_thermal_plot(predictions):
    """Create beautiful thermal properties plot"""
    fig = go.Figure()
    
    # Create gradient bar chart
    temps = [predictions['Predicted_Tg'], predictions['Predicted_Tx'], predictions['Predicted_Tl']]
    labels = ['T<sub>g</sub>', 'T<sub>x</sub>', 'T<sub>l</sub>']
    colors = ['#FFD700', '#FF8C00', '#FF4500']
    
    # Add bars with gradient effect
    for i, (temp, label, color) in enumerate(zip(temps, labels, colors)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[temp],
            name=label,
            marker=dict(
                color=color,
                line=dict(color='white', width=2)
            ),
            hovertemplate=f"<b>{label}</b><br>{temp:.1f} K<extra></extra>",
            text=[f"{temp:.0f} K"],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
    
    fig.update_layout(
        title=dict(
            text="Thermal Properties",
            font=dict(size=24, color='#FFD700', family='Montserrat'),
            x=0.5
        ),
        plot_bgcolor='rgba(20, 20, 50, 0.8)',
        paper_bgcolor='rgba(20, 20, 50, 0.8)',
        font=dict(color='#E0E0FF', family='Poppins'),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(
            tickfont=dict(size=16, color='#FFD700'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title=dict(text="Temperature (K)", font=dict(size=18, color='#FFD700')),
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=14)
        ),
        showlegend=False
    )
    
    return fig

def create_beautiful_gauge(predictions):
    """Create beautiful gauge chart for glass forming ability"""
    dmax = predictions['Predicted_Dmax']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dmax,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Glass Forming Ability",
            'font': {'size': 24, 'color': '#FFD700', 'family': 'Montserrat'}
        },
        delta={'reference': 3, 'font': {'size': 20, 'color': 'white'}},
        number={
            'font': {'size': 40, 'color': '#FFD700', 'family': 'Montserrat'},
            'suffix': " mm"
        },
        gauge={
            'axis': {
                'range': [None, 10],
                'tickwidth': 2,
                'tickcolor': 'white',
                'tickfont': {'size': 14, 'color': 'white'}
            },
            'bar': {'color': "#FFD700"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 3,
            'bordercolor': "#FFD700",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(255, 107, 107, 0.6)'},
                {'range': [1, 3], 'color': 'rgba(255, 193, 7, 0.6)'},
                {'range': [3, 10], 'color': 'rgba(76, 175, 80, 0.6)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.85,
                'value': dmax}
        }
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor='rgba(20, 20, 50, 0.8)',
        font={'color': "white", 'family': "Poppins"}
    )
    
    return fig

def create_composition_visualization(elements, fractions):
    """Create pie chart for composition visualization"""
    colors = ['#FFD700', '#FF8C00', '#FF4500', '#9D4EDD', '#4CC9F0', '#4361EE', '#3A0CA3']
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{elem} ({frac}%)" for elem, frac in zip(elements, fractions)],
        values=fractions,
        hole=.4,
        marker_colors=colors[:len(elements)],
        textinfo='label+percent',
        textfont=dict(size=14, color='white', family='Poppins'),
        hovertemplate="<b>%{label}</b><br>Fraction: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title=dict(
            text="Alloy Composition",
            font=dict(size=22, color='#FFD700', family='Montserrat'),
            x=0.5
        ),
        showlegend=False,
        height=400,
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor='rgba(20, 20, 50, 0.8)'
    )
    
    return fig

# Mock prediction function
def process_alloys_demo(composition_string):
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

# Main App
st.markdown('<h1 class="main-header">⚗️ BMGcalc</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #A0A0D0; font-size: 1.2rem; margin-bottom: 2rem;">Bulk Metallic Glass Property Predictor</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Input Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-title">Element Selection</div>', unsafe_allow_html=True)
        
        # Number of elements
        num_elements = st.select_slider(
            "Number of Elements",
            options=[2, 3, 4, 5, 6, 7, 8],
            value=3,
            help="Select the number of elements in your alloy"
        )
        
        # Display periodic table
        generate_accurate_periodic_table()
        
        if st.session_state.selected_elements:
            if len(st.session_state.selected_elements) > num_elements:
                st.session_state.selected_elements = st.session_state.selected_elements[:num_elements]
                st.info(f"Limited to {num_elements} elements")
    
    with col2:
        st.markdown('<div class="section-title">Composition Setup</div>', unsafe_allow_html=True)
        
        if st.session_state.selected_elements:
            st.markdown('<div class="fraction-input-container">', unsafe_allow_html=True)
            
            total = 0
            fractions = []
            
            for elem in st.session_state.selected_elements:
                default_val = 100 / len(st.session_state.selected_elements)
                fraction = st.slider(
                    f"{elem}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_val),
                    step=0.5,
                    key=f"frac_{elem}",
                    help=f"Adjust percentage for {elem}"
                )
                st.session_state.element_fractions[elem] = fraction
                total += fraction
                fractions.append(fraction)
            
            st.progress(total/100, text=f"Total Composition: {total:.1f}%")
            
            if abs(total - 100) > 0.1:
                st.markdown(f'<div class="alert-box">⚠️ Total must be 100% (Current: {total:.1f}%)</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Composition visualization
            if len(st.session_state.selected_elements) > 1:
                fig_pie = create_composition_visualization(
                    st.session_state.selected_elements, 
                    fractions
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Select elements from the periodic table</div>', unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("🚀 Predict Properties", use_container_width=True):
        if st.session_state.selected_elements and abs(total - 100) < 0.1:
            composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                 for elem in st.session_state.selected_elements])
            with st.spinner("🔬 Analyzing alloy composition..."):
                st.session_state.predictions = process_alloys_demo(composition)
    
    # Results Section
    if st.session_state.predictions is not None:
        st.markdown("---")
        pred = st.session_state.predictions
        
        # Key Metrics
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        
        # Metrics grid
        metric_data = [
            ("Phase", pred['Predicted_Phase'], f"{pred['Phase_Confidence']:.1%}"),
            ("T<sub>g</sub>", f"{pred['Predicted_Tg']:.0f} K", "Glass Transition"),
            ("T<sub>x</sub>", f"{pred['Predicted_Tx']:.0f} K", "Crystallization"),
            ("T<sub>l</sub>", f"{pred['Predicted_Tl']:.0f} K", "Liquidus"),
            ("ΔT", f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.0f} K", "Supercooled Region"),
            ("D<sub>max</sub>", f"{pred['Predicted_Dmax']:.3f} mm", "Critical Diameter"),
            ("R<sub>c</sub>", f"{pred['Predicted_Rc']:.2f} K/s", "Cooling Rate"),
        ]
        
        cols = st.columns(4)
        for idx, (label, value, description) in enumerate(metric_data):
            with cols[idx % 4]:
                st.markdown(f'''
                <div class="metric-tile">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size: 0.8rem; color: #A0A0D0; margin-top: 5px;">{description}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Detailed Properties
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.4rem; font-weight: 600; color: #FFD700; margin-bottom: 25px; font-family: Montserrat;">Detailed Properties</div>', unsafe_allow_html=True)
        
        properties = [
            ("Phase Classification", f"{pred['Predicted_Phase']}", f"Confidence: {pred['Phase_Confidence']:.1%}"),
            ("Glass Transition Temperature (T<sub>g</sub>)", f"{pred['Predicted_Tg']:.1f} K", "Onset of glass transition"),
            ("Crystallization Temperature (T<sub>x</sub>)", f"{pred['Predicted_Tx']:.1f} K", "Start of crystallization"),
            ("Liquidus Temperature (T<sub>l</sub>)", f"{pred['Predicted_Tl']:.1f} K", "Complete melting"),
            ("Supercooled Liquid Region (ΔT)", f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.1f} K", "Thermal stability window"),
            ("Critical Diameter (D<sub>max</sub>)", f"{pred['Predicted_Dmax']:.4f} mm", "Maximum glassy dimension"),
            ("Critical Cooling Rate (R<sub>c</sub>)", f"{pred['Predicted_Rc']:.3f} K/s", "Required cooling speed"),
        ]
        
        for name, value, desc in properties:
            st.markdown(f'''
            <div class="property-row">
                <div class="property-name">
                    <span>{name}</span>
                    <small style="font-size: 0.85rem; color: #A0A0D0; margin-left: 10px;">{desc}</small>
                </div>
                <div class="property-value">{value}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Thermal Analysis", "🎯 GFA Indicator", "📈 Property Matrix"])
        
        with tab1:
            fig_thermal = create_beautiful_thermal_plot(pred)
            st.plotly_chart(fig_thermal, use_container_width=True)
        
        with tab2:
            fig_gauge = create_beautiful_gauge(pred)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with tab3:
            # Property matrix visualization
            properties_matrix = {
                'Property': ['GFA', 'Thermal Stability', 'Glass Formability', 'Processability'],
                'Score': [
                    min(pred['Predicted_Dmax'] * 10, 100),
                    min((pred['Predicted_Tx'] - pred['Predicted_Tg']) * 0.5, 100),
                    min((1000 / max(pred['Predicted_Rc'], 1)), 100),
                    min((800 - pred['Predicted_Tl']) * 0.3, 100)
                ]
            }
            
            fig_matrix = px.bar(
                properties_matrix,
                x='Property',
                y='Score',
                color='Score',
                color_continuous_scale='Viridis',
                range_color=[0, 100]
            )
            
            fig_matrix.update_layout(
                title="Property Matrix Score",
                plot_bgcolor='rgba(20, 20, 50, 0.8)',
                paper_bgcolor='rgba(20, 20, 50, 0.8)',
                font=dict(color='#E0E0FF', family='Poppins'),
                height=400,
                xaxis=dict(tickfont=dict(color='#FFD700')),
                yaxis=dict(tickfont=dict(color='#FFD700'))
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); padding: 30px; font-size: 0.9rem; margin-top: 20px;">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 10px;">
        <span>🔬 Advanced Materials Prediction</span>
        <span>⚡ Machine Learning Powered</span>
        <span>🎯 High Accuracy Models</span>
    </div>
    <div>BMGcalc v2.0 • © 2024 • Alloy Design Platform</div>
</div>
""", unsafe_allow_html=True)
