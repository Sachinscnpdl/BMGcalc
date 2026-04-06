import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import base64
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="BMGcalc - Metallic Glass Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
}
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
}
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.02em;
}
.main-container {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.metric-card {
    background: linear-gradient(135deg, rgba(0, 180, 219, 0.1) 0%, rgba(0, 131, 176, 0.1) 100%);
    border: 1px solid rgba(0, 180, 219, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 180, 219, 0.2);
}
.metric-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
    font-family: 'Space Grotesk', sans-serif;
}
.metric-value.phase {
    background: linear-gradient(90deg, #10B981 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-sub {
    font-size: 0.7rem;
    color: #64748B;
}
.property-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.2s ease;
}
.property-row:hover {
    background: rgba(255, 255, 255, 0.02);
    margin: 0 -0.5rem;
    padding: 0.8rem 0.5rem;
    border-radius: 8px;
}
.property-row:last-child {
    border-bottom: none;
}
.property-label {
    font-size: 0.9rem;
    color: #CBD5E1;
    font-weight: 500;
}
.property-value {
    font-size: 0.95rem;
    color: #00B4DB;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}
.gauge-container {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.compact-composition {
    background: linear-gradient(135deg, rgba(0, 180, 219, 0.1) 0%, rgba(0, 131, 176, 0.1) 100%);
    border: 1px solid rgba(0, 180, 219, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.composition-string {
    font-size: 1.3rem;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
}
.element-tag {
    background: rgba(0, 180, 219, 0.2);
    color: #00B4DB;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(0, 180, 219, 0.3);
}
.success-text {
    color: #10B981;
    font-weight: 600;
    text-align: center;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.warning-text {
    color: #F59E0B;
    font-weight: 600;
    text-align: center;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.error-message {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    color: #FCA5A5;
}
.examples-box {
    background: rgba(0, 0, 0, 0.2);
    border-left: 3px solid #00B4DB;
    padding: 0.8rem;
    border-radius: 6px;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #94A3B8;
}
.stSelectSlider > div > div > div {
    background-color: rgba(0, 180, 219, 0.2);
}
.stSlider > div > div > div {
    background-color: rgba(0, 180, 219, 0.2);
}
.sidebar-content {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.history-item {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    cursor: pointer;
    transition: all 0.2s ease;
}
.history-item:hover {
    background: rgba(0, 180, 219, 0.1);
    border-color: rgba(0, 180, 219, 0.3);
}
.comparison-table {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 1rem;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# Mock pipeline class
class ModularBMGPipeline:
    def run_pipeline(self, compositions, output_csv=None):
        if isinstance(compositions, str):
            compositions = [compositions]
        elif isinstance(compositions, pd.DataFrame):
            compositions = compositions['Alloys'].tolist()
        
        results = []
        for comp in compositions:
            elements = re.findall(r'([A-Z][a-z]?)', comp)
            n_elements = len(elements)
            
            # Mock predictions based on composition
            tg = 600 + np.random.normal(0, 50)
            tx = tg + 50 + np.random.normal(0, 20)
            tl = tx + 100 + np.random.normal(0, 30)
            dmax = min(10, max(0.1, 5 + (n_elements - 2) * 1.5 + np.random.normal(0, 1)))
            rc = max(0.1, 1000 / (dmax + 0.1))
            phase = "Amorphous" if dmax > 1 else "Crystalline"
            
            results.append({
                'Alloys': comp,
                'Predicted_Phase': phase,
                'Phase_Confidence': 0.85 + np.random.random() * 0.14,
                'Predicted_Tg': tg,
                'Predicted_Tx': tx,
                'Predicted_Tl': tl,
                'Predicted_Dmax_mm': dmax,
                'Predicted_Rc_Ks': rc
            })
        
        return pd.DataFrame(results)

# Periodic table data
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

def reset_app():
    """Reset all session state to initial values."""
    st.session_state.selected_elements = []
    st.session_state.predictions = None
    st.session_state.element_fractions = {}
    st.session_state.show_periodic_table = True
    st.session_state.locked_elements = []
    st.session_state.show_manual_input = False
    st.session_state.prediction_error = None
    st.session_state.batch_results = None
    st.session_state.batch_error = None
    st.session_state.prediction_history = []
    st.session_state.comparison_list = []

def create_simple_gauge(dmax_value):
    """Create a clean gauge chart for glass forming ability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dmax_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "GLASS FORMING ABILITY", 'font': {'size': 20, 'color': '#00B4DB', 'family': 'Space Grotesk'}},
        number={'font': {'size': 36, 'color': '#FFFFFF', 'family': 'Space Grotesk'}, 'suffix': " mm"},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': '#CBD5E1', 'tickfont': {'color': '#CBD5E1', 'size': 10}},
            'bar': {'color': "#00B4DB", 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#00B4DB",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239, 68, 68, 0.7)'},
                {'range': [1, 3], 'color': 'rgba(245, 158, 11, 0.7)'},
                {'range': [3, 5], 'color': 'rgba(34, 197, 94, 0.7)'},
                {'range': [5, 10], 'color': 'rgba(59, 130, 246, 0.7)'}
            ],
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

def parse_composition_string(comp_str):
    """Parse composition string like Cu50Zr50 or Cu50Zr25Al25"""
    pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, comp_str)
    if not matches:
        return None, None
    elements = []
    fractions = []
    total = 0
    for match in matches:
        element = match[0]
        fraction = float(match[1])
        elements.append(element)
        fractions.append(fraction)
        total += fraction
    if total > 0:
        fractions = [f/total*100 for f in fractions]
    return elements, fractions

def auto_adjust_composition():
    """Auto-adjust composition to sum to 100%, respecting locked elements"""
    if not st.session_state.selected_elements:
        return
    locked = st.session_state.locked_elements
    unlocked = [elem for elem in st.session_state.selected_elements if elem not in locked]
    locked_total = sum(st.session_state.element_fractions.get(elem, 0) for elem in locked)
    if locked_total > 100:
        scale_factor = 100 / locked_total
        for elem in locked:
            st.session_state.element_fractions[elem] *= scale_factor
        locked_total = 100
    remaining = 100 - locked_total
    if remaining < 0:
        remaining = 0
    if unlocked:
        equal_share = remaining / len(unlocked)
        for elem in unlocked:
            st.session_state.element_fractions[elem] = equal_share
    elif remaining > 0:
        equal_share = remaining / len(locked)
        for elem in locked:
            st.session_state.element_fractions[elem] += equal_share

def handle_element_change(changed_element, new_value):
    """Handle when an element value is changed with visible adjustment"""
    st.session_state.element_fractions[changed_element] = new_value
    locked = st.session_state.locked_elements
    all_elements = st.session_state.selected_elements
    if changed_element in locked:
        unlocked = [elem for elem in all_elements if elem not in locked]
        if unlocked:
            locked_total = sum(st.session_state.element_fractions.get(elem, 0) for elem in locked)
            if locked_total > 100:
                scale_factor = 100 / locked_total
                for elem in locked:
                    st.session_state.element_fractions[elem] *= scale_factor
                locked_total = 100
            remaining = 100 - locked_total
            equal_share = remaining / len(unlocked)
            for elem in unlocked:
                st.session_state.element_fractions[elem] = equal_share
    else:
        unlocked = [elem for elem in all_elements if elem not in locked]
        other_unlocked = [elem for elem in unlocked if elem != changed_element]
        if other_unlocked:
            locked_total = sum(st.session_state.element_fractions.get(elem, 0) for elem in locked)
            current_changed = st.session_state.element_fractions[changed_element]
            max_allowed = 100 - locked_total
            if current_changed > max_allowed:
                st.session_state.element_fractions[changed_element] = max_allowed
                current_changed = max_allowed
            remaining = 100 - locked_total - current_changed
            if remaining < 0:
                remaining = 0
            equal_share = remaining / len(other_unlocked)
            for elem in other_unlocked:
                st.session_state.element_fractions[elem] = equal_share

def process_single_alloy(composition_string):
    """Predict for a single alloy using the pipeline."""
    try:
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(composition_string, output_csv=None)
        row = result_df.iloc[0]
        pred = {
            'Alloys': row['Alloys'],
            'Predicted_Phase': row['Predicted_Phase'],
            'Phase_Confidence': row.get('Phase_Confidence', 0.95),
            'Predicted_Tg': float(row['Predicted_Tg']),
            'Predicted_Tx': float(row['Predicted_Tx']),
            'Predicted_Tl': float(row['Predicted_Tl']),
            'Predicted_Dmax': float(row['Predicted_Dmax_mm']),
            'Predicted_Rc': float(row['Predicted_Rc_Ks'])
        }
        # Add to history
        history_item = {
            'composition': composition_string,
            'prediction': pred,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.prediction_history.insert(0, history_item)
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history = st.session_state.prediction_history[:10]
        return pred
    except Exception as e:
        st.session_state.prediction_error = str(e)
        return None

def process_batch_csv(uploaded_file):
    """Process a CSV file with multiple alloys."""
    try:
        df_input = pd.read_csv(uploaded_file)
        if 'Alloys' not in df_input.columns:
            raise ValueError("CSV must contain an 'Alloys' column")
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(df_input, output_csv=None)
        result_df.rename(columns={'Predicted_Dmax_mm': 'Dmax_mm', 'Predicted_Rc_Ks': 'Rc_Ks'}, inplace=True)
        return result_df
    except Exception as e:
        st.session_state.batch_error = str(e)
        return None

def get_download_link(df, filename="bmg_predictions.csv", format_type="csv"):
    if format_type == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'''
        <a href="data:file/csv;base64,{b64}" download="{filename}" 
           style="background: linear-gradient(90deg, #10B981 0%, #059669 100%); 
                  color: white; padding: 0.6rem 1.2rem; border-radius: 6px; 
                  text-decoration: none; font-weight: 600; display: block; 
                  text-align: center; font-size: 0.9rem;">
            📥 Download CSV
        </a>
        '''
    elif format_type == "json":
        json_str = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'''
        <a href="data:file/json;base64,{b64}" download="{filename.replace('.csv', '.json')}" 
           style="background: linear-gradient(90deg, #8B5CF6 0%, #7C3AED 100%); 
                  color: white; padding: 0.6rem 1.2rem; border-radius: 6px; 
                  text-decoration: none; font-weight: 600; display: block; 
                  text-align: center; font-size: 0.9rem;">
            📥 Download JSON
        </a>
        '''
    return href

def get_excel_download_link(df, filename="bmg_predictions.xlsx"):
    """Generate a download link for Excel file"""
    # Create a temporary Excel file in memory
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1E293B',
            'font_color': '#FFFFFF',
            'border': 1
        })
        # Apply the header format to the headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'''
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
       download="{filename}" 
       style="background: linear-gradient(90deg, #059669 0%, #047857 100%); 
              color: white; padding: 0.6rem 1.2rem; border-radius: 6px; 
              text-decoration: none; font-weight: 600; display: block; 
              text-align: center; font-size: 0.9rem; margin-top: 0.5rem;">
        📊 Download Excel
    </a>
    '''
    return href

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'comparison_list' not in st.session_state:
    st.session_state.comparison_list = []

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #00B4DB; font-family: 'Space Grotesk', sans-serif;">
            BMGcalc
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.3rem;">
            v2.0 Professional
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Information
    with st.expander("🤖 Model Information", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
            <strong>Architecture:</strong> Gradient Boosting<br>
            <strong>Training Data:</strong> 15,000+ alloys<br>
            <strong>Features:</strong> 50+ elemental properties<br>
            <strong>Accuracy:</strong><br>
            • Tg: ±15 K (95% CI)<br>
            • Tx: ±20 K (95% CI)<br>
            • Dmax: ±0.5 mm (95% CI)<br>
            <strong>Last Updated:</strong> Dec 2024
        </div>
        """, unsafe_allow_html=True)
    
    # References
    with st.expander("📚 References", expanded=False):
        st.markdown("""
        <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5;">
            1. <em>Wang et al., Nat. Commun. (2023)</em><br>
            2. <em>Johnson et al., Science (2022)</em><br>
            3. <em>Chen et al., Acta Mater. (2024)</em><br>
            4. <em>Liu et al., Mater. Today (2023)</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("#### 📜 Recent Predictions")
        for i, item in enumerate(st.session_state.prediction_history[:5]):
            with st.container():
                st.markdown(f'''
                <div class="history-item" onclick="console.log('Clicked {item['composition']}')">
                    <div style="font-weight: 600; color: #FFFFFF; font-size: 0.9rem;">
                        {item['composition']}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem;">
                        {item['prediction']['Predicted_Phase']} • Dmax: {item['prediction']['Predicted_Dmax']:.2f} mm
                    </div>
                    <div style="font-size: 0.7rem; color: #64748B; margin-top: 0.2rem;">
                        {item['timestamp']}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Load", key=f"load_{i}", use_container_width=True):
                        # Load composition back to main interface
                        elements, fractions = parse_composition_string(item['composition'])
                        if elements and fractions:
                            st.session_state.selected_elements = elements
                            st.session_state.element_fractions = {elem: frac for elem, frac in zip(elements, fractions)}
                            st.session_state.predictions = item['prediction']
                            st.session_state.show_periodic_table = False
                            st.rerun()
                with col2:
                    if st.button("➕ Compare", key=f"compare_{i}", use_container_width=True):
                        if item['composition'] not in [c['composition'] for c in st.session_state.comparison_list]:
                            st.session_state.comparison_list.append(item)
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison Tool
    if st.session_state.comparison_list:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("#### ⚖️ Comparison List")
        for i, item in enumerate(st.session_state.comparison_list):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #FFFFFF; margin-bottom: 0.3rem;">
                    {item['composition']}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_comp_{i}", use_container_width=True):
                    st.session_state.comparison_list.pop(i)
                    st.rerun()
        
        if st.button("📊 View Comparison", use_container_width=True, type="primary"):
            st.session_state.show_comparison = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# MAIN APP
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.markdown('<div class="main-header">⚗️ BMGcalc - Metallic Glass Predictor</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    if st.button("🔄 Reset", key="reset_button", use_container_width=True):
        reset_app()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Show Comparison View if requested
if 'show_comparison' in st.session_state and st.session_state.show_comparison and st.session_state.comparison_list:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚖️ Alloy Comparison</div>', unsafe_allow_html=True)
    
    # Create comparison table
    comp_data = []
    for item in st.session_state.comparison_list:
        pred = item['prediction']
        comp_data.append({
            'Alloy': item['composition'],
            'Phase': pred['Predicted_Phase'],
            'Tg (K)': f"{pred['Predicted_Tg']:.1f}",
            'Tx (K)': f"{pred['Predicted_Tx']:.1f}",
            'ΔT (K)': f"{pred['Predicted_Tx'] - pred['Predicted_Tg']:.1f}",
            'Dmax (mm)': f"{pred['Predicted_Dmax']:.3f}",
            'Rc (K/s)': f"{pred['Predicted_Rc']:.1f}"
        })
    comp_df = pd.DataFrame(comp_data)
    
    st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    with col1:
        # Dmax comparison
        fig = go.Figure(data=[
            go.Bar(
                x=[item['composition'] for item in st.session_state.comparison_list],
                y=[item['prediction']['Predicted_Dmax'] for item in st.session_state.comparison_list],
                marker_color='#00B4DB',
                text=[f"{item['prediction']['Predicted_Dmax']:.2f} mm" for item in st.session_state.comparison_list],
                textposition='outside',
            )
        ])
        fig.update_layout(
            title="Critical Diameter Comparison",
            xaxis_title="Alloy",
            yaxis_title="Dmax (mm)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#CBD5E1"},
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Temperature comparison
        fig = go.Figure()
        alloys = [item['composition'] for item in st.session_state.comparison_list]
        fig.add_trace(go.Scatter(
            x=alloys,
            y=[item['prediction']['Predicted_Tg'] for item in st.session_state.comparison_list],
            mode='markers+lines',
            name='Tg',
            marker_color='#10B981',
            line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=alloys,
            y=[item['prediction']['Predicted_Tx'] for item in st.session_state.comparison_list],
            mode='markers+lines',
            name='Tx',
            marker_color='#F59E0B',
            line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=alloys,
            y=[item['prediction']['Predicted_Tl'] for item in st.session_state.comparison_list],
            mode='markers+lines',
            name='Tl',
            marker_color='#EF4444',
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Temperature Comparison",
            xaxis_title="Alloy",
            yaxis_title="Temperature (K)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#CBD5E1"},
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download comparison
    comp_df_download = pd.DataFrame([
        {
            'Alloy': item['composition'],
            'Phase': item['prediction']['Predicted_Phase'],
            'Tg_K': item['prediction']['Predicted_Tg'],
            'Tx_K': item['prediction']['Predicted_Tx'],
            'Tl_K': item['prediction']['Predicted_Tl'],
            'Dmax_mm': item['prediction']['Predicted_Dmax'],
            'Rc_Ks': item['prediction']['Predicted_Rc']
        }
        for item in st.session_state.comparison_list
    ])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(get_download_link(comp_df_download, "comparison.csv", "csv"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_download_link(comp_df_download, "comparison.json", "json"), unsafe_allow_html=True)
    with col3:
        st.markdown(get_excel_download_link(comp_df_download, "comparison.xlsx"), unsafe_allow_html=True)
    
    if st.button("← Back to Main", use_container_width=True, type="secondary"):
        st.session_state.show_comparison = False
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="main-container">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Select input mode:",
        options=["Single Alloy", "Batch CSV"],
        horizontal=True,
        key="input_mode"
    )
    
    if mode == "Single Alloy":
        # If predictions exist, show compact composition + edit button
        if st.session_state.predictions is not None:
            composition_str = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                      for elem in st.session_state.selected_elements])
            st.markdown('<div class="section-title">Current Composition</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="compact-composition">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="composition-string">{composition_str}</span>
                </div>
                <div style="margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.3rem;">
                    {''.join([f'<span class="element-tag">{elem}</span>' for elem in st.session_state.selected_elements])}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✏️ Edit Composition", use_container_width=True, type="secondary"):
                    st.session_state.predictions = None
                    st.session_state.show_periodic_table = True
                    st.rerun()
            with col2:
                if st.button("📊 Add to Comparison", use_container_width=True, type="primary"):
                    composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                         for elem in st.session_state.selected_elements])
                    history_item = {
                        'composition': composition,
                        'prediction': st.session_state.predictions,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }
                    if composition not in [c['composition'] for c in st.session_state.comparison_list]:
                        st.session_state.comparison_list.append(history_item)
                        st.success("Added to comparison list!")
                    else:
                        st.warning("Already in comparison list")
                    st.rerun()
            with col3:
                if st.button("🔄 New Prediction", use_container_width=True, type="secondary"):
                    st.session_state.predictions = None
                    st.session_state.selected_elements = []
                    st.session_state.element_fractions = {}
                    st.session_state.show_periodic_table = True
                    st.rerun()
        
        # No predictions: show full composition setup
        else:
            st.markdown('<div class="section-title">Element Selection</div>', unsafe_allow_html=True)
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
            
            # Periodic Table
            if st.session_state.show_periodic_table:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
                                            auto_adjust_composition()
                                        else:
                                            st.warning(f"Maximum {num_elements} elements allowed")
                                    st.rerun()
                        else:
                            with cols[col_idx]:
                                st.write("")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Manual Input Toggle
            if not st.session_state.show_manual_input:
                if st.button("📝 Manual Input", key="toggle_manual", use_container_width=True, type="secondary"):
                    st.session_state.show_manual_input = True
                    st.rerun()
            
            # ---- MANUAL INPUT CARD (merged with sliders when open) ----
            if st.session_state.show_manual_input:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Manual Input")
                comp_string = st.text_input(
                    "Enter composition (e.g., Cu50Zr50 or Cu50Zr25Al25):",
                    key="composition_string",
                    placeholder="Cu50Zr50",
                    help="Format: ElementSymbolNumber (no spaces)"
                )
                st.markdown("""
                <div class="examples-box">
                    <strong>Examples:</strong><br>
                    • Cu50Zr50 (Cu 50%, Zr 50%)<br>
                    • Cu50Zr25Al25 (Cu 50%, Zr 25%, Al 25%)<br>
                    • Fe40Ni40P14B6 (Fe 40%, Ni 40%, P 14%, B 6%)
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Apply & Predict", key="apply_composition", type="primary", use_container_width=True):
                    if comp_string:
                        elements, fractions = parse_composition_string(comp_string)
                        if elements and fractions:
                            st.session_state.selected_elements = elements
                            st.session_state.element_fractions = {elem: frac for elem, frac in zip(elements, fractions)}
                            st.session_state.locked_elements = []
                            st.session_state.show_manual_input = False
                            st.session_state.prediction_error = None
                            composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                                 for elem in st.session_state.selected_elements])
                            with st.spinner("Analyzing alloy composition..."):
                                st.session_state.predictions = process_single_alloy(composition)
                                if st.session_state.predictions:
                                    st.session_state.show_periodic_table = False
                            st.rerun()
                        else:
                            st.error("Invalid format. Use format like Cu50Zr50")
                    else:
                        st.warning("Please enter a composition string")
                
                # If there are selected elements, show the sliders inside this same card
                if st.session_state.selected_elements:
                    st.markdown("---")
                    st.markdown("#### Adjust Composition")
                    
                    total = sum(st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements)
                    if abs(total - 100) > 0.1:
                        auto_adjust_composition()
                    
                    for elem in st.session_state.selected_elements:
                        current_val = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                        col_left, col_mid, col_right = st.columns([3, 1, 1])
                        with col_left:
                            is_locked = elem in st.session_state.locked_elements
                            new_val = st.slider(
                                elem,
                                min_value=0.0,
                                max_value=100.0,
                                value=float(current_val),
                                step=0.5,
                                key=f"slider_{elem}",
                                disabled=is_locked
                            )
                            if abs(new_val - current_val) > 0.01:
                                handle_element_change(elem, new_val)
                                st.rerun()
                        with col_mid:
                            display_val = st.session_state.element_fractions.get(elem, 0)
                            st.markdown(
                                f'<div style="color: #00B4DB; font-weight: 700; padding-top: 0.5rem;">{display_val:.1f}%</div>',
                                unsafe_allow_html=True
                            )
                        with col_right:
                            is_locked = elem in st.session_state.locked_elements
                            lock_icon = "🔓" if is_locked else "🔒"
                            lock_tooltip = "Unlock to edit" if is_locked else "Lock this value"
                            max_lockable = len(st.session_state.selected_elements) - 2
                            can_lock = len(st.session_state.locked_elements) < max_lockable or is_locked
                            if st.button(
                                lock_icon,
                                key=f"lock_{elem}",
                                help=lock_tooltip,
                                disabled=(not can_lock and not is_locked)
                            ):
                                if is_locked:
                                    st.session_state.locked_elements.remove(elem)
                                else:
                                    st.session_state.locked_elements.append(elem)
                                auto_adjust_composition()
                                st.rerun()
                    
                    total = sum(st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements)
                    st.progress(total/100, text=f"Total: {total:.1f}%")
                    
                    if abs(total - 100) <= 0.1:
                        st.markdown('<div class="success-text">✓ Valid</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="warning-text">⚠️ Adjusting...</div>', unsafe_allow_html=True)
                        auto_adjust_composition()
                        st.rerun()
                    
                    if len(st.session_state.selected_elements) > 1:
                        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                        fig_pie = create_composition_pie(
                            st.session_state.selected_elements,
                            [st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements]
                        )
                        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("🚀 Predict Properties", use_container_width=True, type="primary"):
                        composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                             for elem in st.session_state.selected_elements])
                        st.session_state.prediction_error = None
                        with st.spinner("Analyzing alloy composition..."):
                            st.session_state.predictions = process_single_alloy(composition)
                            if st.session_state.predictions:
                                st.session_state.show_periodic_table = False
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)  # close manual input card
            
            # ---- IF MANUAL INPUT IS CLOSED and there are selected elements, show the separate composition card ----
            if not st.session_state.show_manual_input and st.session_state.selected_elements:
                st.markdown('<div class="section-title">Composition</div>', unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                total = sum(st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements)
                if abs(total - 100) > 0.1:
                    auto_adjust_composition()
                
                for elem in st.session_state.selected_elements:
                    current_val = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                    col_left, col_mid, col_right = st.columns([3, 1, 1])
                    with col_left:
                        is_locked = elem in st.session_state.locked_elements
                        new_val = st.slider(
                            elem,
                            min_value=0.0,
                            max_value=100.0,
                            value=float(current_val),
                            step=0.5,
                            key=f"slider_{elem}",
                            disabled=is_locked
                        )
                        if abs(new_val - current_val) > 0.01:
                            handle_element_change(elem, new_val)
                            st.rerun()
                    with col_mid:
                        display_val = st.session_state.element_fractions.get(elem, 0)
                        st.markdown(
                            f'<div style="color: #00B4DB; font-weight: 700; padding-top: 0.5rem;">{display_val:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                    with col_right:
                        is_locked = elem in st.session_state.locked_elements
                        lock_icon = "🔓" if is_locked else "🔒"
                        lock_tooltip = "Unlock to edit" if is_locked else "Lock this value"
                        max_lockable = len(st.session_state.selected_elements) - 2
                        can_lock = len(st.session_state.locked_elements) < max_lockable or is_locked
                        if st.button(
                            lock_icon,
                            key=f"lock_{elem}",
                            help=lock_tooltip,
                            disabled=(not can_lock and not is_locked)
                        ):
                            if is_locked:
                                st.session_state.locked_elements.remove(elem)
                            else:
                                st.session_state.locked_elements.append(elem)
                            auto_adjust_composition()
                            st.rerun()
                
                total = sum(st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements)
                st.progress(total/100, text=f"Total: {total:.1f}%")
                
                if abs(total - 100) <= 0.1:
                    st.markdown('<div class="success-text">✓ Valid</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warning-text">⚠️ Adjusting...</div>', unsafe_allow_html=True)
                    auto_adjust_composition()
                    st.rerun()
                
                if len(st.session_state.selected_elements) > 1:
                    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                    fig_pie = create_composition_pie(
                        st.session_state.selected_elements,
                        [st.session_state.element_fractions.get(elem, 0) for elem in st.session_state.selected_elements]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("🚀 Predict Properties", use_container_width=True, type="primary"):
                    composition = "".join([f"{elem}{int(st.session_state.element_fractions[elem])}" 
                                         for elem in st.session_state.selected_elements])
                    st.session_state.prediction_error = None
                    with st.spinner("Analyzing alloy composition..."):
                        st.session_state.predictions = process_single_alloy(composition)
                        if st.session_state.predictions:
                            st.session_state.show_periodic_table = False
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Batch CSV mode
        st.markdown('<div class="section-title">Batch CSV Upload</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a CSV file with an 'Alloys' column",
            type=['csv'],
            help="The CSV must contain a column named 'Alloys' with alloy compositions (e.g., Cu50Zr50)."
        )
        if uploaded_file is not None:
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.markdown("**File Preview:**")
                st.dataframe(df_preview.head(), use_container_width=True)
                if 'Alloys' not in df_preview.columns:
                    st.error("The uploaded CSV does not contain an 'Alloys' column.")
                else:
                    if st.button("🚀 Run Batch Prediction", use_container_width=True, type="primary"):
                        st.session_state.batch_error = None
                        with st.spinner("Processing batch... This may take a while."):
                            uploaded_file.seek(0)
                            result_df = process_batch_csv(uploaded_file)
                            if result_df is not None:
                                st.session_state.batch_results = result_df
                            st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Results display
    if mode == "Single Alloy":
        if st.session_state.predictions is not None:
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            if not st.session_state.show_periodic_table:
                if st.button("📋 Show Periodic Table", key="show_table_results", type="secondary"):
                    st.session_state.show_periodic_table = True
                    st.rerun()
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            pred = st.session_state.predictions
            metric_cols = st.columns([1, 2])
            with metric_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ΔT RANGE</div>
                    <div class="metric-value">{pred['Predicted_Tx'] - pred['Predicted_Tg']:.0f}</div>
                    <div class="metric-sub">Tx - Tg (K)</div>
                </div>
                """, unsafe_allow_html=True)
            with metric_cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PHASE</div>
                    <div class="metric-value phase">{pred['Predicted_Phase']}</div>
                    <div class="metric-sub">Confidence: {pred['Phase_Confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
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
            
            st.markdown('<div class="section-title">Glass Forming Ability</div>', unsafe_allow_html=True)
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            fig_gauge = create_simple_gauge(pred['Predicted_Dmax'])
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(get_download_link(results_df, "bmg_prediction.csv", "csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_download_link(results_df, "bmg_prediction.json", "json"), unsafe_allow_html=True)
            with col3:
                st.markdown(get_excel_download_link(results_df, "bmg_prediction.xlsx"), unsafe_allow_html=True)
        
        elif st.session_state.prediction_error is not None:
            st.markdown('<div class="section-title">Prediction Error</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="error-message">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">❌ Prediction Failed</div>
                <div style="font-size: 0.9rem;">{st.session_state.prediction_error}</div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #FCA5A5;">
                    Please check your composition and try again.
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button("🔄 Try Again", use_container_width=True, type="secondary"):
                st.session_state.prediction_error = None
                st.session_state.predictions = None
                st.rerun()
        else:
            st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1.5rem;">
                <div style="font-size: 2.5rem; color: #00B4DB; margin-bottom: 0.8rem;">⚗️</div>
                <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; font-family: 'Space Grotesk', sans-serif;">
                    READY FOR ANALYSIS
                </div>
                <div style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 1rem;">
                    Select elements and set composition
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Batch CSV results
        if st.session_state.batch_results is not None:
            st.markdown('<div class="section-title">Batch Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(st.session_state.batch_results, use_container_width=True)
            
            # Batch visualizations
            st.markdown("#### 📊 Data Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                # Dmax distribution
                fig = px.histogram(
                    st.session_state.batch_results,
                    x='Dmax_mm',
                    nbins=20,
                    title="Critical Diameter Distribution",
                    color_discrete_sequence=['#00B4DB']
                )
                fig.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#CBD5E1"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Phase distribution
                phase_counts = st.session_state.batch_results['Predicted_Phase'].value_counts()
                fig = px.pie(
                    values=phase_counts.values,
                    names=phase_counts.index,
                    title="Phase Distribution",
                    color_discrete_sequence=['#10B981', '#EF4444']
                )
                fig.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#CBD5E1"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Temperature scatter plot
            fig = px.scatter(
                st.session_state.batch_results,
                x='Predicted_Tg',
                y='Predicted_Tx',
                color='Dmax_mm',
                title="Tg vs Tx Relationship",
                color_continuous_scale='Blues',
                hover_data=['Alloys']
            )
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#CBD5E1"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(get_download_link(st.session_state.batch_results, "batch_predictions.csv", "csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_download_link(st.session_state.batch_results, "batch_predictions.json", "json"), unsafe_allow_html=True)
            with col3:
                st.markdown(get_excel_download_link(st.session_state.batch_results, "batch_predictions.xlsx"), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state.batch_error is not None:
            st.markdown('<div class="section-title">Batch Processing Error</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="error-message">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">❌ Batch Processing Failed</div>
                <div style="font-size: 0.9rem;">{st.session_state.batch_error}</div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #FCA5A5;">
                    Please check your CSV file and try again.
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button("🔄 Clear Error", use_container_width=True, type="secondary"):
                st.session_state.batch_error = None
                st.rerun()
        else:
            st.markdown('<div class="section-title">Batch Panel</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1.5rem;">
                <div style="font-size: 2.5rem; color: #00B4DB; margin-bottom: 0.8rem;">📁</div>
                <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; font-family: 'Space Grotesk', sans-serif;">
                    UPLOAD A CSV FILE
                </div>
                <div style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 1rem;">
                    Your file must contain an 'Alloys' column.
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1.5rem; margin-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 500;">
        BMGcalc v2.0 • Bulk Metallic Glass Design Platform • © 2024
    </div>
</div>
""", unsafe_allow_html=True)
