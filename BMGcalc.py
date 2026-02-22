# Authored by Sachin Poudel, Silesian University, Poland
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import base64

from bmg_pipeline import ModularBMGPipeline

st.set_page_config(
    page_title="BMGcalc - Metallic Glass Predictor",
    page_icon="BMG",                     # simple text, no emoji
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Helper to ensure ASCII-only strings ----
def safe_str(s):
    """Convert to string and remove any non-ASCII characters."""
    return str(s).encode('ascii', 'ignore').decode()

# ---- Custom CSS (all ASCII) ----
st.markdown("""
<style>
    .main-header { background: #0F172A; padding: 1.5rem; border-radius: 12px; text-align: center; color: white; font-size: 2rem; font-weight: 700; margin-bottom: 2rem; }
    .glass-card { background: rgba(30,41,59,0.8); border: 1px solid rgba(0,180,219,0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .section-title { color: #00B4DB; font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; }
    .element-tag { background: #00B4DB; color: white; padding: 0.3rem 0.6rem; border-radius: 6px; display: inline-block; margin: 0.2rem; }
    .metric-card { background: rgba(0,180,219,0.1); border: 1px solid #00B4DB; border-radius: 10px; padding: 1rem; text-align: center; }
    .metric-label { color: #94A3B8; font-size: 0.75rem; font-weight: 600; }
    .metric-value { color: #00B4DB; font-size: 2rem; font-weight: 800; }
    .property-row { display: flex; justify-content: space-between; padding: 0.75rem 0; border-bottom: 1px solid rgba(0,180,219,0.1); }
    .property-label { color: #A0AEC0; }
    .property-value { color: #00B4DB; font-weight: 700; }
    .compact-composition { background: rgba(30,41,59,0.5); border: 1px solid rgba(0,180,219,0.2); border-radius: 8px; padding: 1rem; }
    .composition-string { font-family: monospace; font-size: 1.2rem; color: #00B4DB; }
    .gauge-container { background: rgba(30,41,59,0.9); border: 1px solid #00B4DB; border-radius: 12px; padding: 1.5rem; }
    .stButton > button { background: #00B4DB; color: white; border: none; border-radius: 8px; font-weight: 600; }
    .stButton > button:hover { background: #0083B0; }
    .warning-text { color: #F87171 !important; font-weight: 600; }
    .success-text { color: #4ADE80 !important; font-weight: 600; }
    .examples-box { background: #1F2A3A; border: 1px solid #2D3A4A; border-radius: 6px; padding: 0.6rem; color: #E0E0E0; }
    .error-message { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 1rem; color: #F87171; }
</style>
""", unsafe_allow_html=True)

# ---- Session state ----
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'element_fractions' not in st.session_state:
    st.session_state.element_fractions = {}
if 'show_periodic_table' not in st.session_state:
    st.session_state.show_periodic_table = True
if 'locked_elements' not in st.session_state:
    st.session_state.locked_elements = []
if 'show_manual_input' not in st.session_state:
    st.session_state.show_manual_input = False
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_error' not in st.session_state:
    st.session_state.batch_error = None
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Single Alloy"

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
    st.session_state.selected_elements = []
    st.session_state.predictions = None
    st.session_state.element_fractions = {}
    st.session_state.show_periodic_table = True
    st.session_state.locked_elements = []
    st.session_state.show_manual_input = False
    st.session_state.prediction_error = None
    st.session_state.batch_results = None
    st.session_state.batch_error = None

def create_simple_gauge(dmax_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dmax_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "GLASS FORMING ABILITY", 'font': {'color': '#00B4DB'}},
        number={'suffix': " mm", 'font': {'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "#00B4DB"},
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239,68,68,0.7)'},
                {'range': [1, 3], 'color': 'rgba(245,158,11,0.7)'},
                {'range': [3, 5], 'color': 'rgba(34,197,94,0.7)'},
                {'range': [5, 10], 'color': 'rgba(59,130,246,0.7)'}
            ],
        }
    ))
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_composition_pie(elements, fractions):
    colors = ['#00B4DB', '#0083B0', '#006994', '#005073', '#003752', '#001F3F']
    fig = go.Figure(data=[go.Pie(
        labels=[safe_str(e) for e in elements],
        values=fractions,
        hole=.4,
        marker_colors=colors[:len(elements)],
        textinfo='label+percent',
        textfont=dict(color='white'),
        marker=dict(line=dict(color='white', width=1))
    )])
    fig.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def parse_composition_string(comp_str):
    pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, comp_str)
    if not matches:
        return None, None
    elements, fractions, total = [], [], 0
    for el, fr in matches:
        elements.append(el)
        fractions.append(float(fr))
        total += float(fr)
    if total > 0:
        fractions = [f/total*100 for f in fractions]
    return elements, fractions

def auto_adjust_composition():
    if not st.session_state.selected_elements:
        return
    locked = st.session_state.locked_elements
    unlocked = [e for e in st.session_state.selected_elements if e not in locked]
    locked_total = sum(st.session_state.element_fractions.get(e, 0) for e in locked)
    if locked_total > 100:
        scale = 100 / locked_total
        for e in locked:
            st.session_state.element_fractions[e] *= scale
        locked_total = 100
    remaining = 100 - locked_total
    if remaining < 0:
        remaining = 0
    if unlocked:
        equal = remaining / len(unlocked)
        for e in unlocked:
            st.session_state.element_fractions[e] = equal
    elif remaining > 0:
        equal = remaining / len(locked)
        for e in locked:
            st.session_state.element_fractions[e] += equal

def handle_element_change(changed, new_val):
    st.session_state.element_fractions[changed] = new_val
    locked = st.session_state.locked_elements
    all_elems = st.session_state.selected_elements
    if changed in locked:
        unlocked = [e for e in all_elems if e not in locked]
        if unlocked:
            locked_total = sum(st.session_state.element_fractions.get(e, 0) for e in locked)
            if locked_total > 100:
                scale = 100 / locked_total
                for e in locked:
                    st.session_state.element_fractions[e] *= scale
                locked_total = 100
            remaining = 100 - locked_total
            equal = remaining / len(unlocked)
            for e in unlocked:
                st.session_state.element_fractions[e] = equal
    else:
        unlocked = [e for e in all_elems if e not in locked]
        other = [e for e in unlocked if e != changed]
        if other:
            locked_total = sum(st.session_state.element_fractions.get(e, 0) for e in locked)
            current = st.session_state.element_fractions[changed]
            max_allowed = 100 - locked_total
            if current > max_allowed:
                st.session_state.element_fractions[changed] = max_allowed
                current = max_allowed
            remaining = 100 - locked_total - current
            if remaining < 0:
                remaining = 0
            equal = remaining / len(other)
            for e in other:
                st.session_state.element_fractions[e] = equal

def process_single_alloy(comp_str):
    try:
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(comp_str, output_csv=None)
        row = result_df.iloc[0]
        pred = {
            'Alloys': safe_str(row['Alloys']),
            'Predicted_Phase': safe_str(row['Predicted_Phase']),
            'Phase_Confidence': float(row.get('Phase_Confidence', 0.95)),
            'Predicted_Tg': float(row['Predicted_Tg']),
            'Predicted_Tx': float(row['Predicted_Tx']),
            'Predicted_Tl': float(row['Predicted_Tl']),
            'Predicted_Dmax': float(row['Predicted_Dmax_mm']),
            'Predicted_Rc': float(row['Predicted_Rc_Ks'])
        }
        return pred
    except Exception as e:
        st.session_state.prediction_error = safe_str(e)
        return None

def process_batch_csv(uploaded_file):
    try:
        df_input = pd.read_csv(uploaded_file)
        if 'Alloys' not in df_input.columns:
            raise ValueError("CSV must contain an 'Alloys' column")
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(df_input, output_csv=None)
        result_df.rename(columns={'Predicted_Dmax_mm': 'Dmax_mm', 'Predicted_Rc_Ks': 'Rc_Ks'}, inplace=True)
        # sanitize all string columns
        for col in result_df.select_dtypes(include='object'):
            result_df[col] = result_df[col].apply(safe_str)
        return result_df
    except Exception as e:
        st.session_state.batch_error = safe_str(e)
        return None

def get_download_link(df, filename="bmg_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode().replace('\n', '')
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background:#10B981;color:white;padding:0.6rem 1.2rem;border-radius:6px;text-decoration:none;display:block;text-align:center;">Download Results</a>'
    return href

# ---- Header ----
col1, col2 = st.columns([6,1])
with col1:
    st.markdown('<div class="main-header">BMGcalc - Metallic Glass Predictor</div>', unsafe_allow_html=True)
with col2:
    if st.button("Reset", use_container_width=True):
        reset_app()
        st.rerun()

colA, colB = st.columns([1,1], gap="large")

with colA:
    st.markdown('<div class="section-title">Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Select input mode:", ["Single Alloy", "Batch CSV"], horizontal=True, key="input_mode")

    if mode == "Single Alloy":
        if st.session_state.predictions is not None:
            # show current composition + edit button
            comp_str = "".join([f"{e}{int(st.session_state.element_fractions[e])}" for e in st.session_state.selected_elements])
            st.markdown('<div class="section-title">Current Composition</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compact-composition"><span class="composition-string">{safe_str(comp_str)}</span></div>', unsafe_allow_html=True)
            if st.button("Edit Composition", use_container_width=True):
                st.session_state.predictions = None
                st.session_state.show_periodic_table = True
                st.rerun()
        else:
            # element selection
            st.markdown('<div class="section-title">Element Selection</div>', unsafe_allow_html=True)
            num_elements = st.select_slider("Number of elements", [2,3,4,5,6], value=3)

            # toggle table button
            btn_label = "Hide Table" if st.session_state.show_periodic_table else "Show Table"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_periodic_table = not st.session_state.show_periodic_table
                st.rerun()

            if st.session_state.show_periodic_table:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                for row_idx, row in PERIODIC_TABLE.items():
                    cols = st.columns(len(row))
                    for col_idx, elem in enumerate(row):
                        if elem:
                            with cols[col_idx]:
                                is_sel = elem in st.session_state.selected_elements
                                if st.button(elem, key=f"btn_{elem}_{row_idx}", type="primary" if is_sel else "secondary", use_container_width=True):
                                    if is_sel:
                                        st.session_state.selected_elements.remove(elem)
                                        if elem in st.session_state.locked_elements:
                                            st.session_state.locked_elements.remove(elem)
                                    else:
                                        if len(st.session_state.selected_elements) < num_elements:
                                            st.session_state.selected_elements.append(elem)
                                            auto_adjust_composition()
                                        else:
                                            st.warning(f"Max {num_elements} elements")
                                    st.rerun()
                        else:
                            with cols[col_idx]:
                                st.markdown("&nbsp;")
                st.markdown('</div>', unsafe_allow_html=True)

            if not st.session_state.show_manual_input:
                if st.button("Manual Input", use_container_width=True):
                    st.session_state.show_manual_input = True
                    st.rerun()

            if st.session_state.show_manual_input:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Manual Input")
                comp_str = st.text_input("Enter composition (e.g., Cu50Zr50):", placeholder="Cu50Zr50")
                st.markdown('<div class="examples-box"><strong>Examples:</strong><br>Cu50Zr50<br>Cu50Zr25Al25</div>', unsafe_allow_html=True)

                if st.button("Apply & Predict", type="primary", use_container_width=True):
                    if comp_str:
                        elems, fracs = parse_composition_string(comp_str)
                        if elems and fracs:
                            st.session_state.selected_elements = elems
                            st.session_state.element_fractions = dict(zip(elems, fracs))
                            st.session_state.locked_elements = []
                            st.session_state.show_manual_input = False
                            st.session_state.prediction_error = None
                            comp_for_pipe = "".join([f"{e}{int(st.session_state.element_fractions[e])}" for e in elems]).encode('ascii','ignore').decode()
                            with st.spinner("Analyzing..."):
                                st.session_state.predictions = process_single_alloy(comp_for_pipe)
                                if st.session_state.predictions:
                                    st.session_state.show_periodic_table = False
                            st.rerun()
                        else:
                            st.error("Invalid format")
                    else:
                        st.warning("Enter composition")
                st.markdown('</div>', unsafe_allow_html=True)

            # if elements selected, show sliders
            if not st.session_state.show_manual_input and st.session_state.selected_elements:
                st.markdown('<div class="section-title">Composition</div>', unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)

                total = sum(st.session_state.element_fractions.get(e,0) for e in st.session_state.selected_elements)
                if abs(total-100)>0.1:
                    auto_adjust_composition()

                for elem in st.session_state.selected_elements:
                    cur = st.session_state.element_fractions.get(elem, 100/len(st.session_state.selected_elements))
                    cols = st.columns([3,1,1])
                    with cols[0]:
                        is_locked = elem in st.session_state.locked_elements
                        new_val = st.slider(elem, 0.0, 100.0, float(cur), step=0.5, disabled=is_locked, key=f"slider_{elem}")
                        if abs(new_val - cur) > 0.01:
                            handle_element_change(elem, new_val)
                            st.rerun()
                    with cols[1]:
                        st.markdown(f'<div style="color:#00B4DB;padding-top:0.5rem;">{st.session_state.element_fractions[elem]:.1f}%</div>', unsafe_allow_html=True)
                    with cols[2]:
                        lock_icon = "Unlock" if is_locked else "Lock"
                        max_lock = len(st.session_state.selected_elements)-2
                        can_lock = len(st.session_state.locked_elements) < max_lock or is_locked
                        if st.button(lock_icon, key=f"lock_{elem}", disabled=not can_lock and not is_locked):
                            if is_locked:
                                st.session_state.locked_elements.remove(elem)
                            else:
                                st.session_state.locked_elements.append(elem)
                            auto_adjust_composition()
                            st.rerun()

                total = sum(st.session_state.element_fractions.get(e,0) for e in st.session_state.selected_elements)
                st.progress(total/100, text=f"Total: {total:.1f}%")
                if abs(total-100)<=0.1:
                    st.markdown('<div class="success-text">Valid</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-text">Adjusting...</div>', unsafe_allow_html=True)
                    auto_adjust_composition()
                    st.rerun()

                if len(st.session_state.selected_elements)>1:
                    fig_pie = create_composition_pie(st.session_state.selected_elements,
                                                      [st.session_state.element_fractions.get(e,0) for e in st.session_state.selected_elements])
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar':False})

                if st.button("Predict Properties", type="primary", use_container_width=True):
                    comp_for_pipe = "".join([f"{e}{int(st.session_state.element_fractions[e])}" for e in st.session_state.selected_elements]).encode('ascii','ignore').decode()
                    st.session_state.prediction_error = None
                    with st.spinner("Analyzing..."):
                        st.session_state.predictions = process_single_alloy(comp_for_pipe)
                        if st.session_state.predictions:
                            st.session_state.show_periodic_table = False
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

    else:  # Batch CSV mode
        st.markdown('<div class="section-title">Batch CSV Upload</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV with 'Alloys' column", type=['csv'])
        if uploaded:
            try:
                preview = pd.read_csv(uploaded)
                st.dataframe(preview.head())
                if 'Alloys' not in preview.columns:
                    st.error("Missing 'Alloys' column")
                else:
                    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                        st.session_state.batch_error = None
                        with st.spinner("Processing..."):
                            uploaded.seek(0)
                            result = process_batch_csv(uploaded)
                            if result is not None:
                                st.session_state.batch_results = result
                            st.rerun()
            except Exception as e:
                st.error(safe_str(e))
        st.markdown('</div>', unsafe_allow_html=True)

with colB:
    if mode == "Single Alloy":
        if st.session_state.predictions is not None:
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            if not st.session_state.show_periodic_table:
                if st.button("Show Periodic Table"):
                    st.session_state.show_periodic_table = True
                    st.rerun()
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            p = st.session_state.predictions
            mcol1, mcol2 = st.columns([1,2])
            with mcol1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Delta T (K)</div><div class="metric-value">{p["Predicted_Tx"]-p["Predicted_Tg"]:.0f}</div></div>', unsafe_allow_html=True)
            with mcol2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Phase</div><div class="metric-value">{p["Predicted_Phase"]}</div><div>Conf: {p["Phase_Confidence"]:.1%}</div></div>', unsafe_allow_html=True)

            props = [
                ("Tg (K)", f"{p['Predicted_Tg']:.1f}"),
                ("Tx (K)", f"{p['Predicted_Tx']:.1f}"),
                ("Tl (K)", f"{p['Predicted_Tl']:.1f}"),
                ("Dmax (mm)", f"{p['Predicted_Dmax']:.4f}"),
                ("Rc (K/s)", f"{p['Predicted_Rc']:.3f}")
            ]
            for label, val in props:
                st.markdown(f'<div class="property-row"><span class="property-label">{label}</span><span class="property-value">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Glass Forming Ability</div>', unsafe_allow_html=True)
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            st.plotly_chart(create_simple_gauge(p['Predicted_Dmax']), use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

            comp_disp = "".join([f"{e}{int(st.session_state.element_fractions[e])}" for e in st.session_state.selected_elements]).encode('ascii','ignore').decode()
            df_out = pd.DataFrame({
                'Alloy': [comp_disp],
                'Phase': [p['Predicted_Phase']],
                'Confidence': [p['Phase_Confidence']],
                'Tg_K': [p['Predicted_Tg']],
                'Tx_K': [p['Predicted_Tx']],
                'Tl_K': [p['Predicted_Tl']],
                'DeltaT_K': [p['Predicted_Tx']-p['Predicted_Tg']],
                'Dmax_mm': [p['Predicted_Dmax']],
                'Rc_Ks': [p['Predicted_Rc']]
            })
            st.markdown(get_download_link(df_out), unsafe_allow_html=True)

        elif st.session_state.prediction_error:
            st.markdown('<div class="section-title">Error</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error-message">Prediction failed: {st.session_state.prediction_error}</div>', unsafe_allow_html=True)
            if st.button("Try Again"):
                st.session_state.prediction_error = None
                st.session_state.predictions = None
                st.rerun()
        else:
            st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card" style="text-align:center;padding:2rem;">Select elements and set composition</div>', unsafe_allow_html=True)

    else:  # Batch results
        if st.session_state.batch_results is not None:
            st.markdown('<div class="section-title">Batch Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(st.session_state.batch_results, use_container_width=True)
            st.markdown(get_download_link(st.session_state.batch_results, "batch_predictions.csv"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state.batch_error:
            st.markdown('<div class="section-title">Error</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error-message">Batch processing failed: {st.session_state.batch_error}</div>', unsafe_allow_html=True)
            if st.button("Clear Error"):
                st.session_state.batch_error = None
                st.rerun()
        else:
            st.markdown('<div class="section-title">Batch Panel</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card" style="text-align:center;padding:2rem;">Upload a CSV file</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:1.5rem;border-top:1px solid rgba(255,255,255,0.1);">BMGcalc v2.0</div>', unsafe_allow_html=True)
