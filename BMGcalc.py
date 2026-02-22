# bmg_app_safe.py
# Safe Streamlit app — no unsafe_allow_html, no inline HTML injection.
# Replace your app with this file and deploy.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import base64
from typing import List, Tuple, Optional

# Keep your pipeline import (if your pipeline fails, the app will show the exception)
from bmg_pipeline import ModularBMGPipeline

st.set_page_config(page_title="BMGcalc - Metallic Glass Predictor (Safe)",
                   page_icon="⚗️", layout="wide")

# -----------------------
# Utilities and constants
# -----------------------

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

def parse_composition_string(comp_str: str) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    """Parse composition like Cu50Zr50 or Cu50Zr25Al25 -> (['Cu','Zr'], [50,50]) then normalize to percent."""
    pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, comp_str)
    if not matches:
        return None, None
    elements = []
    fractions = []
    total = 0.0
    for el, num in matches:
        try:
            val = float(num)
        except Exception:
            return None, None
        elements.append(el)
        fractions.append(val)
        total += val
    if total <= 0:
        return None, None
    fractions = [f / total * 100.0 for f in fractions]
    return elements, fractions

def create_simple_gauge(dmax_value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dmax_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "GLASS FORMING ABILITY"},
        number={'suffix': " mm"}
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_composition_pie(elements: List[str], fractions: List[float]) -> go.Figure:
    fig = go.Figure(data=[go.Pie(labels=elements, values=fractions, hole=0.4)])
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def format_comp_string(elements: List[str], fractions: List[float]) -> str:
    """Return canonical composition string like Cu50Zr50 (integers)"""
    parts = []
    for el, frac in zip(elements, fractions):
        parts.append(f"{el}{int(round(frac))}")
    return "".join(parts)

def safe_download_button_from_df(df: pd.DataFrame, filename: str="bmg_predictions.csv"):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download results (CSV)", data=csv_bytes, file_name=filename, mime="text/csv")

# -----------------------
# App state init
# -----------------------
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'element_fractions' not in st.session_state:
    st.session_state.element_fractions = {}
if 'locked_elements' not in st.session_state:
    st.session_state.locked_elements = []
if 'show_periodic_table' not in st.session_state:
    st.session_state.show_periodic_table = True
if 'show_manual_input' not in st.session_state:
    st.session_state.show_manual_input = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_error' not in st.session_state:
    st.session_state.batch_error = None
if 'num_elements' not in st.session_state:
    st.session_state.num_elements = 3
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Single Alloy"

# -----------------------
# Composition helpers
# -----------------------
def auto_adjust_composition():
    sel = st.session_state.selected_elements
    if not sel:
        return
    locked = st.session_state.locked_elements
    locked_total = sum(st.session_state.element_fractions.get(e, 0.0) for e in locked)
    unlocked = [e for e in sel if e not in locked]
    remaining = max(0.0, 100.0 - locked_total)
    if unlocked:
        share = remaining / len(unlocked)
        for e in unlocked:
            st.session_state.element_fractions[e] = share
    else:
        # only locked elements exist and their total may not be 100; rescale locked to 100
        if locked_total > 0:
            for e in locked:
                st.session_state.element_fractions[e] = st.session_state.element_fractions.get(e, 0.0) / locked_total * 100.0
        else:
            # fallback equal split
            for e in locked:
                st.session_state.element_fractions[e] = 100.0 / len(locked)

def handle_element_change(elem: str, new_val: float):
    st.session_state.element_fractions[elem] = float(new_val)
    locked = st.session_state.locked_elements
    all_elements = st.session_state.selected_elements
    if elem in locked:
        # adjusted a locked element: distribute remainder to unlocked
        unlocked = [e for e in all_elements if e not in locked]
        locked_total = sum(st.session_state.element_fractions.get(e, 0.0) for e in locked)
        remaining = max(0.0, 100.0 - locked_total)
        if unlocked:
            share = remaining / len(unlocked)
            for e in unlocked:
                st.session_state.element_fractions[e] = share
        else:
            # rescale locked if needed
            if locked_total > 0:
                for e in locked:
                    st.session_state.element_fractions[e] = st.session_state.element_fractions.get(e, 0.0) / locked_total * 100.0
    else:
        # adjusted an unlocked element: redistribute to other unlocked
        unlocked = [e for e in all_elements if e not in locked]
        other_unlocked = [e for e in unlocked if e != elem]
        locked_total = sum(st.session_state.element_fractions.get(e, 0.0) for e in locked)
        cur_val = st.session_state.element_fractions.get(elem, 0.0)
        max_allowed = max(0.0, 100.0 - locked_total)
        if cur_val > max_allowed:
            st.session_state.element_fractions[elem] = max_allowed
            cur_val = max_allowed
        remaining = max(0.0, 100.0 - locked_total - cur_val)
        if other_unlocked:
            share = remaining / len(other_unlocked)
            for e in other_unlocked:
                st.session_state.element_fractions[e] = share
        else:
            # only one unlocked element, ensure total 100
            if locked_total + cur_val != 100.0:
                # adjust locked proportionally
                adjust = 100.0 - cur_val
                if locked_total > 0:
                    for e in locked:
                        st.session_state.element_fractions[e] = st.session_state.element_fractions.get(e, 0.0) / locked_total * adjust
                else:
                    st.session_state.element_fractions[elem] = 100.0

# -----------------------
# Prediction helpers
# -----------------------
def process_single_alloy(composition_string: str):
    try:
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(composition_string, output_csv=None)
        # Expect a dataframe with columns similar to your original app
        row = result_df.iloc[0]
        pred = {
            'Alloys': row.get('Alloys', composition_string),
            'Predicted_Phase': row.get('Predicted_Phase', 'Unknown'),
            'Phase_Confidence': float(row.get('Phase_Confidence', 0.95)),
            'Predicted_Tg': float(row.get('Predicted_Tg', np.nan)),
            'Predicted_Tx': float(row.get('Predicted_Tx', np.nan)),
            'Predicted_Tl': float(row.get('Predicted_Tl', np.nan)),
            'Predicted_Dmax': float(row.get('Predicted_Dmax_mm', np.nan)),
            'Predicted_Rc': float(row.get('Predicted_Rc_Ks', np.nan))
        }
        return pred
    except Exception as e:
        st.session_state.prediction_error = str(e)
        return None

def process_batch_csv(uploaded_file):
    try:
        df_input = pd.read_csv(uploaded_file)
        if 'Alloys' not in df_input.columns:
            raise ValueError("CSV must contain an 'Alloys' column")
        pipeline = ModularBMGPipeline()
        result_df = pipeline.run_pipeline(df_input, output_csv=None)
        # rename if original pipeline uses different names
        result_df = result_df.rename(columns={'Predicted_Dmax_mm': 'Dmax_mm', 'Predicted_Rc_Ks': 'Rc_Ks'})
        return result_df
    except Exception as e:
        st.session_state.batch_error = str(e)
        return None

# -----------------------
# Layout
# -----------------------
st.title("BMGcalc — Metallic Glass Predictor (Safe mode)")

header_cols = st.columns([3,1])
with header_cols[1]:
    if st.button("Reset app state"):
        # reset everything relevant
        st.session_state.selected_elements = []
        st.session_state.element_fractions = {}
        st.session_state.locked_elements = []
        st.session_state.show_manual_input = False
        st.session_state.predictions = None
        st.session_state.prediction_error = None
        st.session_state.batch_results = None
        st.session_state.batch_error = None

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Mode", ["Single Alloy", "Batch CSV"], index=0)
st.session_state.input_mode = mode

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Element selection")
    num_elements = st.select_slider("Number of elements", options=[2,3,4,5,6], value=st.session_state.get('num_elements', 3))
    st.session_state.num_elements = num_elements

    if st.checkbox("Show periodic table", value=st.session_state.show_periodic_table):
        st.session_state.show_periodic_table = True
    else:
        st.session_state.show_periodic_table = False

    if st.session_state.show_periodic_table:
        st.markdown("Select elements from periodic table:")
        # render periodic table as grid of buttons
        for row_idx, row in PERIODIC_TABLE.items():
            cols = st.columns(len(row))
            for col_idx, element in enumerate(row):
                if element:
                    btn_key = f"pt_{element}_{row_idx}_{col_idx}"
                    if cols[col_idx].button(element, key=btn_key):
                        if element in st.session_state.selected_elements:
                            st.session_state.selected_elements.remove(element)
                            if element in st.session_state.locked_elements:
                                st.session_state.locked_elements.remove(element)
                        else:
                            if len(st.session_state.selected_elements) < num_elements:
                                st.session_state.selected_elements.append(element)
                                # set equal distribution immediately
                                auto_adjust_composition()
                            else:
                                st.warning(f"Maximum {num_elements} elements allowed")
                        # immediate short-circuit to refresh UI
                        st.experimental_rerun()

    st.write("---")
    # Manual input
    if st.button("Manual composition input"):
        st.session_state.show_manual_input = True

    if st.session_state.show_manual_input:
        comp_str = st.text_input("Enter composition (e.g., Cu50Zr50 or Cu50Zr25Al25)", value="")
        if st.button("Apply manual composition"):
            if not comp_str:
                st.warning("Enter something like Cu50Zr50")
            else:
                el, fr = parse_composition_string(comp_str)
                if not el:
                    st.error("Invalid composition format. Example: Cu50Zr50")
                else:
                    st.session_state.selected_elements = el
                    st.session_state.element_fractions = {e: f for e, f in zip(el, fr)}
                    st.session_state.locked_elements = []
                    st.session_state.show_manual_input = False
                    st.session_state.prediction_error = None
                    # trigger immediate prediction
                    comp = format_comp_string(el, fr)
                    with st.spinner("Predicting..."):
                        st.session_state.predictions = process_single_alloy(comp)
                    st.experimental_rerun()

    # Show composition controls when elements are selected
    if st.session_state.selected_elements:
        st.write("Selected elements:", ", ".join(st.session_state.selected_elements))
        # ensure fractions sum to 100
        total = sum(st.session_state.element_fractions.get(e, 0.0) for e in st.session_state.selected_elements)
        if abs(total - 100.0) > 0.5:
            auto_adjust_composition()

        for elem in st.session_state.selected_elements:
            is_locked = elem in st.session_state.locked_elements
            cols = st.columns([4,1,1])
            new_val = cols[0].slider(f"{elem} (%)", min_value=0.0, max_value=100.0,
                                     value=float(st.session_state.element_fractions.get(elem, 100.0/len(st.session_state.selected_elements))),
                                     step=0.5, key=f"slider_{elem}")
            if abs(new_val - st.session_state.element_fractions.get(elem, 0.0)) > 0.01:
                handle_element_change(elem, new_val)
                st.experimental_rerun()
            cols[1].write(f"{st.session_state.element_fractions.get(elem, 0.0):.1f}%")
            lock_label = "Unlock" if is_locked else "Lock"
            if cols[2].button(lock_label, key=f"lockbtn_{elem}"):
                if is_locked:
                    st.session_state.locked_elements.remove(elem)
                else:
                    # prevent locking too many
                    if len(st.session_state.locked_elements) < max(0, len(st.session_state.selected_elements)-2):
                        st.session_state.locked_elements.append(elem)
                    else:
                        # allow unlocking only or inform user
                        st.warning("Cannot lock that many elements")
                auto_adjust_composition()
                st.experimental_rerun()

        total = sum(st.session_state.element_fractions.get(e, 0.0) for e in st.session_state.selected_elements)
        st.progress(min(1.0, total/100.0))
        st.caption(f"Total: {total:.1f}%")

        if len(st.session_state.selected_elements) > 1:
            fig_pie = create_composition_pie(st.session_state.selected_elements,
                                             [st.session_state.element_fractions.get(e, 0.0) for e in st.session_state.selected_elements])
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        if st.button("Predict properties for selection"):
            composition = format_comp_string(st.session_state.selected_elements,
                                             [st.session_state.element_fractions.get(e, 0.0) for e in st.session_state.selected_elements])
            st.session_state.prediction_error = None
            with st.spinner("Running prediction..."):
                st.session_state.predictions = process_single_alloy(composition)
            st.experimental_rerun()

with col2:
    st.subheader("Results")
    if st.session_state.input_mode == "Single Alloy":
        if st.session_state.predictions is not None:
            p = st.session_state.predictions
            st.metric(label="Predicted Phase", value=str(p.get('Predicted_Phase', 'N/A')),
                      delta=f"Conf {p.get('Phase_Confidence', 0.0):.0%}")
            tg = p.get('Predicted_Tg', np.nan)
            tx = p.get('Predicted_Tx', np.nan)
            if not np.isnan(tg) and not np.isnan(tx):
                st.write(f"ΔT (Tx - Tg) = {tx - tg:.1f} K")
            st.write("Thermal properties:")
            st.write(f"- Tg: {p.get('Predicted_Tg')}")
            st.write(f"- Tx: {p.get('Predicted_Tx')}")
            st.write(f"- Tl: {p.get('Predicted_Tl')}")
            st.write(f"- Critical diameter (Dmax, mm): {p.get('Predicted_Dmax')}")
            st.write(f"- Critical cooling rate (Rc, K/s): {p.get('Predicted_Rc')}")

            # gauge
            try:
                fig_gauge = create_simple_gauge(float(p.get('Predicted_Dmax', 0.0)))
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            except Exception:
                pass

            # show single-row dataframe and provide safe download
            comp = format_comp_string(st.session_state.selected_elements,
                                     [st.session_state.element_fractions.get(e, 0.0) for e in st.session_state.selected_elements]) \
                   if st.session_state.selected_elements else p.get('Alloys', 'Unknown')
            results_df = pd.DataFrame({
                'Alloy': [comp],
                'Phase': [p.get('Predicted_Phase')],
                'Confidence': [p.get('Phase_Confidence')],
                'Tg_K': [p.get('Predicted_Tg')],
                'Tx_K': [p.get('Predicted_Tx')],
                'Tl_K': [p.get('Predicted_Tl')],
                'Delta_T_K': [p.get('Predicted_Tx') - p.get('Predicted_Tg') if p.get('Predicted_Tx') and p.get('Predicted_Tg') else np.nan],
                'Dmax_mm': [p.get('Predicted_Dmax')],
                'Rc_Ks': [p.get('Predicted_Rc')]
            })
            st.dataframe(results_df, use_container_width=True)
            safe_download_button_from_df(results_df, filename="bmg_prediction.csv")

        elif st.session_state.prediction_error is not None:
            st.error("Prediction failed. See message below.")
            st.write(st.session_state.prediction_error)
            if st.button("Clear error"):
                st.session_state.prediction_error = None
                st.session_state.predictions = None
        else:
            st.info("No prediction yet. Choose elements or use manual input, then click Predict.")

    else:
        st.subheader("Batch CSV")
        uploaded_file = st.file_uploader("Upload CSV with 'Alloys' column", type=['csv'])
        if uploaded_file is not None:
            try:
                preview = pd.read_csv(uploaded_file, nrows=5)
                st.write("Preview:")
                st.dataframe(preview)
            except Exception as e:
                st.error(f"Cannot read CSV preview: {e}")

            if st.button("Run batch prediction"):
                uploaded_file.seek(0)
                with st.spinner("Running batch predictions..."):
                    res = process_batch_csv(uploaded_file)
                    if res is not None:
                        st.session_state.batch_results = res
                    else:
                        st.error(st.session_state.batch_error or "Unknown batch error")
                st.experimental_rerun()

        if st.session_state.batch_results is not None:
            st.success("Batch results ready")
            st.dataframe(st.session_state.batch_results)
            safe_download_button_from_df(st.session_state.batch_results, filename="bmg_batch_results.csv")
        elif st.session_state.batch_error is not None:
            st.error(f"Batch error: {st.session_state.batch_error}")
            if st.button("Clear batch error"):
                st.session_state.batch_error = None

st.write("---")
st.caption("BMGcalc safe mode — uses only Streamlit native components. If this resolves your issue, we can reintroduce styling carefully.")
