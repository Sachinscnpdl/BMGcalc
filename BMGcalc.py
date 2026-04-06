# Authored by Sachin Poudel, Silesian University, Poland

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import base64
import pickle
import sys
import io

# Import the full pipeline
from bmg_pipeline import ModularBMGPipeline

# --- WEB-SAFE HELPER ---
def web_safe(val, precision=2):
    """Prevents JavaScript crashes by sanitizing data for HTML injection."""
    if pd.isna(val) or val is None:
        return "N/A"
    if isinstance(val, (int, float, np.float32, np.float64)):
        return f"{val:.{precision}f}"
    # Remove non-printable characters and newlines
    clean_str = "".join(char for char in str(val) if char.isprintable()).strip()
    return clean_str

# --- COMPATIBILITY FIX FOR SCIKIT-LEARN DECISION TREE ---
class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler to handle scikit-learn version compatibility issues"""
    def find_class(self, module, name):
        if module == 'sklearn.tree._tree' and name == 'Tree':
            # Return a compatible Tree class
            from sklearn.tree._tree import Tree
            return Tree
        return super().find_class(module, name)

def compatible_pickle_load(file_obj):
    """Load pickle with compatibility handling for decision trees"""
    try:
        # First try normal unpickling
        return pickle.load(file_obj)
    except (ValueError, TypeError) as e:
        if "node array from the pickle has an incompatible dtype" in str(e):
            # Reset file position
            file_obj.seek(0)
            # Try with our compatible unpickler
            return CompatibleUnpickler(file_obj).load()
        raise e

# Monkey patch pickle.load in the bmg_pipeline module
original_pickle_load = pickle.load
pickle.load = compatible_pickle_load

# Page configuration
st.set_page_config(
    page_title="BMGcalc - Metallic Glass Predictor",
    page_icon="🧪", # Using standard emoji for stability
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with font fallbacks
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        border: 1px solid rgba(0, 180, 219, 0.2);
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .unit { font-size: 0.9rem; color: #64748B; margin-left: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Main UI Logic
st.markdown('<div class="main-header">BMGcalc Predictor</div>', unsafe_allow_html=True)

# Initialize Pipeline
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = ModularBMGPipeline()
        st.success("Pipeline initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        st.info("This might be due to a version compatibility issue with the saved models.")

# Input Section
alloy_input = st.text_input("Enter Alloy Composition (e.g., Zr65Cu15Ni10Al10)", "Zr65Cu15Ni10Al10")

if st.button("Predict"):
    with st.spinner("Processing through 5-stage pipeline..."):
        try:
            # Run the actual calculation
            results = st.session_state.pipeline.run_pipeline(alloy_input)
            
            if results is not None and not results.empty:
                res = results.iloc[0]
                
                # --- START SANITIZED RESULTS DISPLAY ---
                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                # Phase Card
                phase = web_safe(res.get('Predicted_Phase', 'Unknown'))
                with col1:
                    st.markdown(f'''
                    <div class="glass-card">
                        <div class="metric-label">Predicted Phase</div>
                        <div class="metric-value" style="color: #00B4DB;">{phase}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Critical Diameter Card
                dmax = web_safe(res.get('Dmax_mm', 0))
                with col2:
                    st.markdown(f'''
                    <div class="glass-card">
                        <div class="metric-label">Critical Diameter (Dmax)</div>
                        <div class="metric-value">{dmax}<span class="unit">mm</span></div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Critical Cooling Rate Card
                rc = web_safe(res.get('Rc_mm/s', 0))
                with col3:
                    st.markdown(f'''
                    <div class="glass-card">
                        <div class="metric-label">Cooling Rate (Rc)</div>
                        <div class="metric-value">{rc}<span class="unit">K/s</span></div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Thermal Properties Row
                st.markdown("### Thermal Stability")
                t_cols = st.columns(3)
                tg = web_safe(res.get('Tg', 0))
                tx = web_safe(res.get('Tx', 0))
                tl = web_safe(res.get('Tl', 0))
                
                labels = ["Glass Transition (Tg)", "Crystallization (Tx)", "Liquid T (Tl)"]
                vals = [tg, tx, tl]
                
                for i, col in enumerate(t_cols):
                    with col:
                        st.markdown(f'''
                        <div class="glass-card">
                            <div class="metric-label">{labels[i]}</div>
                            <div class="metric-value">{vals[i]}<span class="unit">K</span></div>
                        </div>
                        ''', unsafe_allow_html=True)
                # --- END SANITIZED RESULTS ---

        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")
            st.info("This error is likely due to a version compatibility issue with the saved models. Consider updating scikit-learn or retraining the models with your current version.")
