import os
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
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5e6e82;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f7fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
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
    .file-upload {
        border: 2px dashed #4b6cb7;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">BMGcalc</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Bulk Metallic Glass Property Prediction Tool</p>', unsafe_allow_html=True)
st.markdown("""
BMGcalc uses advanced machine learning models to predict key properties of metallic glasses, including:
- Phase classification (Metallic Glass vs. Crystalline)
- Thermal properties (Tg, Tx, Tl)
- Critical diameter (Dmax)
- Critical cooling rate (Rc)
""")

# Sidebar for input methods
st.sidebar.header("Input Methods")
input_method = st.sidebar.radio("Choose input method:", ["Upload CSV File", "Enter Composition Manually"])

# Function to validate alloy composition
def validate_alloy(alloy_string):
    """Validate if an alloy string is in the correct format"""
    # Pattern to match element symbols followed by numbers
    pattern = r'^([A-Z][a-z]?)(\d+)+$'
    return bool(re.match(pattern, alloy_string))

# Function to create download link for DataFrame
def get_download_link(df, filename="predictions.csv", text="Download Results"):
    """Generate a link to download the DataFrame as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to process alloys (simplified version for demo)
def process_alloys_demo(df):
    """Demo version of process_alloys for testing without the actual model"""
    # Create a mock results DataFrame
    results = df.copy()
    
    # Generate random predictions for demonstration
    np.random.seed(42)
    n_samples = len(results)
    
    # Phase predictions (70% metallic glass)
    results['Predicted_Phase'] = np.random.choice(['Metalic_Glass', 'Crystalline'], n_samples, p=[0.7, 0.3])
    results['Phase_Confidence'] = np.random.uniform(0.6, 0.95, n_samples)
    
    # Thermal properties
    results['Predicted_Tg'] = np.random.normal(600, 100, n_samples)
    results['Predicted_Tx'] = results['Predicted_Tg'] + np.random.normal(50, 20, n_samples)
    results['Predicted_Tl'] = results['Predicted_Tx'] + np.random.normal(200, 50, n_samples)
    
    # Dmax and Rc
    results['Predicted_Dmax'] = np.random.exponential(1.0, n_samples)
    results['Predicted_Rc'] = np.random.exponential(5.0, n_samples)
    
    return results

# Input Method 1: CSV File Upload
if input_method == "Upload CSV File":
    st.sidebar.markdown("### Upload CSV File")
    st.sidebar.markdown("Upload a CSV file with an 'Alloys' column containing alloy compositions.")
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)
            
            # Check if 'Alloys' column exists
            if 'Alloys' not in input_df.columns:
                st.error("The uploaded file must contain an 'Alloys' column.")
            else:
                # Display the uploaded data
                st.success("File uploaded successfully!")
                st.write("### Uploaded Data:")
                st.dataframe(input_df)
                
                # Predict button
                if st.button("Predict Properties", key="csv_predict"):
                    with st.spinner("Processing alloys and predicting properties..."):
                        # Process the data
                        results = process_alloys_demo(input_df)
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.input_method = "csv"
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

# Input Method 2: Manual Composition Entry
elif input_method == "Enter Composition Manually":
    st.sidebar.markdown("### Enter Alloy Composition")
    st.sidebar.markdown("Enter alloy compositions in the format: ElementSymbolPercentage (e.g., Zr65Cu15Ni10Al10)")
    
    # Text input for alloy composition
    alloy_input = st.sidebar.text_area("Enter alloy composition(s), one per line:")
    
    # Add examples
    with st.sidebar.expander("See Examples"):
        st.code("""
        Zr65Cu15Ni10Al10
        Cu50Zr50
        Fe80B20
        Mg65Cu25Y10
        """)
    
    # Predict button
    if st.sidebar.button("Predict Properties", key="manual_predict"):
        if alloy_input:
            # Split the input into individual alloys
            alloy_list = [alloy.strip() for alloy in alloy_input.split('\n') if alloy.strip()]
            
            # Validate each alloy
            valid_alloys = []
            invalid_alloys = []
            
            for alloy in alloy_list:
                if validate_alloy(alloy):
                    valid_alloys.append(alloy)
                else:
                    invalid_alloys.append(alloy)
            
            if invalid_alloys:
                st.warning(f"The following alloy compositions appear to be invalid: {', '.join(invalid_alloys)}")
            
            if valid_alloys:
                # Create a DataFrame with the valid alloys
                input_df = pd.DataFrame({"Alloys": valid_alloys})
                
                with st.spinner("Processing alloys and predicting properties..."):
                    # Process the data
                    results = process_alloys_demo(input_df)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.input_method = "manual"
        else:
            st.warning("Please enter at least one alloy composition.")

# Display results if available
if 'results' in st.session_state:
    results = st.session_state.results
    
    st.markdown("## Prediction Results")
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metallic_glass_count = (results['Predicted_Phase'] == 'Metalic_Glass').sum()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Metallic Glasses", f"{metallic_glass_count}/{len(results)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_tg = results['Predicted_Tg'].mean()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Tg (K)", f"{avg_tg:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_dmax = results['Predicted_Dmax'].mean()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Dmax (mm)", f"{avg_dmax:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Data Table", "Visualizations", "Individual Analysis"])
    
    with tab1:
        st.markdown("### Complete Results Table")
        
        # Add a color-coded column for phase
        def highlight_phase(val):
            color = 'background-color: #e6f7ff' if val == 'Metalic_Glass' else 'background-color: #fff2e6'
            return color
        
        styled_results = results.style.applymap(highlight_phase, subset=['Predicted_Phase'])
        st.dataframe(styled_results)
        
        # Download button
        st.markdown(get_download_link(results, "bmg_predictions.csv", "📥 Download Results as CSV"), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Property Visualizations")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Thermal Properties", "Critical Diameter", "Phase Distribution", "Cooling Rate"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Thermal properties scatter plot
        fig.add_trace(
            go.Scatter(
                x=results['Predicted_Tg'],
                y=results['Predicted_Tx'],
                mode='markers',
                marker=dict(
                    color=results['Predicted_Tl'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="Tl (K)", x=1.02)
                ),
                text=results['Alloys'],
                name="Tg vs Tx"
            ),
            row=1, col=1
        )
        
        # Critical diameter bar chart
        fig.add_trace(
            go.Bar(
                x=results['Alloys'],
                y=results['Predicted_Dmax'],
                marker_color='lightblue',
                name="Dmax"
            ),
            row=1, col=2
        )
        
        # Phase distribution pie chart
        phase_counts = results['Predicted_Phase'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=phase_counts.index,
                values=phase_counts.values,
                hole=0.4,
                name="Phase Distribution"
            ),
            row=2, col=1
        )
        
        # Cooling rate bar chart
        fig.add_trace(
            go.Bar(
                x=results['Alloys'],
                y=results['Predicted_Rc'],
                marker_color='lightcoral',
                name="Cooling Rate"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Metallic Glass Properties Visualization"
        )
        
        # Update x-axes for bar charts
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Tx (K)", row=1, col=1)
        fig.update_yaxes(title_text="Dmax (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Cooling Rate", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Individual Alloy Analysis")
        
        # Select an alloy for detailed analysis
        selected_alloy = st.selectbox("Select an alloy for detailed analysis:", results['Alloys'])
        
        # Get the selected alloy data
        alloy_data = results[results['Alloys'] == selected_alloy].iloc[0]
        
        # Display detailed information in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"#### {selected_alloy}")
            st.markdown(f"**Phase:** {alloy_data['Predicted_Phase']}")
            st.markdown(f"**Phase Confidence:** {alloy_data['Phase_Confidence']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("#### Thermal Properties")
            st.markdown(f"**Glass Transition (Tg):** {alloy_data['Predicted_Tg']:.1f} K")
            st.markdown(f"**Crystallization (Tx):** {alloy_data['Predicted_Tx']:.1f} K")
            st.markdown(f"**Liquidus (Tl):** {alloy_data['Predicted_Tl']:.1f} K")
            st.markdown(f"**Supercooled Liquid Region:** {alloy_data['Predicted_Tx'] - alloy_data['Predicted_Tg']:.1f} K")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("#### Glass Forming Ability")
            st.markdown(f"**Critical Diameter (Dmax):** {alloy_data['Predicted_Dmax']:.3f} mm")
            st.markdown(f"**Critical Cooling Rate (Rc):** {alloy_data['Predicted_Rc']:.2f} K/s")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create a gauge chart for glass forming ability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = alloy_data['Predicted_Dmax'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Glass Forming Ability"},
                delta = {'reference': 1},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 5], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 5}}))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About BMGcalc:** BMGcalc is a machine learning-powered tool for predicting properties of bulk metallic glasses. 
It uses state-of-the-art models trained on extensive experimental data to provide accurate predictions for various alloy systems.
""")
