import streamlit as st
import pandas as pd
import periodictable

# Initialize session state for selected elements
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []

# Function to display periodic table layout with selectable buttons
def generate_periodic_table():
    layout = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        ["Li", "Be", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
        ["Na", "Mg", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        ["Cs", "Ba", "", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
    ]
    
    with st.expander("Click to open Periodic Table", expanded=True):
        for row in layout:
            # Create columns with adjusted width
            cols = st.columns([2 if element else 0.5 for element in row])
            for col, element in zip(cols, row):
                if element:
                    if col.button(element, key=element):
                        if element in st.session_state.selected_elements:
                            st.session_state.selected_elements.remove(element)  # Deselect
                        else:
                            st.session_state.selected_elements.append(element)  # Select

# Streamlit UI
st.title("Multi-Component Alloy Property Predictor")

# Step 1: Number of Elements
num_elements = st.number_input("Number of elements", min_value=1, max_value=10, step=1, value=2)

# Step 2: Element Selection Method
selection_method = st.radio("Choose element selection method:", ("Periodic Table", "Dropdown"))

# Step 3: Periodic Table Popup
if selection_method == "Periodic Table":
    st.subheader("Select Elements")
    generate_periodic_table()
    selected_elements = st.session_state.selected_elements
else:
    st.subheader("Select Elements from Dropdown")
    all_elements = [el.symbol for el in periodictable.elements if el.symbol]
    selected_elements = st.multiselect("Choose elements:", all_elements, default=all_elements[:num_elements])
    st.session_state.selected_elements = selected_elements

# Step 4: Element Fraction Input
if len(selected_elements) == num_elements:
    st.subheader("Enter Element Fraction (%)")
    default_fraction = 100.0 / num_elements
    element_fractions = {}
    for elem in selected_elements:
        element_fractions[elem] = st.number_input(f"{elem} fraction (%)", min_value=0.0, max_value=100.0, step=0.1, value=default_fraction)

    # Ensure the total fraction is 100%
    total_fraction = sum(element_fractions.values())
    if total_fraction != 100:
        st.warning("Total fraction must be 100%")

    # Output Section: Predictions
    st.subheader("Prediction Panel")
    st.write("Predicted Phase: CMG")
    st.write("Glass Transition Temperature (T_g) [K]: 632")
    st.write("Crystallization Temperature (T_c) [K]: 650")
    st.write("Liquidus Temperature (T_l) [K]: 756")
    st.write("Critical Diameter of Alloy (d_c) [mm]: 15")
    st.write("Critical Cooling Rate (R_c) [K/s]: 586")
