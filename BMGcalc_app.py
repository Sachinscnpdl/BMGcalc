import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import periodictable

# Function to create periodic table buttons
def display_periodic_table(selected_elements):
    elements = [e.symbol for e in periodictable.elements if e.number]
    cols = 18
    
    table = ""  # HTML table structure
    for i, elem in enumerate(elements):
        checked = "selected" if elem in selected_elements else ""
        table += f"<button class='element-btn {checked}' onclick='selectElement(\"{elem}\")'>{elem}</button>"
        if (i + 1) % cols == 0:
            table += "<br>"
    
    return table

# Streamlit UI
st.title("Multi-Component Alloy Property Predictor")

# Step 1: Number of Elements
num_elements = st.number_input("Number of elements", min_value=1, max_value=10, step=1, value=2)

# Step 2: Periodic Table Selection
st.subheader("Select Elements")
selected_elements = st.multiselect("Click to select elements", [e.symbol for e in periodictable.elements if e.number])

# Step 3: Element Fraction Input
if len(selected_elements) == num_elements:
    st.subheader("Enter Element Fraction (%)")
    element_fractions = {}
    for elem in selected_elements:
        element_fractions[elem] = st.number_input(f"{elem} fraction (%)", min_value=0.0, max_value=100.0, step=0.1)

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
