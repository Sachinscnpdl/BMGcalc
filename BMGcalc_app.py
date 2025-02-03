import streamlit as st
import pandas as pd
import periodictable

# Initialize session state for selected elements
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []

# Function to generate the periodic table using HTML and CSS
def generate_periodic_table():
    # Define the periodic table layout (classical layout)
    layout = [
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        ["Li", "Be", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
        ["Na", "Mg", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
        ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
        ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["", "", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "", ""],
        ["", "", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "", ""]
    ]

    # CSS for styling the periodic table
    st.markdown(
        """
        <style>
        .periodic-table {
            display: grid;
            grid-template-columns: repeat(18, 50px);
            gap: 5px;
            margin: 20px 0;
        }
        .element {
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f9f9f9;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        .element:hover {
            background-color: #ddd;
        }
        .element.selected {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # HTML for the periodic table
    table_html = '<div class="periodic-table">'
    for row in layout:
        for element in row:
            if element:
                is_selected = "selected" if element in st.session_state.selected_elements else ""
                table_html += f'<div class="element {is_selected}" onclick="selectElement(\'{element}\')">{element}</div>'
            else:
                table_html += '<div class="element" style="visibility: hidden;"></div>'
    table_html += '</div>'

    # JavaScript to handle element selection
    st.markdown(
        """
        <script>
        function selectElement(element) {
            fetch('/select_element?element=' + element)
                .then(response => response.json())
                .then(data => {
                    window.location.reload();
                });
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    # Display the periodic table
    st.markdown(table_html, unsafe_allow_html=True)

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

# Handle element selection via query parameters
if st.experimental_get_query_params().get("element"):
    element = st.experimental_get_query_params()["element"][0]
    if element in st.session_state.selected_elements:
        st.session_state.selected_elements.remove(element)
    else:
        st.session_state.selected_elements.append(element)
    st.experimental_set_query_params()
