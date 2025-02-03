import streamlit as st
import pandas as pd
import periodictable

# Initialize session state for selected elements
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []

# Function to generate the periodic table using HTML and CSS
def generate_periodic_table():
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

    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 24px;
            font-weight: bold;
            color: #FFA500;
        }
        .periodic-table {
            display: grid;
            grid-template-columns: repeat(18, 60px);
            gap: 5px;
            margin: 20px 0;
        }
        .element {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid #ccc;
            border-radius: 8px;
            background-color: #f0f0f0;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        .element:hover {
            background-color: #ddd;
        }
        .element.selected {
            background-color: #4CAF50;
            color: white;
        }
        .warning {
            color: red;
            font-size: 18px;
        }
        .prediction-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #333;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="periodic-table">', unsafe_allow_html=True)
    for row in layout:
        cols = st.columns(len(row))
        for col, element in zip(cols, row):
            if element:
                is_selected = element in st.session_state.selected_elements
                if col.button(
                    element,
                    key=f"btn_{element}",
                    help=f"Select {element}",
                    type="primary" if is_selected else "secondary"
                ):
                    if element in st.session_state.selected_elements:
                        st.session_state.selected_elements.remove(element)
                    else:
                        st.session_state.selected_elements.append(element)
            else:
                col.write("")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="title">Welcome to Bulk Metallic Glass Design Calculator</div>', unsafe_allow_html=True)

num_elements = st.number_input("Number of elements", min_value=1, max_value=10, step=1, value=2)

selection_method = st.radio("Choose element selection method:", ("Periodic Table", "Dropdown"))

if selection_method == "Periodic Table":
    st.markdown('<div class="subtitle">Select Elements</div>', unsafe_allow_html=True)
    generate_periodic_table()
    selected_elements = st.session_state.selected_elements
else:
    st.markdown('<div class="subtitle">Select Elements from Dropdown</div>', unsafe_allow_html=True)
    all_elements = [el.symbol for el in periodictable.elements if el.symbol]
    selected_elements = st.multiselect("Choose elements:", all_elements, default=all_elements[:num_elements])
    st.session_state.selected_elements = selected_elements

if len(selected_elements) == num_elements:
    st.markdown('<div class="subtitle">Enter Element Fraction (%)</div>', unsafe_allow_html=True)
    default_fraction = 100.0 / num_elements
    element_fractions = {}
    for elem in selected_elements:
        element_fractions[elem] = st.number_input(f"{elem} fraction (%)", min_value=0.0, max_value=100.0, step=0.1, value=default_fraction)

    total_fraction = sum(element_fractions.values())
    if total_fraction != 100:
        st.markdown('<div class="warning">Total fraction must be 100%</div>', unsafe_allow_html=True)

    st.markdown('<div class="subtitle">Prediction Panel</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="prediction-box">
        **Predicted Phase:** CMG  
        **Glass Transition Temperature (T_g) [K]:** 632  
        **Crystallization Temperature (T_c) [K]:** 650  
        **Liquidus Temperature (T_l) [K]:** 756  
        **Critical Diameter of Alloy (d_c) [mm]:** 15  
        **Critical Cooling Rate (R_c) [K/s]:** 586  
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("**Selected Elements:**", ", ".join(st.session_state.selected_elements))
