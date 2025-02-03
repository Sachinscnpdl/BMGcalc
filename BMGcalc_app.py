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
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 24px;
            font-weight: bold;
            color: #FFA500;
            margin-top: 20px;
            margin-bottom: 10px;
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
            font-weight: bold;
            margin-top: 10px;
        }
        .prediction-box {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            color: #333;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .prediction-box h3 {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 15px;
        }
        .prediction-box p {
            margin: 10px 0;
        }
        .selected-elements {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-top: 20px;
        }
        .dropdown-container {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .dropdown-container .stMultiSelect {
            font-size: 18px;
            font-weight: bold;
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

# Streamlit UI
st.markdown('<div class="title">Welcome to Bulk Metallic Glass Design Calculator</div>', unsafe_allow_html=True)

# Step 1: Number of Elements
num_elements = st.number_input("Number of elements", min_value=1, max_value=10, step=1, value=2)

# Step 2: Element Selection Method
selection_method = st.radio("Choose element selection method:", ("Periodic Table", "Dropdown"))

# Step 3: Periodic Table or Dropdown Selection
st.markdown('<div class="subtitle">Select Elements</div>', unsafe_allow_html=True)
if selection_method == "Periodic Table":
    generate_periodic_table()
else:
    st.markdown('<div class="dropdown-container">', unsafe_allow_html=True)
    all_elements = [el.symbol for el in periodictable.elements if el.symbol]
    selected_elements = st.multiselect(
        "Choose elements:", 
        all_elements, 
        default=all_elements[:num_elements],
        key="dropdown_elements"
    )
    st.session_state.selected_elements = selected_elements
    st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Element Fraction Input



st.markdown(
    """
    <style>
    .fraction-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: left;
        color: #333;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .fraction-box h3 {
        font-size: 24px;
        font-weight: bold;
        color: #FFA500;
        margin-bottom: 15px;
    }
    .fraction-box .fraction-input {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .fraction-box .fraction-input strong {
        min-width: 150px; /* Fixed width for element names */
        font-weight: bold;
        color: #333;
    }
    .fraction-box .fraction-input input {
        padding: 8px 12px;
        border-radius: 5px;
        border: 1px solid #4CAF50;
        font-size: 16px;
        font-weight: bold;
        color: #00796b;
        width: 100px; /* Fixed width for input boxes */
        text-align: center; /* Center-align text inside input boxes */
    }
    </style>
    <div class="fraction-box">
        <h3>Enter Element Fraction (%)</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Dynamically generate fraction inputs for selected elements
for elem in st.session_state.selected_elements:
    default_fraction = 100.0 / len(st.session_state.selected_elements)
    fraction = st.number_input(
        f"{elem} fraction (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=default_fraction,
        key=f"fraction_{elem}"
    )
    st.markdown(
        f"""
        <style>
        .fraction-box .fraction-input strong {{
            min-width: 150px;
        }}
        </style>
        <div class="fraction-box">
            <div class="fraction-input">
                <strong>{elem} fraction (%):</strong>
                <input type="number" value="{fraction}" disabled>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Ensure the total fraction is 100%
total_fraction = sum(st.session_state[f"fraction_{elem}"] for elem in st.session_state.selected_elements)
if total_fraction != 100:
    st.markdown('<div class="warning">Total fraction must be 100%</div>', unsafe_allow_html=True)



    
    # Output Section: Predictions
    st.markdown('<div class="subtitle">Prediction Panel</div>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .prediction-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: left;
        color: #333;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .prediction-box h3 {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 15px;
    }
    .prediction-box .property {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .prediction-box .property strong {
        min-width: 300px; /* Fixed width for property names */
        font-weight: bold;
        color: #333;
    }
    .prediction-box .value-box {
        background-color: #e0f7fa;
        padding: 8px 12px;
        border-radius: 5px;
        border: 1px solid #4CAF50;
        font-weight: bold;
        color: #00796b;
        min-width: 100px; /* Fixed width for value boxes */
        text-align: center; /* Center-align text inside value boxes */
    }
    </style>
    <div class="prediction-box">
        <h3>Predicted Properties</h3>
        <div class="property">
            <strong>Predicted Phase:</strong>
            <div class="value-box">CMG</div>
        </div>
        <div class="property">
            <strong>Glass Transition Temperature (T<sub>g</sub>) [K]:</strong>
            <div class="value-box">632</div>
        </div>
        <div class="property">
            <strong>Crystallization Temperature (T<sub>c</sub>) [K]:</strong>
            <div class="value-box">650</div>
        </div>
        <div class="property">
            <strong>Liquidus Temperature (T<sub>l</sub>) [K]:</strong>
            <div class="value-box">756</div>
        </div>
        <div class="property">
            <strong>Critical Diameter of Alloy (d<sub>c</sub>) [mm]:</strong>
            <div class="value-box">15</div>
        </div>
        <div class="property">
            <strong>Critical Cooling Rate (R<sub>c</sub>) [K/s]:</strong>
            <div class="value-box">586</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
