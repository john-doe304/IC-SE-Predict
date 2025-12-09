# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Final)
#   With Materials Project Crystal Rendering + Fallback Cell
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import py3Dmol
import traceback
import gc
import re

# ML
from autogluon.tabular import TabularPredictor

# matminer descriptors
from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

# Materials Project
from mp_api.client import MPRester
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# ----------------------------------
# MP 官方配色
# ----------------------------------
MP_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "O": "#FF0D0D", "La": "#4A76D0",
    "Zr": "#A0C8FF", "P": "#FF8000", "S": "#FFFF30", "Ge": "#668F8F"
}

# ----------------------------------
# Streamlit UI CSS
# ----------------------------------
st.markdown("""
<style>
.stApp {max-width: 900px; margin: auto;}
.element-legend {
    border: 1px solid #ccc; padding: 10px; border-radius: 10px;
    width: 160px; background: #f8f8f8; font-size: 14px;
}
.color-box {
    width: 14px; height: 14px; display: inline-block;
    margin-right: 6px; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Title
# ----------------------------------
st.title("🔬 Solid Electrolyte Ionic Conductivity Predictor")

formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12")
temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298)

mp_api_key = st.text_input("Materials Project API key:", type="password", value="Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN")
disable_mp = st.checkbox("Always use placeholder structure (ignore MP)")

submit_button = st.button("Submit and Predict")

# -------------------------------
# Cached model loader
# -------------------------------
@st.cache_resource
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

# -------------------------------
# Material Feature Calculation
# -------------------------------
def calculate_material_features(formula):
    try:
        df = pd.DataFrame({"Formula": [formula]})
        df = StrToComposition().featurize_dataframe(df, "Formula", ignore_errors=True)
        if "composition" not in df or df["composition"].iloc[0] is None:
            return {}

        df = ElementProperty.from_preset("magpie").featurize_dataframe(df, "composition", ignore_errors=True)
        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(df, "composition", ignore_errors=True)
        
        df = CompositionToOxidComposition().featurize_dataframe(df, "composition", ignore_errors=True)
        df = IonProperty().featurize_dataframe(df, "composition_oxid", ignore_errors=True)

        features = df.select_dtypes(include=[np.number]).iloc[0].to_dict()
        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        traceback.print_exc()
        return {}

# -------------------------------
# Load MP structure
# -------------------------------
def load_mp_structure(formula, api_key):
    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula)
            if len(docs) == 0:
                return None, None
            doc = docs[0]
            struct = doc.structure.get_conventional_structure()
            return struct, doc.material_id
    except:
        return None, None

# -------------------------------
# Generate Placeholder Cell
# -------------------------------
def generate_placeholder_cell(formula):
    elems = re.findall(r"[A-Z][a-z]?", formula)
    elems = list(dict.fromkeys(elems))
    coords = np.random.rand(len(elems), 3) * 0.9

    lattice = Lattice.cubic(10)
    structure = Structure(lattice, elems, coords)
    return structure

# -------------------------------
# Py3Dmol Rendering (Correct)
# -------------------------------
def render_structure(structure):
    cif = CifWriter(structure).write_string()
    view = py3Dmol.view(width=600, height=450)
    view.addModel(cif, "cif")

    # Set atom styles
    for i, site in enumerate(structure.sites):
        elem = site.specie.symbol
        color = MP_COLORS.get(elem, "gray")

        view.setStyle(
            {"serial": i+1},
            {
                "sphere": {"radius": 0.5, "color": color},
                "stick": {"radius": 0.2, "color": color}
            }
        )

    view.addUnitCell()
    view.zoomTo()
    return view

# -------------------------------
# Element Legend
# -------------------------------
def draw_element_legend(structure):
    elems = sorted({str(s.specie.symbol) for s in structure.sites})

    html = "<div class='element-legend'><b>Element colors</b><br>"
    for e in elems:
        c = MP_COLORS.get(e, "#888888")
        html += f"<div><span class='color-box' style='background:{c};'></span>{e}</div>"
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

# -------------------------------
# Main Execution
# -------------------------------
if submit_button:

    if not formula_input:
        st.error("Please enter a formula.")
        st.stop()

    with st.spinner("Processing..."):

        # Try MP structure
        structure = None
        mp_id = None

        if mp_api_key and (not disable_mp):
            structure, mp_id = load_mp_structure(formula_input, mp_api_key)

        # fallback
        if structure is None:
            st.warning("No MP structure found — using placeholder structure.")
            structure = generate_placeholder_cell(formula_input)

        # Render
        st.subheader("Crystal Structure Preview (Unit Cell)")

        if mp_id:
            st.success(f"Loaded from Materials Project: {mp_id}")

        viewer = render_structure(structure)
        viewer.show()

        draw_element_legend(structure)

        # ML Features
        features = calculate_material_features(formula_input)
        features["Temp"] = temperature

        st.subheader("Extracted Features")
        st.dataframe(pd.DataFrame([features]))

        # Prediction
        predictor = load_predictor()

        # Keep only known features
        model_feats = predictor.feature_metadata.get_features()
        row = {f: features.get(f, 0.0) for f in model_feats}

        pred = predictor.predict(pd.DataFrame([row])).iloc[0]

        st.subheader("Predicted Ionic Conductivity")
        st.success(f"{pred:.4e} S/cm")

        del predictor
        gc.collect()

