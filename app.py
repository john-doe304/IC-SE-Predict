import importlib, sys
import numpy as _np
if not hasattr(_np, "product"):
    _np.product = _np.prod

import streamlit as st
import os
import gc
import re
import requests
import numpy as np
import pandas as pd
import py3Dmol
from io import BytesIO
from autogluon.tabular import TabularPredictor

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# --- Matminer ---
from mordred import Calculator, descriptors

# --- Pymatgen ---
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition


# =====================================================
# Materials Project API Key
# =====================================================
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# =====================================================
# Streamlit Style
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 45%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h2 style="text-align:center;">Predict Ionic Conductivity of Solid Electrolytes</h2>
    <blockquote>
        1. Enter a chemical formula.<br>
        2. Crystal structure will be loaded from Materials Project or COD.<br>
        3. The ML model predicts ionic conductivity.
    </blockquote>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Inputs
# =====================================================
formula_input = st.text_input(
    "Enter Chemical Formula:",
    placeholder="e.g., Li7La3Zr2O12",
)

temperature = st.number_input(
    "Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10
)

submit_button = st.button("Submit and Predict")


# =====================================================
# Load ML Model
# =====================================================
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# =====================================================
# Structure Fetching
# =====================================================
def load_from_MP(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:

            # summary API
            try:
                results = mpr.summary.search(formula=formula)
                if results:
                    s = results[0].structure
                    try: s = s.get_conventional_structure()
                    except: pass
                    return s
            except:
                pass

            # fallback
            try:
                q = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                if q:
                    mid = q[0]["material_id"]
                    s = mpr.get_structure_by_material_id(mid)
                    try: s = s.get_conventional_structure()
                    except: pass
                    return s
            except:
                pass

        return None
    except Exception as e:
        st.error(f"MP fetch failed: {e}")
        return None


def load_from_COD(formula):
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        lines = r.text.strip().split()
        if not lines:
            return None

        cod_id = lines[0]
        cif = requests.get(f"https://www.crystallography.net/cod/{cod_id}.cif").content
        return Structure.from_str(cif.decode(), fmt="cif")
    except:
        return None


def load_crystal_structure_public(formula):
    st.info("Searching public databases for structure...")

    s = load_from_MP(formula)
    if s:
        st.success("Found in Materials Project ✓")
        try:
            s = s.get_conventional_structure()
            s = Structure.from_dict(s.as_dict())
        except:
            pass
        return s

    s = load_from_COD(formula)
    if s:
        st.success("Found in COD ✓")
        return s

    st.error("No structure found in MP or COD.")
    return None


# =====================================================
#  MP official colors (partial map)
# =====================================================
MP_COLOR_MAP = {
    "H": "#FFFFFF", "Li": "#CC0000", "O": "#FF6600", "La": "#0000FF",
    "Zr": "#00AA00", "Cl": "#00FFFF", "S": "#FFFF00", "Y": "#9900FF",
}
def get_mp_color(elem):
    return MP_COLOR_MAP.get(elem, "#AAAAAA")


# =====================================================
# 3D Rendering - ONLY BASIC UNIT CELL (no supercell)
# =====================================================
def display_structure_py3Dmol(structure):
    try:
        cif_str = structure.to(fmt="cif")

        # 建立 HTML 内容，手动创建 py3Dmol viewer
        html = f"""
        <div id="viewer" style="width:800px;height:600px; position: relative;"></div>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol.js"></script>
        <script>
            let element = document.getElementById("viewer");
            let config = {{ backgroundColor: "white" }};
            let viewer = $3Dmol.createViewer(element, config);
            viewer.addModel(`{cif_str}`, "cif");
            viewer.setStyle({{}}, {{"stick":{{"radius":0.12}}, "sphere":{{"scale":0.3}}}});
            viewer.addUnitCell();
            viewer.zoomTo();
            viewer.render();
        </script>
        """

        st.components.v1.html(html, height=620)
    except Exception as e:
        st.error(f"3D structure visualization failed: {e}")



# =====================================================
# Feature extraction
# =====================================================
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import (
            StrToComposition, CompositionToOxidComposition
        )

        df = pd.DataFrame({"Formula": [formula]})
        df = StrToComposition().featurize_dataframe(df, "Formula", ignore_errors=True)

        if "composition" not in df.columns:
            return {"Formula": formula}

        features = {"Formula": formula}

        df = ElementProperty.from_preset("magpie").featurize_dataframe(
            df, "composition", ignore_errors=True
        )
        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(df, "composition", ignore_errors=True)

        df = CompositionToOxidComposition().featurize_dataframe(
            df, "composition", ignore_errors=True
        )
        df = IonProperty().featurize_dataframe(
            df, "composition_oxid", ignore_errors=True
        )

        for col in df.select_dtypes(include=[np.number]).columns:
            v = df[col].iloc[0]
            features[col] = float(v) if not pd.isna(v) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {"Formula": formula}


required_descriptors = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]


def filter_selected_features(features, selected, temperature):
    result = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            result[f] = features.get(f, 0.0)
    return result


# =====================================================
# Main app flow
# =====================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # 1. Structure
    st.subheader("Crystal Structure (Basic Unit Cell Only)")

    structure = load_crystal_structure_public(formula_input)

    if structure:
        display_structure_py3Dmol(structure)
    else:
        st.warning("Cannot find structure for this material.")

    # 2. Feature extraction
    with st.spinner("Extracting features..."):
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features.")

        selected = filter_selected_features(features, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    # 3. Prediction
    st.subheader("Prediction Results")

    try:
        predictor = load_predictor()
    except:
        predictor = None
        st.error("Model loading failed.")

    if predictor:
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]

        input_df = pd.DataFrame(input_data)

        models = [
            "CatBoost", "ExtraTreesMSE", "LightGBM",
            "KNeighborsDist", "WeightedEnsemble_L2", "XGBoost"
        ]

        results = {}
        for m in models:
            try:
                results[m] = predictor.predict(input_df, model=m)
            except:
                results[m] = "Error"

        st.dataframe(pd.DataFrame(results).iloc[:1, :])

        del predictor
        gc.collect()


