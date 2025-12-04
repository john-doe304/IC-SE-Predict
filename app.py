# ============================================================
#  FIX NUMPY "product" MISSING (STREAMLIT CLOUD BUG)
# ============================================================
import numpy as _np
if not hasattr(_np, "product"):
    _np.product = _np.prod

# ============================================================
#  IMPORTS
# ============================================================
import streamlit as st
import gc
import requests
import numpy as np
import pandas as pd
import py3Dmol
from io import BytesIO
from autogluon.tabular import TabularPredictor

# RDKit
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# Mordred
from mordred import Calculator, descriptors

# Pymatgen
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.analysis.local_env import CrystalNN


# ============================================================
#  MATERIALS PROJECT API KEY（直接写，不使用 secrets）
# ============================================================
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# ============================================================
#  STREAMLIT UI STYLE
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
        <blockquote>
            1. Enter a chemical formula to load crystal structure + predict conductivity.<br>
            2. Crystal structure uses Materials Project official 1:1 rendering style.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  INPUT SECTION
# ============================================================
formula_input = st.text_input(
    "Enter Chemical Formula:",
    placeholder="Li7La3Zr2O12, Li10GeP2S12, Li3YCl6 ..."
)

temperature = st.number_input(
    "Temperature (K):", min_value=200, max_value=1000, value=298, step=10
)

submit_button = st.button("Submit and Predict")


# ============================================================
#  MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ============================================================
#  1. MATERIALS PROJECT — SAFE STRUCTURE LOADING
# ============================================================
def _to_conventional_safe(s):
    try:
        return s.get_conventional_structure()
    except:
        return s


def load_from_MP(formula: str):
    try:
        with MPRester(MP_API_KEY) as mpr:

            # -------- Case 1: direct mp-id --------
            if formula.lower().startswith("mp-"):
                return _to_conventional_safe(mpr.get_structure_by_material_id(formula))

            # -------- Case 2: find exact composition match --------
            comp = Composition(formula).reduced_formula
            results = mpr.summary.search(formula=comp)

            # filter exact composition
            def same_comp(e):
                return Composition(e.composition.reduced_formula) == Composition(comp)

            filtered = list(filter(same_comp, results))

            if filtered:
                # choose lowest energy (stable structure)
                filtered.sort(key=lambda x: x.energy_per_atom)
                s = filtered[0].structure
                return _to_conventional_safe(s)

            return None

    except Exception as e:
        st.error(f"Materials Project fetch failed: {e}")
        return None


# ============================================================
#  2. CRYSTAL RENDERING (1:1 MATERIALS PROJECT STYLE)
# ============================================================
def build_bonds(structure):
    """Use CrystalNN to build bond graph (MP uses similar method)."""
    try:
        cnn = CrystalNN()
        bonds = []
        for i, site in enumerate(structure):
            neighs = cnn.get_nn_info(structure, i)
            for nn in neighs:
                j = nn["site_index"]
                if j > i:  # avoid duplicates
                    bonds.append((i, j))
        return bonds
    except:
        return []


def display_structure_mp_style(structure):
    try:
        # ---------- 1. FORCE conventional cell ----------
        try:
            structure = structure.get_conventional_structure()
        except:
            pass

        # ---------- 2. Supercell 2×2×2 (same as MP viewer) ----------
        structure = structure.copy()
        structure.make_supercell([2, 2, 2])

        cif_str = structure.to(fmt="cif")

        # ---------- 3. Build bonds ----------
        bonds = build_bonds(structure)

        # ---------- 4. py3Dmol rendering ----------
        view = py3Dmol.view(width=650, height=520)
        view.addModel(cif_str, "cif")

        # Atoms (Jmol palette = MP official colors)
        view.setStyle({"sphere": {"scale": 0.30, "colorscheme": "Jmol"}})

        # Add bonds
        for i, j in bonds:
            view.addBond(
                {"atom1": i, "atom2": j, "radius": 0.12, "color": "gray"}
            )

        # Unit cell
        view.addUnitCell({"color": "white", "linewidth": 2})

        # Camera
        view.setBackgroundColor("white")
        view.setProjection("orthographic")
        view.zoomTo()

        st.components.v1.html(view._make_html(), height=540, scrolling=False)

    except Exception as e:
        st.error(f"3D rendering failed: {e}")


# ============================================================
#  3. FEATURE EXTRACTION
# ============================================================
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

        ep = ElementProperty.from_preset("magpie")
        df = ep.featurize_dataframe(df, "composition", ignore_errors=True)
        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(df, "composition", ignore_errors=True)

        df = CompositionToOxidComposition().featurize_dataframe(
            df, "composition", ignore_errors=True
        )
        df = IonProperty().featurize_dataframe(
            df, "composition_oxid", ignore_errors=True
        )

        # collect numeric features
        numeric = df.select_dtypes(include=[np.number]).columns
        for col in numeric:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.

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


# ============================================================
#  4. MAIN PROGRAM
# ============================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid formula.")
        st.stop()

    # ============================
    #  A. STRUCTURE
    # ============================
    st.subheader("Crystal Structure (Materials Project Style)")

    structure = load_from_MP(formula_input)

    if structure:
        display_structure_mp_style(structure)
    else:
        st.error("No structure found for this formula.")
        st.stop()

    # ============================
    #  B. FEATURE EXTRACTION
    # ============================
    with st.spinner("Extracting features..."):
        features = calculate_material_features(formula_input)
        selected = filter_selected_features(features, required_descriptors, temperature)

        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    # ============================
    #  C. PREDICTION
    # ============================
    st.subheader("Prediction Results")

    try:
        predictor = load_predictor()
    except:
        st.error("Model loading failed.")
        st.stop()

    input_dict = {"Formula": [formula_input], "Temp": [temperature]}
    for f in required_descriptors:
        if f != "Temp":
            input_dict[f] = [features.get(f, 0.0)]

    input_df = pd.DataFrame(input_dict)

    models = [
        "CatBoost",
        "ExtraTreesMSE",
        "LightGBM",
        "KNeighborsDist",
        "WeightedEnsemble_L2",
        "XGBoost",
    ]

    outputs = {}
    for m in models:
        try:
            outputs[m] = predictor.predict(input_df, model=m)
        except:
            outputs[m] = "Error"

    st.dataframe(pd.DataFrame(outputs).iloc[:1, :])

    del predictor
    gc.collect()
