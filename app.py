import streamlit as st
import os
import re
import gc
import requests
import numpy as np
import pandas as pd
import py3Dmol
from io import BytesIO
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# --- Matminer ---
from mordred import Calculator, descriptors

# --- Pymatgen ---
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester

# ========= Materials Project API KEY ==========
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

###############################################################
# UI SETTINGS
###############################################################
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h2 style="text-align:center;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
    <blockquote>
        1. Enter a chemical formula.<br>
        2. System retrieves crystal structure and predicts ionic conductivity.
    </blockquote>
    """,
    unsafe_allow_html=True,
)


###############################################################
# INPUT FIELDS
###############################################################

formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12")

temperature = st.number_input(
    "Temperature (K):", min_value=200, max_value=1000, value=298, step=10
)

submit_button = st.button("Submit and Predict")

required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]


###############################################################
# MODEL LOADER
###############################################################
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


###############################################################
# CRYSTAL STRUCTURE RENDER (py3Dmol)
###############################################################
def display_structure_py3Dmol(structure):
    try:
        cif_str = structure.to(fmt="cif")

        view = py3Dmol.view(width=600, height=400)
        view.addModel(cif_str, "cif")

        view.setStyle({"sphere": {"scale": 0.25}, "stick": {"radius": 0.15}})
        view.addUnitCell()
        view.zoomTo()

        st.components.v1.html(view._make_html(), height=450)
    except Exception as e:
        st.error(f"Structure visualization failed: {e}")


###############################################################
# MATERIALS PROJECT RETRIEVAL
###############################################################
def get_structure_cif(formula):
    """
    尝试从 Materials Project 获取结构（多种 fallback）
    返回 cif 字符串
    """
    try:
        with MPRester(MP_API_KEY) as mpr:

            # 第一优先：summary.search（新 API）
            try:
                if hasattr(mpr, "summary") and hasattr(mpr.summary, "search"):
                    res = mpr.summary.search(formula=formula)
                    if res:
                        s = res[0].structure
                        return s.to(fmt="cif")
            except Exception:
                pass

            # 第二优先：query（经典）
            try:
                query = mpr.query(
                    criteria={"formula": formula},
                    properties=["material_id"]
                )
                if query:
                    mid = query[0]["material_id"]
                    s = mpr.get_structure_by_material_id(mid)
                    return s.to(fmt="cif")
            except Exception:
                pass

            # 第三优先：entries
            try:
                entries = mpr.get_entries(formula)
                if entries:
                    s = entries[0].structure
                    return s.to(fmt="cif")
            except Exception:
                pass

            # 第四：get_structures（非常新）
            try:
                structs = mpr.get_structures(formula)
                if structs:
                    return structs[0].to(fmt="cif")
            except Exception:
                pass

            return None

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None

###############################################################
# COD DATABASE RETRIEVAL
###############################################################
def load_from_COD(formula):
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        lines = r.text.strip().split()
        if len(lines) == 0:
            return None

        cod_id = lines[0]
        cif_data = requests.get(f"https://www.crystallography.net/cod/{cod_id}.cif").content

        return Structure.from_str(cif_data.decode(), fmt="cif")

    except:
        return None


###############################################################
# STRUCTURE SEARCH ORDER
###############################################################
def load_crystal_structure_public(formula):
    st.info("Searching public databases...")

    s = load_from_MP(formula)
    if s:
        st.success("Structure found in Materials Project ✓")
        return s

    s = load_from_COD(formula)
    if s:
        st.success("Structure found in COD ✓")
        return s

    st.error("No structure found.")
    return None


###############################################################
# FEATURE CALCULATION
###############################################################
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({'Formula': [formula]})
        df = StrToComposition().featurize_dataframe(df, 'Formula', ignore_errors=True)

        if df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        df = ElementProperty.from_preset('magpie').featurize_dataframe(df, 'composition')
        df = Meredig().featurize_dataframe(df, 'composition')
        df = Stoichiometry().featurize_dataframe(df, 'composition')

        df = CompositionToOxidComposition().featurize_dataframe(df, 'composition')
        df = IonProperty().featurize_dataframe(df, 'composition_oxid')

        for col in df.select_dtypes(include=[np.number]).columns:
            v = df[col].iloc[0]
            features[col] = float(v) if not pd.isna(v) else 0.0

        return features

    except Exception as e:
        st.error(f"Feature calculation failed: {e}")
        return {'Formula': formula}


###############################################################
# SAFE FEATURE FILTER
###############################################################
def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0

def filter_selected_features(features, temp):
    filtered = {"Temp": safe_float(temp)}
    for name in required_descriptors:
        if name != "Temp":
            filtered[name] = safe_float(features.get(name, 0.0))
    return filtered


###############################################################
# MAIN
###############################################################
if submit_button:

    if not formula_input:
        st.error("Please enter formula.")
    else:

        st.subheader("Crystal Structure")
        structure = load_crystal_structure_public(formula_input)

        if structure:
            display_structure_py3Dmol(structure)

        st.subheader("Extracting Features...")
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features")

        selected = filter_selected_features(features, temperature)
        st.dataframe(pd.DataFrame([selected]))

        st.subheader("Prediction Results")

        input_df = pd.DataFrame({
            "Formula": [formula_input],
            **{k: [v] for k, v in selected.items()}
        })

        try:
            predictor = load_predictor()

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

            st.dataframe(pd.DataFrame(results).iloc[:1])

            del predictor
            gc.collect()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

