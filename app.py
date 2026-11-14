# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Final Perfection)
#   With Detailed D_ens Disagreement Indicator & Tooltips
# -----------------------------------------------------------

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import py3Dmol
import traceback
import gc
import re
import tempfile
import os
import random
from io import BytesIO
import base64

# rdkit, mordred, autogluon, matminer may be heavy — import guarded
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    from rdkit.Chem.Draw import MolDraw2DSVG
    from rdkit.ML.Descriptors import MoleculeDescriptors
except Exception:
    rdkit = None

try:
    from mordred import Calculator, descriptors
except Exception:
    pass

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None

# matminer descriptors (guarded import)
try:
    from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
    from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
except Exception:
    ElementProperty = Meredig = Stoichiometry = IonProperty = StrToComposition = CompositionToOxidComposition = None

# Materials Project (mp-api) and pymatgen
try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifWriter
except Exception:
    Structure = Lattice = CifWriter = None

st.set_page_config(layout="wide", page_title="Ionic Conductivity Predictor")

# ----------------------------------
# MP 官方配色
# ----------------------------------
MP_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
    "K": "#8F40D4", "Ca": "#FFD478", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB",
    "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#4C4CFF", "As": "#BD80E3",
    "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00",
    "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
    "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#4DC4FF",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7",
    "Nd": "#C7FFC7", "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7",
    "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452",
    "Yb": "#00BF69", "Lu": "#00AB6B", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6",
    "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00",
    "At": "#754F45", "Rn": "#428296", "Fr": "#420066", "Ra": "#00C900", "Ac": "#70ABFA",
    "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF"
}

# CSS 样式
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
    .stMetric {
        font-size: 0.9em;
    }
    .stWrite {
        font-size: 0.9em;
    }
    h3 {
        font-size: 1.2em;
        margin-bottom: 0.5em;
    }
    .dataframe {
        font-size: 0.8em;
    }
    div[data-testid="column"] {
        padding: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
        <blockquote>
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
            2. Enter a valid chemical formula string below to get the predicted result. (Note: Restricted to inorganic crystalline solid electrolytes).
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Inputs --------------------
col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12")
    temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=1)
    submit_button = st.button("Submit and Predict")
with col2:
    MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
    mp_key_input = st.text_input("Materials Project API key :", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Always use placeholder structure (ignore MP)", value=False)
    st.markdown("If MP key left empty, app will use placeholder cell.")

# 恢复了您原本的 7 个核心特征，并把密度相关特征（如 MagpieData mean Density）补回
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence',
    'MagpieData mean Density'  # 已恢复密度相关特征展示
]

@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

@st.cache_data(show_spinner=False)
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        try:
            cto = CompositionToOxidComposition()
            df = cto.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)
            if 'composition_oxid' in df.columns:
                ion = IonProperty()
                df = ion.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)
        except Exception:
            pass

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {'Formula': formula}

def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered_features = {}
    filtered_features['Temp'] = float(temperature)
    
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            continue
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            filtered_features[feature_name] = 0.0
    return filtered_features

def load_structure_from_mp(formula, api_key):
    if MPRester is None:
        return None, "mp-api not installed"
    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula, fields=["structure"])
            if not results:
                return None, "No MP entry found"
            doc = results[0]
            try:
                struct = doc.structure.get_primitive_structure()
            except Exception:
                struct = doc.structure
            return struct, "Successfully loaded from MP" 
    except Exception as e:
        return None, f"MP error: {e}"

def generate_placeholder_structure(formula):
    elems = re.findall(r"[A-Z][a-z]?", formula or "")
    elems = list(dict.fromkeys(elems))
    if len(elems) == 0
