# ============================================================
#  FIX NUMPY "product" (STREAMLIT CLOUD BUG)
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
from autogluon.tabular import TabularPredictor

from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition


MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# ============================================================
#  STREAMLIT UI STYLE
# ============================================================
st.markdown("""
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
#  LOAD MP STRUCTURE (SAFE, FAST)
# ============================================================
def load_from_MP(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:
            comp = Composition(formula).reduced_formula

            results = mpr.summary.search(formula=comp)

            if results:
                # 取能量最低的
                results.sort(key=lambda x: x.energy_per_atom)
                s = results[0].structure

                # conventional cell
                try:
                    s = s.get_conventional_structure()
                except:
                    pass

                return s

        return None

    except Exception as e:
        st.error(f"MP fetch error: {e}")
        return None


# ============================================================
#  DISPLAY STRUCTURE (FAST, NON-FREEZING)
# ============================================================
def display_structure_fast(structure):
    try:
        structure = structure.copy()

        # 不扩胞（避免卡死）
        cif = structure.to(fmt="cif")

        view = py3Dmol.view(width=650, height=520)
        view.addModel(cif, "cif")

        # 使用 py3Dmol 自动键（非常快）
        view.setStyle({
            "stick": {"radius": 0.16},
            "sphere": {"scale": 0.30, "colorscheme": "Jmol"}
        })

        view.addUnitCell({"color": "white", "linewidth": 2})

        view.setBackgroundColor("white")
        view.setProjection("orthographic")
        view.zoomTo()

        st.components.v1.html(view._make_html(), height=540, scrolling=False)

    except Exception as e:
        st.error(f"Render error: {e}")


# ============================================================
#  FEATURE ENGINEERING
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
            df, "composition", ignore_errors=True)
        df = IonProperty().featurize_dataframe(
            df, "composition_oxid", ignore_errors=True)

        for col in df.select_dtypes(include=[np.number]).columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature extraction error: {e}")
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
    out = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            out[f] = features.get(f, 0.0)
    return out


# ============================================================
#  MAIN APP LOGIC
# ============================================================
st.title("Predict Ionic Conductivity of Solid Electrolytes")

formula = st.text_input("Chemical Formula")
temperature = st.number_input("Temperature (K)", 200, 1000, 298)
submit = st.button("Submit & Predict")

if submit:

    # ---------------- CRYSTAL STRUCTURE ----------------
    st.subheader("Crystal Structure")

    s = load_from_MP(formula)

    if s:
        display_structure_fast(s)
    else:
        st.error("No structure found in Materials Project.")
        st.stop()

    # ---------------- FEATURES ----------------
    with st.spinner("Extracting features..."):
        feats = calculate_material_features(formula)
        feats_sel = filter_selected_features(feats, required_descriptors, temperature)
        st.write(feats_sel)

    # ---------------- PREDICT ----------------
    predictor = load_predictor()

    df_in = {"Formula": [formula], "Temp": [temperature]}
    for f in required_descriptors:
        if f != "Temp":
            df_in[f] = [feats.get(f, 0.0)]

    df_in = pd.DataFrame(df_in)

    models = [
        "CatBoost",
        "ExtraTreesMSE",
        "LightGBM",
        "KNeighborsDist",
        "WeightedEnsemble_L2",
        "XGBoost",
    ]

    st.subheader("Prediction Results")
    out = {}
    for m in models:
        try:
            out[m] = predictor.predict(df_in, model=m)
        except:
            out[m] = "Error"

    st.dataframe(pd.DataFrame(out).iloc[:1, :])

    gc.collect()
