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
#  Materials Project API KEY（直接写在代码，不使用 secrets）
# =====================================================
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# =====================================================
# Streamlit 页面样式
# =====================================================
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
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
            2. Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 输入区
# =====================================================
formula_input = st.text_input(
    "Enter Chemical Formula of the Material:",
    placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6",
)

temperature = st.number_input(
    "Select Temperature (K):",
    min_value=200,
    max_value=1000,
    value=298,
    step=10,
)

submit_button = st.button("Submit and Predict")


# =====================================================
# 缓存模型加载
# =====================================================
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# =====================================================
# 1. Public database structure retrieval
# =====================================================
def load_from_MP(formula: str):
    """
    Robust Materials Project loader:
    - iterate candidates returned by MP summary.search
    - use conventional cell
    - prefer structures that contain all elements in the formula
    - remove partial-occupancy sites (occupancy < 1) when safe
    - return a pymatgen Structure or None
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # get candidates (may return multiple entries)
            try:
                results = mpr.summary.search(formula=formula)
            except Exception:
                # fallback to older query if summary.search not available
                try:
                    q = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                    results = []
                    for item in q:
                        mid = item.get("material_id")
                        if mid:
                            # fetch structure object
                            s = mpr.get_structure_by_material_id(mid)
                            # wrap into a small helper object with attributes similar to summary
                            class _Dummy:
                                def __init__(self, structure):
                                    self.structure = structure
                            results.append(_Dummy(s))
                except Exception:
                    results = []

            if not results:
                return None

            # expected element symbols from input formula
            try:
                expected = {el.symbol for el in Composition(formula).elements}
            except Exception:
                expected = set()

            # iterate candidates and try to find a "clean" one
            for entry in results:
                try:
                    s = entry.structure
                except Exception:
                    continue

                # ensure conventional cell when possible
                try:
                    s_conv = s.get_conventional_structure()
                except Exception:
                    s_conv = s

                # quick check: does this structure contain all expected elements?
                try:
                    present = {el.symbol for el in s_conv.composition.elements}
                except Exception:
                    present = set()

                if expected and not expected.issubset(present):
                    # this candidate lacks some elements — skip
                    # (but keep trying other candidates)
                    continue

                # Build cleaned list of sites: keep sites with max occupancy ~1
                clean_sites = []
                for site in s_conv.sites:
                    # species_and_occu is a mapping Element -> occupancy (float)
                    try:
                        occu_vals = list(site.species_and_occu.values())
                        max_occu = max(occu_vals) if occu_vals else 1.0
                    except Exception:
                        # if we cannot read occupancies, assume fully occupied
                        max_occu = 1.0

                    # keep site if highest species occupancy ~ 1.0 (tolerance)
                    if max_occu >= 0.999:
                        clean_sites.append(site)
                    # else: skip partial-occupancy site

                # If we removed too many sites, fallback to original s_conv
                if len(clean_sites) < max(1, int(len(s_conv.sites) * 0.5)):
                    # too aggressive removal — use original candidate
                    final_struct = s_conv
                else:
                    # rebuild Structure from clean sites
                    try:
                        final_struct = Structure.from_sites(clean_sites)
                    except Exception:
                        final_struct = s_conv

                # final verification: contain expected elements?
                try:
                    final_present = {el.symbol for el in final_struct.composition.elements}
                except Exception:
                    final_present = set()

                if expected and not expected.issubset(final_present):
                    # cleaned structure lost some elements -> skip this candidate
                    continue

                # success: return the cleaned conventional structure
                return final_struct

            # if none passed checks, as a last resort return first candidate's conventional
            try:
                first = results[0].structure
                try:
                    return first.get_conventional_structure()
                except:
                    return first
            except:
                return None

    except Exception as e:
        st.error(f"Materials Project fetch failed: {e}")
        return None


def load_from_COD(formula):
    """Fallback database"""
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        lines = r.text.strip().split()
        if len(lines) == 0:
            return None

        cod_id = lines[0]
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        cif_bytes = requests.get(cif_url).content

        return Structure.from_str(cif_bytes.decode(), fmt="cif")
    except:
        return None


def load_crystal_structure_public(formula):
    st.info("Searching public databases for crystal structure...")

    s = load_from_MP(formula)
    if s:
        st.success("Structure found in Materials Project ✓")

        # -------- ★ 关键：标准化为 Materials Project 传统晶胞（官网也用它） --------
        try:
            s = s.get_conventional_structure()
            s = s.as_dict()  # 防止 py3Dmol 读取错误
            s = Structure.from_dict(s)
        except:
            pass

        return s

    s = load_from_COD(formula)
    if s:
        st.success("Structure found in COD ✓")
        return s

    st.error("No structure found in public databases.")
    return None


# =====================================================
# 2. Structure display (py3Dmol)
# =====================================================
def display_structure_py3Dmol(structure):
    try:
        # 强制 conventional cell
        try:
            structure = structure.get_conventional_structure()
        except:
            pass

        cif_str = structure.to(fmt="cif")


        view = py3Dmol.view(width=600, height=420)
        view.addModel(cif_str, "cif")

        view.setStyle({
            "sphere": {"scale": 0.28},
            "stick": {"radius": 0.15}
        })

        view.addUnitCell()
        view.zoomTo()

        st.components.v1.html(view._make_html(), height=450)

    except Exception as e:
        st.error(f"3D structure visualization failed: {e}")


# =====================================================
# 3. Feature extraction
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

        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {"Formula": formula}


# 指定的 7 个特征
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
# 4. Main Program Flow
# =====================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ---------- (1) Load & Show Crystal Structure ----------
    st.subheader("Crystal Structure")

    structure = load_crystal_structure_public(formula_input)

    if structure:
        display_structure_py3Dmol(structure)
    else:
        st.warning("Cannot find structure for this material.")

    # ---------- (2) Feature Extraction ----------
    with st.spinner("Extracting features..."):
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features.")

        selected = filter_selected_features(features, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    # ---------- (3) Prediction ----------
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
            "CatBoost",
            "ExtraTreesMSE",
            "LightGBM",
            "KNeighborsDist",
            "WeightedEnsemble_L2",
            "XGBoost",
        ]

        results = {}
        for model in models:
            try:
                results[model] = predictor.predict(input_df, model=model)
            except:
                results[model] = "Error"

        st.dataframe(pd.DataFrame(results).iloc[:1, :])

        del predictor
        gc.collect()



