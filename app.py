# ===== app.py - MP-style crystal viewer + feature extraction + prediction =====
# Top compatibility shim (prevent numpy.product import errors in some libs)
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

# --- Matminer / Mordred ---
from mordred import Calculator, descriptors

# --- Pymatgen ---
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

# Optional: convex hull for polyhedral meshes
try:
    from scipy.spatial import ConvexHull
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ==========================
# Materials Project API Key
# ==========================
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"  # <- 你已要求硬编码

# ==========================
# Page style
# ==========================
st.set_page_config(page_title="MP-style Crystal Viewer", layout="centered")
st.markdown("""
    <style>
    .stApp { border: 2px solid #808080; border-radius: 12px; margin: 30px auto; max-width: 1100px; padding: 18px; }
    </style>
""", unsafe_allow_html=True)

st.title("Predict Ionic Conductivity — MP-style Structure Viewer")
st.caption("Fetch structure from Materials Project (or COD), visualize in MP-like style, extract features and predict.")

# ==========================
# Inputs
# ==========================
formula_input = st.text_input("Chemical formula (or MP id e.g. mp-1234):", placeholder="Li7La3Zr2O12")
temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=10)
submit_button = st.button("Submit and Predict")

# ==========================
# Load predictor (cached)
# ==========================
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ==========================
# Utilities: MP element colors (subset; default white fallback)
# Full map includes many elements; extended as needed.
# Colors given in hex (no '#'), convert to 0xRRGGBB when used.
# ==========================
MP_ELEMENT_COLORS = {
    "H":"ffffff","He":"d9ffff","Li":"cc80ff","Be":"c2ff00","B":"ffb5b5",
    "C":"909090","N":"3050f8","O":"ff0d0d","F":"90e050","Ne":"b3e3f5",
    "Na":"ab5cf2","Mg":"8aff00","Al":"bfa6a6","Si":"f0c8a0","P":"ff8000",
    "S":"ffff30","Cl":"1ff01f","Ar":"80d1e3","K":"8f40d4","Ca":"3dff00",
    "Sc":"e6e6e6","Ti":"bfc2c7","V":"a6a6ab","Cr":"8a99c7","Mn":"9c7ac7",
    "Fe":"e06633","Co":"f090a0","Ni":"50d050","Cu":"c88033","Zn":"7d80b0",
    "Ga":"c28f8f","Ge":"668f8f","As":"bd80e3","Se":"ffa100","Br":"a62929",
    "Kr":"5cb8d1","Rb":"702eb0","Sr":"00ff00","Y":"94ffff","Zr":"94e0e0",
    "Nb":"73c2c9","Mo":"54b5b5","Tc":"3b9e9e","Ru":"248f8f","Rh":"0a7d8c",
    "Pd":"006985","Ag":"c0c0c0","Cd":"ffd98f","In":"a67573","Sn":"668080",
    "Sb":"9e63b5","Te":"d47a00","I":"940094","Xe":"429eb0","Cs":"57178f",
    "Ba":"00c900","La":"70d4ff","Ce":"ffffc7","Pr":"d9ffc7","Nd":"c7ffc7",
    # Add more if needed...
}

def hex_to_0x(hexstr):
    if hexstr is None: return "0xffffff"
    h = hexstr.lstrip("#")
    if len(h)==6:
        return "0x" + h
    return "0xffffff"

# ==========================
# Structure retrieval (MP then COD)
# ==========================
def load_from_MP(formula_or_id: str):
    """
    Load structure from Materials Project.
    Accepts formula or mp-id (mp-xxxxx).
    Returns pymatgen Structure or None.
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # If user supplied mp-id, use direct get
            if isinstance(formula_or_id, str) and formula_or_id.lower().startswith("mp-"):
                try:
                    s = mpr.get_structure_by_material_id(formula_or_id)
                    s = _to_conventional_safe(s)
                    return s
                except Exception:
                    pass

            # try summary.search (modern)
            try:
                results = mpr.summary.search(formula=formula_or_id)
                if results:
                    # Prefer an entry whose conventional cell contains all elements in query
                    for entry in results:
                        s = entry.structure
                        s_conv = _to_conventional_safe(s)
                        if _structure_matches_formula(s_conv, formula_or_id):
                            return s_conv
                    # fallback first result
                    s = results[0].structure
                    return _to_conventional_safe(s)
            except Exception:
                pass

            # fallback: query material_id then fetch structure
            try:
                q = mpr.query(criteria={"formula": formula_or_id}, properties=["material_id"])
                if q:
                    mid = q[0].get("material_id")
                    if mid:
                        s = mpr.get_structure_by_material_id(mid)
                        return _to_conventional_safe(s)
            except Exception:
                pass

            # fallback: get_entries/get_structures
            try:
                entries = mpr.get_entries(formula_or_id)
                if entries:
                    s = entries[0].structure
                    return _to_conventional_safe(s)
            except Exception:
                pass

            try:
                structs = mpr.get_structures(formula_or_id)
                if structs:
                    return _to_conventional_safe(structs[0])
            except Exception:
                pass

        return None
    except Exception as e:
        st.error(f"Materials Project fetch failed: {e}")
        return None

def load_from_COD(formula: str):
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split()
        if len(lines)==0:
            return None
        cod_id = lines[0]
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        cif_bytes = requests.get(cif_url, timeout=10).content
        return Structure.from_str(cif_bytes.decode(), fmt="cif")
    except Exception:
        return None

def load_crystal_structure_public(query: str):
    st.info("Searching public databases for crystal structure...")
    s = load_from_MP(query)
    if s:
        st.success("Structure found in Materials Project ✓")
        return s
    s = load_from_COD(query)
    if s:
        st.success("Structure found in COD ✓")
        return s
    st.error("No structure found in public databases.")
    return None

# ==========================
# Helpers for safe manipulations
# ==========================
def _to_conventional_safe(s):
    try:
        sc = s.get_conventional_structure()
        return sc
    except Exception:
        return s

def _structure_matches_formula(s, formula):
    """
    Check that structure contains all element symbols in formula.
    """
    try:
        expected = {el.symbol for el in Composition(formula).elements}
        present = {el.symbol for el in s.composition.elements}
        return expected.issubset(present)
    except Exception:
        return True

def make_supercell(structure, size=(2,2,2)):
    try:
        stc = structure.copy()
        stc.make_supercell(size)
        return stc
    except Exception:
        return structure

# ==========================
# MP-style renderer (integrated)
# ==========================
def display_structure_mp(structure):
    """
    Render structure in MP-like style:
    - wrap
    - orthogonalize
    - supercell
    - wrap AGAIN (super important!)
    - MP colors, sphere scale, sticks
    - optional polyhedra
    """
    try:
        # Step 1: initial wrap
        try:
            s = structure.get_wrapped_structure()
        except Exception:
            s = structure.copy()

        # Step 2: orthogonalize initial cell
        try:
            s = s.get_orthogonalized_structure()
        except Exception:
            pass

        # Step 3: supercell 2x2x2
        s = make_supercell(s, (2,2,2))

        # ⭐⭐⭐ VERY IMPORTANT ⭐⭐⭐
        # after supercell, wrap again to remove floating atoms!
        try:
            s = s.get_wrapped_structure()
        except:
            pass

        # and orthogonalize AGAIN to match MP's behavior
        try:
            s = s.get_orthogonalized_structure()
        except:
            pass

        # Step 4: Export CIF
        cif_str = s.to(fmt="cif")

        view = py3Dmol.view(width=800, height=620)
        view.addModel(cif_str, "cif")

        # Step 5: MP-style sphere & stick
        for site in s:
            elem = site.specie.symbol

            # covalent radius → scale
            try:
                cov = Element(elem).covalent_radius or 0.7
            except:
                cov = 0.7

            sphere_scale = max(0.18, cov * 0.45)
            color = hex_to_0x(MP_ELEMENT_COLORS.get(elem, None))

            view.setStyle(
                {"elem": elem},
                {
                    "sphere": {"scale": sphere_scale, "color": color},
                    "stick": {"radius": 0.10, "color": color}
                }
            )

        # Step 6: unit cell & camera
        view.addUnitCell({"color":"0x000000","linewidth":1.2})
        view.setBackgroundColor("white")
        view.setProjection("orthographic")
        view.zoomTo()

        # Step 7: (optional) polyhedra if scipy present
        if _HAVE_SCIPY:
            try:
                _add_polyhedra_to_view(view, s)
            except:
                pass

        st.components.v1.html(view._make_html(), height=680)

    except Exception as e:
        st.error(f"MP-style visualization failed: {e}")


# ==========================
# Feature extraction (unchanged from yours)
# ==========================
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


# 7 required descriptors (same as before)
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

# ==========================
# Main flow
# ==========================
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula or MP id.")
        st.stop()

    st.subheader("Crystal Structure (MP-style rendering)")
    structure = load_crystal_structure_public(formula_input)
    if structure:
        display_structure_mp(structure)
    else:
        st.warning("Cannot find structure for this material.")

    # Features + prediction
    with st.spinner("Extracting features and predicting..."):
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features.")
        selected = filter_selected_features(features, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    st.subheader("Prediction Results")
    try:
        predictor = load_predictor()
    except Exception:
        predictor = None
        st.error("Model loading failed.")
    if predictor:
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)
        models = ["CatBoost","ExtraTreesMSE","LightGBM","KNeighborsDist","WeightedEnsemble_L2","XGBoost"]
        results = {}
        for model in models:
            try:
                results[model] = predictor.predict(input_df, model=model)
            except Exception:
                results[model] = "Error"
        st.dataframe(pd.DataFrame(results).iloc[:1,:])
        del predictor
        gc.collect()

