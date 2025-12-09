# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Final)
#   With Materials Project Crystal Rendering + Fallback Cell
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
    # rdkit optional; warn at runtime if missing when used
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
# MP 官方配色（可扩充）
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

# ----------------------------------
# UI header & style
# ----------------------------------
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 12px;
        margin: 20px auto;
        max-width: 1200px;
        background-color: #fbfbfb;
        padding: 16px;
        box-sizing: border-box;
    }
    .legend-box {
        position: absolute;
        bottom: 12px;
        right: 12px;
    }
    .element-legend {
        border: 1px solid #ccc; padding: 10px; border-radius: 10px;
        width: 180px; background: #f8f8f8; font-size: 13px;
    }
    .color-box {
        width: 14px; height: 14px; display: inline-block;
        margin-right: 6px; border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Predict Ionic Conductivity of Solid Electrolytes")

# -------------------- Inputs --------------------
col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12")
    temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=1)
    submit_button = st.button("Submit and Predict")
with col2:
    MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
    mp_key_input = st.text_input("Materials Project API key (optional):", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Always use placeholder structure (ignore MP)", value=False)
    st.markdown("If MP key left empty, app will use placeholder cell.")

# ------------------------------- Cached model loader -------------------------------
@st.cache_resource
def load_predictor():
    if TabularPredictor is None:
        st.warning("AutoGluon not installed or failed to import; prediction disabled.")
        return None
    try:
        return TabularPredictor.load("./ag-20251024_075719")
    except Exception as e:
        st.warning(f"Failed to load AutoGluon predictor: {e}")
        return None

# ------------------------------- Material feature calculation -------------------------------
def calculate_material_features(formula):
    if StrToComposition is None:
        st.warning("matminer not available — feature calculation skipped.")
        return {}
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

# ------------------------------- MP structure loader -------------------------------
def load_structure_from_mp(formula, api_key):
    if MPRester is None:
        return None, "mp-api not installed"
    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula)
            if not results:
                return None, "No MP entry found"
            doc = results[0]

            try:
                # *** 强制单胞 ***
                struct = doc.structure.get_primitive_structure()
            except Exception:
                struct = doc.structure

            return struct, doc.material_id
    except Exception as e:
        return None, f"MP error: {e}"


# ------------------------------- Placeholder cell generator -------------------------------
def generate_placeholder_structure(formula):
    elems = re.findall(r"[A-Z][a-z]?", formula or "")
    elems = list(dict.fromkeys(elems))
    if len(elems) == 0:
        elems = ["Li", "O"]  # fallback
    coords = []
    n = len(elems)
    for i in range(n):
        coords.append([0.1 + 0.8*((i+1)/(n+1)), 0.1 + 0.6*random.random(), 0.1 + 0.6*random.random()])
    if Lattice is None or Structure is None:
        st.warning("pymatgen not installed — cannot create placeholder Structure.")
        return None
    lattice = Lattice.cubic(10.0)
    struct = Structure(lattice, elems, coords)
    return struct

# ------------------------------- Structure -> CIF string (robust) -------------------------------
def structure_to_cif_string(structure):
    """Write structure to temp CIF and read string back — robust across pymatgen versions."""
    if CifWriter is None:
        st.warning("pymatgen.io.cif.CifWriter not available.")
        return None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            fname = tmp.name
        # write
        try:
            CifWriter(structure).write_file(fname)
        except Exception as e:
            # older/newer api differences — try alternate approach
            try:
                # If structure has to() with filename accepted
                structure.to(filename=fname)
            except Exception:
                raise e
        # read back
        with open(fname, "r", encoding="utf-8") as f:
            cif_str = f.read()
        return cif_str
    finally:
        # cleanup
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass

# ------------------------------- Render structure to HTML for Streamlit -------------------------------
def render_structure_to_html(structure, width=700, height=520):
    """Render pymatgen Structure to py3Dmol HTML using atom index (correct)."""
    if structure is None:
        return None

    cif_str = structure_to_cif_string(structure)
    if not cif_str:
        return None

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_str, "cif")

    # Correct style: use index (0-based), never serial
    for i, site in enumerate(structure.sites):
        el = str(site.specie)
        color = MP_COLORS.get(el, "#9E9E9E")
        view.setStyle({"index": i}, {
            "sphere": {"radius": 0.45, "color": color},
            "stick": {"radius": 0.18, "color": color}
        })

    # Add bonds explicitly to avoid under/over bonding
    view.addUnitCell()
    view.zoomTo()

    # Build HTML
    model_html = view._make_html()

    # Legend
    elements = sorted({str(s.specie) for s in structure.sites})
    legend_items = ""
    for el in elements:
        c = MP_COLORS.get(el, "#9E9E9E")
        legend_items += f"""
        <div style="display:flex;align-items:center;margin-bottom:4px;">
            <div style="width:14px;height:14px;background:{c};
                border:1px solid #444;border-radius:3px;margin-right:6px;"></div>
            {el}
        </div>
        """

    legend_html = f"""
    <div style="
        position:absolute; bottom:10px; right:10px;
        background:rgba(255,255,255,0.95);
        padding:8px; border-radius:6px;
        border:1px solid #ccc; font-size:12px;
    ">
        <b>Element colors</b><br>{legend_items}
    </div>
    """

    final = f"""
    <div style="position:relative;width:{width}px;height:{height}px;">
        {model_html}
        {legend_html}
    </div>
    """
    return final


# ------------------------------- Main Execution -------------------------------
if submit_button:
    if not formula_input:
        st.error("Please enter a formula.")
        st.stop()

    with st.spinner("Processing..."):
        # decide structure source
        structure = None
        mp_id = None
        mp_msg = ""

        if (mp_key_input and not use_placeholder_checkbox) and (MPRester is not None):
            try:
                struct, info = load_structure_from_mp(formula_input, mp_key_input)
                if struct:
                    structure = struct
                    mp_id = info
                    mp_msg = f"Loaded from Materials Project: {mp_id}"
                else:
                    mp_msg = f"MP lookup failed: {info}"
            except Exception as e:
                mp_msg = f"MP lookup exception: {e}"
        else:
            if use_placeholder_checkbox:
                mp_msg = "Placeholder structure selected by user."
            else:
                mp_msg = "MP key not provided or mp-api not installed."

        # fallback to placeholder if needed
        if structure is None:
            structure = generate_placeholder_structure(formula_input)
            if structure is None:
                st.error("Could not generate any structure (pymatgen missing).")
                st.stop()

        # render and show HTML in Streamlit using components
        st.subheader("Crystal Structure Preview (Unit Cell)")
        if mp_id:
            st.success(mp_msg)
        else:
            st.info(mp_msg)

        html = render_structure_to_html(structure, width=700, height=520)
        if html:
            components.html(html, height=560, scrolling=False)
        else:
            st.error("Failed to render structure.")

        # Draw legend separately too (redundant but helpful)
        # draw_element_legend(structure)
        # Features & prediction
        st.subheader("Extracted Features & Prediction")

        features = calculate_material_features(formula_input)
        features["Temp"] = float(temperature)

        st.write("Extracted features (partial):")
        try:
            st.dataframe(pd.DataFrame([features]))
        except Exception:
            st.write(features)

        predictor = load_predictor()
        if predictor is None:
            st.warning("Predictor not available. Ensure AutoGluon and model folder exist.")
        else:
            try:
                model_feats = predictor.feature_metadata.get_features()
            except Exception:
                model_feats = list(features.keys())  # fallback
            row = {f: features.get(f, 0.0) for f in model_feats}
            try:
                pred = predictor.predict(pd.DataFrame([row])).iloc[0]
                st.subheader("Predicted Ionic Conductivity")
                st.success(f"{pred:.4e} S/cm")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                traceback.print_exc()

        # cleanup
        try:
            del predictor
            gc.collect()
        except Exception:
            pass

# End of file

