# app.py
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
from autogluon.tabular import FeatureMetadata
import gc
import re
from tqdm import tqdm
import numpy as np

# crystal visualization imports
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import streamlit.components.v1 as components
import os
import traceback
import random

st.set_page_config(layout="wide")

# -------------------- User-provided default MP API key --------------------
# You've provided this key; it's placed here as default. If you don't want it
# embedded, set MP_API_KEY_DEFAULT = "" and use the input box in the app.
MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

# -------------------- Materials Project-ish color table --------------------
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

# -------------------- UI header & style --------------------
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predict Ionic Conductivity of Solid Electrolytes")
st.write("Enter a chemical formula to preview a unit cell (from Materials Project if available) and predict ionic conductivity.")

# -------------------- Inputs --------------------
col1, col2 = st.columns([2,1])

with col1:
    formula_input = st.text_input("Enter Chemical Formula (e.g., Li7La3Zr2O12):", placeholder="Li7La3Zr2O12")
    temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=1)
    submit_button = st.button("Submit and Predict", key="predict_button")

with col2:
    mp_key_input = st.text_input("Materials Project API key (optional):", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Always use placeholder structure (ignore MP)", value=False)
    st.markdown("If MP key left empty, app will try placeholder structure.")

# -------------------- Helper: get structure from MP --------------------
def get_structure_from_mp(formula, mp_api_key):
    """Return (structure, mpid or error string)"""
    try:
        if not mp_api_key:
            return None, "No API key provided"
        with MPRester(mp_api_key) as mpr:
            # Search by formula; summary.search is faster
            res = mpr.summary.search(formula=formula)
            if not res:
                return None, "No MP entry found for formula"
            mpid = res[0].material_id
            struct = mpr.get_structure_by_material_id(mpid)
            return struct, mpid
    except Exception as e:
        return None, f"MP error: {e}"

# -------------------- Helper: placeholder structure generator --------------------
def make_placeholder_structure(formula):
    """
    Create a simple placeholder Structure that displays nicely.
    Will create one site per unique element and place them in a cubic lattice.
    """
    try:
        elems = re.findall(r"[A-Z][a-z]?", formula)
        elems = list(dict.fromkeys(elems))
        if len(elems) == 0:
            raise ValueError("No elements parsed from formula.")
        # create a cubic lattice (10 Angstrom)
        lattice = Lattice.cubic(10.0)
        coords = []
        for i in range(len(elems)):
            # distribute points inside unit cube but avoid exact overlaps
            coords.append([0.1 + 0.8*((i+1)/(len(elems)+1)), 0.1 + 0.8*random.random(), 0.1 + 0.8*random.random()])
        struct = Structure(lattice, elems, coords)
        return struct
    except Exception as e:
        st.warning(f"Failed to create placeholder structure: {e}")
        return None

# -------------------- Helper: render structure to HTML (py3Dmol) --------------------
def render_structure_to_html(structure, width=640, height=480):
    """
    Convert pymatgen Structure to py3Dmol HTML with right-bottom legend.
    Returns HTML string or None on failure.
    """
    try:
        # Use cif string as the model source
        cif_str = structure.to(fmt="cif")
        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_str, "cif")
        # style atoms by element color (create per-atom styles)
        # py3Dmol can style by atom index; we will set sphere color per atom
        for i, site in enumerate(structure.sites):
            el = str(site.specie)
            color = MP_COLORS.get(el, "#9E9E9E")
            # style by atom index (serial starts at 1 in many py3Dmol viewers, but setStyle with serial works)
            view.setStyle({"serial": i+1}, {"sphere": {"radius": 0.45, "color": color}})
        # add stick style as well
        view.setStyle({"stick": {}})
        view.addUnitCell()
        view.zoomTo()
        model_html = view._make_html()

        # prepare legend html
        elements = sorted({str(s.specie) for s in structure.sites})
        legend_items = ""
        for el in elements:
            color = MP_COLORS.get(el, "#9E9E9E")
            legend_items += f"""
            <div style="display:flex;align-items:center;margin-bottom:4px;">
                <div style="width:14px;height:14px;background:{color};border:1px solid #444;border-radius:3px;margin-right:8px;"></div>
                <div style="font-size:12px;">{el}</div>
            </div>
            """

        legend_html = f"""
        <div style="
            position:absolute;
            bottom:12px;
            right:12px;
            background: rgba(255,255,255,0.92);
            border: 1px solid #ccc;
            padding: 8px;
            border-radius:6px;
            max-width:180px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            z-index:9999;
        ">
            <div style="font-weight:600;margin-bottom:6px;font-size:13px;">Element colors</div>
            {legend_items}
        </div>
        """

        final_html = f"""
        <div style="position:relative;width:{width}px;height:{height}px;">
            {model_html}
            {legend_html}
        </div>
        """
        return final_html
    except Exception as e:
        # return None and log
        st.warning(f"Render failed: {e}")
        return None

# -------------------- Crystal preview area --------------------
st.subheader("Crystal Structure Preview (unit cell)")

structure_to_show = None
mp_info_msg = ""

if formula_input:
    # If user explicitly checked placeholder, use placeholder
    if use_placeholder_checkbox or not mp_key_input:
        structure_to_show = make_placeholder_structure(formula_input)
        mp_info_msg = "Using placeholder unit cell (no MP lookup)."
    else:
        # try MP first
        struct, info = get_structure_from_mp(formula_input, mp_key_input)
        if struct is not None:
            structure_to_show = struct
            mp_info_msg = f"Loaded from Materials Project: {info}"
        else:
            # fallback to placeholder but inform user
            structure_to_show = make_placeholder_structure(formula_input)
            mp_info_msg = f"Materials Project lookup failed: {info}. Using placeholder."

    if structure_to_show is not None:
        st.success(mp_info_msg)
        html = render_structure_to_html(structure_to_show, width=700, height=520)
        if html:
            components.html(html, height=540, scrolling=False)
        else:
            st.error("Failed to render structure. See logs for details.")
    else:
        st.info("No structure available to show.")

else:
    st.info("Enter a formula above to preview the unit cell (MP lookup attempted if key provided).")

# -------------------- Your original ML / feature extraction / prediction code --------------------
# 指定的描述符列表 - 你选择的七个特征
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# 缓存模型加载器以避免重复加载
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    try:
        return TabularPredictor.load("./ag-20251024_075719")
    except Exception as e:
        st.warning(f"Failed to load AutoGluon predictor: {e}")
        return None

def mol_to_image(mol, size=(200, 200)):
    """将分子转换为背景颜色为 #f9f9f9 的SVG图像"""
    d2d = MolDraw2DSVG(size[0], size[1])
    draw_options = d2d.drawOptions()
    draw_options.background = '#f9f9f9'
    draw_options.padding = 0.0
    draw_options.additionalBondPadding = 0.0
    draw_options.annotationFontScale = 1.0
    draw_options.addAtomIndices = False
    draw_options.addStereoAnnotation = False
    draw_options.bondLineWidth = 1.5
    draw_options.includeMetadata = False
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect[^>]*/>', '', svg, flags=re.DOTALL)
    if 'viewBox' in svg:
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    return svg

# 计算材料特征
def calculate_material_features(formula):
    """计算材料的组成特征"""
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        # 元素属性特征
        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        # Meredig
        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        # 化学计量特征
        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        # 离子特征
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
        ion = IonProperty()
        df = ion.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)

        # 数值特征提取
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        traceback.print_exc()
        return {'Formula': formula}

# 过滤特征 - 只显示指定的七个特征
def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered_features = {}
    filtered_features['Temp'] = float(temperature)
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            continue
        filtered_features[feature_name] = features_dict.get(feature_name, 0.0)
    return filtered_features

# 自动匹配模型特征
def align_features_with_model(features_dict, predictor, temperature, formula):
    if predictor is None:
        return pd.DataFrame([features_dict])
    try:
        model_features = predictor.feature_metadata.get_features()
    except Exception:
        model_features = []
    aligned = {}
    lower_map = {k.lower(): k for k in features_dict.keys()}
    for feat in model_features:
        f_low = feat.lower()
        if feat in features_dict:
            aligned[feat] = features_dict[feat]
        elif f_low in lower_map:
            aligned[feat] = features_dict[lower_map[f_low]]
        elif f_low in ['temp', 'temperature', 'temperature_k']:
            aligned[feat] = temperature
        elif f_low in ['formula']:
            aligned[feat] = formula
        else:
            aligned[feat] = 0.0
    return pd.DataFrame([aligned])

# -------------------- Prediction flow --------------------
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")
                selected_features = filter_selected_features(features, required_descriptors, temperature)
                feature_df = pd.DataFrame([selected_features])
                st.subheader("Material Features")
                st.dataframe(feature_df)

                # create input dataframe for model
                input_data = {"Formula": [formula_input], "Temp": [temperature]}
                for name in required_descriptors:
                    if name == "Temp":
                        continue
                    input_data[name] = [features.get(name, 0.0)]
                input_df = pd.DataFrame(input_data)

                predictor = load_predictor()
                if predictor is None:
                    st.error("Predictor not available. Ensure model folder exists.")
                else:
                    essential_models = [
                        'CatBoost',
                        'ExtraTreesMSE',
                        'LightGBM',
                        'KNeighborsDist',
                        'WeightedEnsemble_L2',
                        'XGBoost'
                    ]
                    predictions_dict = {}
                    for model in essential_models:
                        try:
                            predictions_dict[model] = predictor.predict(input_df, model=model)
                        except Exception as me:
                            st.warning(f"Model {model} failed: {me}")
                            predictions_dict[model] = ["Error"]
                    st.write("Prediction Results (Essential Models):")
                    st.markdown("**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    results_df = pd.DataFrame(predictions_dict)
                    st.dataframe(results_df.iloc[:1, :])
                    # cleanup
                    del predictor
                    gc.collect()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                traceback.print_exc()

# End of app.py
