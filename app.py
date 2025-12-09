import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
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

# -------------------- 新增导入：晶体结构 --------------------
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
# -----------------------------------------------------------


# 添加 CSS 样式
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

    /* ---------- 晶体结构颜色图例 ---------- */
    .legend-box {
        position: absolute;
        bottom: 15px;
        right: 15px;
        background: rgba(250,250,250,0.9);
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .legend-color {
        width: 14px;
        height: 14px;
        margin-right: 6px;
        border-radius: 4px;
        border: 1px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
        <blockquote>
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
             2.  Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- 晶体结构部分：新增 --------------------
st.subheader("Crystal Structure (Materials Project)")
mp_api_key = st.text_input("Enter Your Materials Project API Key:", type="password")

show_structure = False
structure_obj = None

def get_mp_structure(formula, api_key):
    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula, num_chunks=1)
            if len(results) == 0:
                return None
            mp_id = results[0].material_id
            struct = mpr.get_structure_by_material_id(mp_id)
            return struct
    except:
        return None


if mp_api_key and st.button("Load Crystal Structure from Materials Project"):
    struct = get_mp_structure(formula_input if 'formula_input' in locals() else "", mp_api_key)
    if struct:
        structure_obj = struct
        show_structure = True
        st.success("Crystal Structure Loaded Successfully!")
    else:
        st.error("Failed to load structure. Check formula or API key.")

# 显示晶体结构
if show_structure and structure_obj:
    st.subheader("Crystal Structure Viewer (Single Unit Cell)")

    # py3Dmol 显示单胞结构
    xyz = structure_obj.to(fmt="xyz")  # 仅单胞

    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz, "xyz")
    view.setStyle({"sphere": {"scale": 0.30}, "stick": {"radius": 0.15}})
    view.zoomTo()
    view.show()

    # ------ 图例部分 ------
    elements = sorted(set([str(s.specie) for s in structure_obj.sites]))

    legend_html = "<div class='legend-box'><b>Element Colors</b><br>"
    for e in elements:
        color_hex = Element(e).color  # MP 的标准颜色
        legend_html += f"""
            <div class='legend-item'>
                <div class='legend-color' style='background:{color_hex};'></div>{e}
            </div>
        """
    legend_html += "</div>"

    st.markdown(legend_html, unsafe_allow_html=True)

# -------------------- 晶体结构模块结束 --------------------




# -------------------------------------------------------------
# 下面开始保持你原有的全部代码（预测模型等）
# -------------------------------------------------------------

formula_input = st.text_input("Enter Chemical Formula of the Material:", placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")

temperature = st.number_input("Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10)

submit_button = st.button("Submit and Predict", key="predict_button")


required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


def mol_to_image(mol, size=(200, 200)):
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
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg)
    svg = re.sub(r'<rect[^>]*/>', '', svg)
    if 'viewBox' in svg:
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    return svg


def calculate_material_features(formula):
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

        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
        ion = IonProperty()
        df = ion.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)

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
        filtered_features[feature_name] = features_dict.get(feature_name, 0.0)
    return filtered_features


def align_features_with_model(features_dict, predictor, temperature, formula):
    try:
        model_features = predictor.feature_metadata.get_features()
    except:
        model_features = []

    aligned = {}
    lower_map = {k.lower(): k for k in features_dict.keys()}

    for feat in model_features:
        f_low = feat.lower()
        if feat in features_dict:
            aligned[feat] = features_dict[feat]
        elif f_low in lower_map:
            aligned[feat] = features_dict[lower_map[f_low]]
        elif f_low in ['temp', 'temperature']:
            aligned[feat] = temperature
        elif f_low == 'formula':
            aligned[feat] = formula
        else:
            aligned[feat] = 0.0

    return pd.DataFrame([aligned])


if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            features = calculate_material_features(formula_input)
            st.write(f"✅ Total features extracted: {len(features)}")

            selected_features = filter_selected_features(features, required_descriptors, temperature)
            feature_df = pd.DataFrame([selected_features])

            st.subheader("Material Features")
            st.dataframe(feature_df)

            input_data = {"Formula": [formula_input], "Temp": [temperature]}
            for name in required_descriptors:
                if name == "Temp":
                    continue
                input_data[name] = [features.get(name, 0.0)]

            input_df = pd.DataFrame(input_data)

            try:
                predictor = load_predictor()

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
                    except Exception as model_error:
                        st.warning(f"Model {model} prediction failed: {model_error}")
                        predictions_dict[model] = "Error"

                st.write("Prediction Results (Essential Models):")
                st.markdown("**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                results_df = pd.DataFrame(predictions_dict)
                st.dataframe(results_df.iloc[:1, :])

                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"Model loading failed: {e}")

