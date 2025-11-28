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

# ========== 新增：晶体结构显示依赖 ==========
import py3Dmol
from pymatgen.ext.matproj import MPRester

# ========== Materials Project API KEY ==========
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"   # <-- 换成你自己的 API key

# 页面样式
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

# 输入区
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


@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ---------------------------------------------------
# 🔹 从 Materials Project 获取晶体结构（新增功能）
# ---------------------------------------------------
def get_structure_cif(formula):
    """从 Materials Project 查询晶体结构并返回 cif 字符串"""
    try:
        with MPRester(MP_API_KEY) as mpr:
            results = mpr.summary.search(formula=formula)
            if not results:
                return None
            structure = results[0].structure
            cif_str = structure.to(fmt="cif")
            return cif_str
    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None


# ---------------------------------------------------
# 🔹 使用 Py3Dmol 渲染晶体结构（新增功能）
# ---------------------------------------------------
def show_structure_3d(cif_string):
    """使用 py3Dmol 显示 3D 晶体结构"""
    view = py3Dmol.view(width=600, height=450)
    view.addModel(cif_string, "cif")
    view.setStyle({"stick": {}})
    view.addUnitCell()
    view.zoomTo()
    
    st.write(view._make_html(), unsafe_allow_html=True)


# ---------------------------------------------------
# 🔹 特征计算
# ---------------------------------------------------
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import (
            ElementProperty,
            Meredig,
            Stoichiometry,
            IonProperty,
        )
        from matminer.featurizers.conversions import (
            StrToComposition,
            CompositionToOxidComposition,
        )

        df = pd.DataFrame({"Formula": [formula]})
        df = StrToComposition().featurize_dataframe(df, "Formula", ignore_errors=True)

        if "composition" not in df.columns:
            return {"Formula": formula}

        features = {"Formula": formula}

        ep = ElementProperty.from_preset("magpie")
        df = ep.featurize_dataframe(df, "composition", ignore_errors=True)

        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(
            df, "composition", ignore_errors=True
        )
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


required_descriptors = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]


def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered = {"Temp": float(temperature)}
    for f in selected_descriptors:
        if f != "Temp":
            filtered[f] = features_dict.get(f, 0.0)
    return filtered


# ---------------------------------------------------
# 🔹 主逻辑
# ---------------------------------------------------
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # =======================
    # 1. 显示晶体结构（新增）
    # =======================
    st.subheader("Crystal Structure (from Materials Project)")
    cif_data = get_structure_cif(formula_input)

    if cif_data:
        show_structure_3d(cif_data)
    else:
        st.warning("No structure found for this material in Materials Project.")

    # =======================
    # 2. 特征工程 + 模型预测
    # =======================
    with st.spinner("Processing material and making predictions..."):

        features = calculate_material_features(formula_input)
        st.write(f"Total features extracted: {len(features)}")

        selected_features = filter_selected_features(
            features, required_descriptors, temperature
        )
        st.subheader("Material Features")
        st.dataframe(pd.DataFrame([selected_features]))

        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)

        # 模型预测
        predictor = load_predictor()
        models = [
            "CatBoost",
            "ExtraTreesMSE",
            "LightGBM",
            "KNeighborsDist",
            "WeightedEnsemble_L2",
            "XGBoost",
        ]

        results = {}
        for m in models:
            try:
                results[m] = predictor.predict(input_df, model=m)
            except:
                results[m] = "Error"

        st.subheader("Prediction Results")
        st.dataframe(pd.DataFrame(results).iloc[:1, :])

        del predictor
        gc.collect()

