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

# -----------------------------
# 新增的晶体结构可视化依赖
# -----------------------------
from pymatgen.ext.matproj import MPRester
from crystal_toolkit.helpers.asymptote import StructureMoleculeComponent

# 你必须替换这里！
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

# 页面基本样式
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

# 页面标题
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

# FORMULA 输入区域
formula_input = st.text_input(
    "Enter Chemical Formula of the Material:",
    placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6",
)

# 温度输入
temperature = st.number_input(
    "Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10
)

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 选定的七个特征
required_descriptors = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]

# 缓存模型
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

# ---------------------------
# 材料结构查询（新增功能）
# ---------------------------
def get_structure_from_mp(formula):
    """从 Materials Project 查询晶体结构"""
    try:
        with MPRester(MP_API_KEY) as mpr:
            results = mpr.summary.search(formula=formula)
            if not results:
                return None
            structure = results[0].structure
            return structure
    except Exception as e:
        st.error(f"Error retrieving structure: {e}")
        return None


# ---------------------------
# 材料特征提取（你的原代码）
# ---------------------------
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
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, "Formula", ignore_errors=True)

        if "composition" not in df.columns or df["composition"].iloc[0] is None:
            return {"Formula": formula}

        features = {"Formula": formula}

        ep = ElementProperty.from_preset("magpie")
        df = ep.featurize_dataframe(df, "composition", ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, "composition", ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, "composition", ignore_errors=True)

        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, "composition", ignore_errors=True)

        ion = IonProperty()
        df = ion.featurize_dataframe(df, "composition_oxid", ignore_errors=True)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {"Formula": formula}


def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered = {"Temp": float(temperature)}
    for f in selected_descriptors:
        if f == "Temp":
            continue
        filtered[f] = features_dict.get(f, 0.0)
    return filtered


# ---------------------------
# 提交后执行
# ---------------------------
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:

        st.subheader("Crystal Structure (from Materials Project)")

        # ----------------------------
        # 新增：查询晶体结构并显示 3D
        # ----------------------------
        structure = get_structure_from_mp(formula_input)

        if structure:
            component = StructureMoleculeComponent(structure)
            st.components.v1.html(component.to_html(), height=600)
        else:
            st.warning("No crystal structure found for this formula in Materials Project.")

        # ----------------------------
        # 原本的特征工程和预测流程
        # ----------------------------
        with st.spinner("Processing material and making predictions..."):

            features = calculate_material_features(formula_input)
            st.write(f"✅ Total features extracted: {len(features)}")

            selected_features = filter_selected_features(
                features, required_descriptors, temperature
            )
            st.subheader("Material Features")
            st.dataframe(pd.DataFrame([selected_features]))

            # 构建预测输入
            input_data = {"Formula": [formula_input], "Temp": [temperature]}
            for f in required_descriptors:
                if f != "Temp":
                    input_data[f] = [features.get(f, 0.0)]

            input_df = pd.DataFrame(input_data)

            # 模型预测
            try:
                predictor = load_predictor()
                essential_models = [
                    "CatBoost",
                    "ExtraTreesMSE",
                    "LightGBM",
                    "KNeighborsDist",
                    "WeightedEnsemble_L2",
                    "XGBoost",
                ]

                predictions_dict = {}
                for model in essential_models:
                    try:
                        predictions = predictor.predict(input_df, model=model)
                        predictions_dict[model] = predictions
                    except Exception as e:
                        predictions_dict[model] = "Error"

                st.subheader("Prediction Results")
                st.dataframe(pd.DataFrame(predictions_dict).iloc[:1, :])

                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"Model loading failed: {e}")
