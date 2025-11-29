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
import gc
import re
from tqdm import tqdm
import numpy as np

# ========== 晶体结构依赖 ==========
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# ======= Materials Project API KEY =======
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# ===================== Streamlit 样式 =====================
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
    <h2 style="text-align:center;">Predict Ionic Conductivity of Solid Electrolytes</h2>
    """,
    unsafe_allow_html=True,
)

# ===================== 输入区 =====================
formula_input = st.text_input(
    "Enter Chemical Formula:",
    placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6",
)

temperature = st.number_input(
    "Temperature (K):",
    min_value=200,
    max_value=1000,
    value=298,
    step=10,
)

submit_button = st.button("Submit and Predict")


@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ====================================================================
# 🔹（已修复）新版 MP API —— 正确获取材料结构
# ====================================================================
def get_structure_cif(formula):
    """
    Uses ONLY the new MP API (2023+) —确保不会调用不存在的旧接口
    """

    try:
        with MPRester(MP_API_KEY) as mpr:

            # 1) 用 summary.search 查找材料
            results = mpr.materials.summary.search(formula=formula)

            if not results:
                return None

            # 2) 取第一个 material_id
            material_id = results[0].material_id

            # 3) 获取结构（新版正确接口）
            struct_dict = mpr.materials.structures.get(material_id)

            if "structure" not in struct_dict:
                return None

            struct = Structure.from_dict(struct_dict["structure"])

            return struct.to(fmt="cif")

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None


# ====================================================================
# 🔹 正常显示 py3Dmol（无 Jupyter，可用）
# ====================================================================
def show_structure_3d(cif_string, width=700, height=520):
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, "cif")
        view.setStyle({"stick": {}})
        view.addUnitCell()
        view.zoomTo()

        html = view._make_html()
        st.components.v1.html(html, height=height + 20, scrolling=False)

    except Exception as e:
        st.error(f"3D visualization error: {e}")


# ====================================================================
# 🔹 特征工程
# ====================================================================
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


# ====================================================================
# 🔹 主程序
# ====================================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ========== 显示晶体结构 ==========
    st.subheader("Crystal Structure")

    cif_data = get_structure_cif(formula_input)

    if cif_data:
        show_structure_3d(cif_data)
    else:
        st.warning("No structure found for this material.")

    # ========== 特征提取 ==========
    with st.spinner("Calculating features..."):
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

        # ========== 模型预测 ==========
        try:
            predictor = load_predictor()
        except Exception as e:
            st.error(f"Failed to load predictor: {e}")
            predictor = None

        if predictor is not None:
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
                except Exception:
                    results[m] = "Error"

            st.subheader("Prediction Results")
            st.dataframe(pd.DataFrame(results).iloc[:1, :])

            del predictor
            gc.collect()

