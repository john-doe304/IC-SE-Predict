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
# 如果你有自己的 API key，请替换下面的值；保密处理请不要公开分享 Key
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

# ===================== 输入区 =====================
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
    # 请确认模型目录存在且包含训练好的 predictor
    return TabularPredictor.load("./ag-20251024_075719")


# ====================================================================
# 🔹 1. 从 Materials Project 获取晶体结构（多方法尝试，增强兼容性）
# ====================================================================
def get_structure_cif(formula):
    """
    尝试使用多种常见 MPRester 接口来获取结构并返回 cif 字符串。
    兼容不同 pymatgen 版本。
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # 1) 尝试 summary.search（某些版本可用）
            try:
                if hasattr(mpr, "summary") and hasattr(mpr.summary, "search"):
                    res = mpr.summary.search(formula=formula)
                    if res:
                        # res 元素可能有 .structure 或 .entry
                        first = res[0]
                        if hasattr(first, "structure"):
                            return first.structure.to(fmt="cif")
                        # 有些版本返回 dict-like summary
                        if isinstance(first, dict) and "material_id" in first:
                            mid = first["material_id"]
                            struct = mpr.get_structure_by_material_id(mid)
                            return struct.to(fmt="cif")
            except Exception:
                # 忽略并尝试下一种方法
                pass

            # 2) 尝试 mpr.query（常见且稳定）
            try:
                if hasattr(mpr, "query"):
                    query_res = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                    if query_res:
                        mid = query_res[0].get("material_id")
                        if mid:
                            struct = mpr.get_structure_by_material_id(mid)
                            return struct.to(fmt="cif")
            except Exception:
                pass

            # 3) 尝试 get_entries（有时可用）
            try:
                if hasattr(mpr, "get_entries"):
                    entries = mpr.get_entries(formula)
                    if entries:
                        entry = entries[0]
                        if hasattr(entry, "structure"):
                            return entry.structure.to(fmt="cif")
            except Exception:
                pass

            # 4) 最后尝试 get_structures_by_formula（某些 pymatgen 版本提供该方法）
            try:
                if hasattr(mpr, "get_structures"):
                    structs = mpr.get_structures(formula)
                    if structs:
                        s0 = structs[0]
                        if isinstance(s0, Structure):
                            return s0.to(fmt="cif")
                        # 有时返回 dict/list 中包含 structure
            except Exception:
                pass

            # 若所有方式均失败，返回 None
            return None

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None


# ====================================================================
# 🔹 2. 通过 py3Dmol 渲染晶体结构（Streamlit 完整兼容）
# ====================================================================
def show_structure_3d(cif_string, width=700, height=520):
    """
    在 Streamlit 中显示 py3Dmol 生成的 3D 视图。
    注意：不要使用 view.show()（Jupyter 专用），而是通过 HTML 注入。
    """
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, "cif")
        # 你可以更改展示样式：'stick','sphere','line','cartoon' 等
        view.setStyle({"stick": {}})
        view.addUnitCell()
        view.zoomTo()
        html = view._make_html()
        st.components.v1.html(html, height=height + 20, scrolling=False)
    except Exception as e:
        st.error(f"3D visualization error: {e}")


# ====================================================================
# 🔹 3. 特征工程（保持原逻辑）
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


# ====================================================================
# 🔹 4. 主逻辑（整合显示 + 特征 + 预测）
# ====================================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ===================== 显示晶体结构 =====================
    st.subheader("Crystal Structure (from Materials Project)")
    cif_data = get_structure_cif(formula_input)

    if cif_data:
        show_structure_3d(cif_data)
    else:
        st.warning("No structure found for this material in Materials Project.")

    # ===================== 特征 + 预测 =====================
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
        else:
            st.error("Predictor unavailable; cannot generate predictions.")
