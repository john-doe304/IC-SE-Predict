import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import gc
import numpy as np
import re
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# ========= Materials Project API KEY ==========
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# ================== Streamlit 页面样式 ==================
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
    .legend-container {
        position: relative;
        width: 750px;
        margin-top: -40px;
        text-align: right;
        padding-right: 20px;
        font-size: 18px;
        font-weight: 500;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-left: 12px;
    }
    .legend-circle {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================== 输入区 ===================
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


# ============ 模型加载缓存 ============
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ======================================================
# 🔹 1. 统一 API 获取 CIF
# ======================================================
def get_structure_cif(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:

            # 新 API summary
            try:
                res = mpr.summary.search(formula=formula)
                if res:
                    s = res[0].structure
                    return s.to(fmt="cif")
            except:
                pass

            # 经典 query
            try:
                q = mpr.query({"formula": formula}, ["material_id"])
                if q:
                    mid = q[0]["material_id"]
                    s = mpr.get_structure_by_material_id(mid)
                    return s.to(fmt="cif")
            except:
                pass

            # entries
            try:
                ents = mpr.get_entries(formula)
                if ents:
                    return ents[0].structure.to(fmt="cif")
            except:
                pass

            # get_structures
            try:
                structs = mpr.get_structures(formula)
                if structs:
                    return structs[0].to(fmt="cif")
            except:
                pass

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")

    return None


# ======================================================
# 🔹 py3Dmol 内部原子颜色表（Jmol 颜色）
# ======================================================
JMOL_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "O": "#FF0D0D", "La": "#70D4FF",
    "Zr": "#94E0E0", "S": "#FFFF30", "P": "#FF8000", "Cl": "#1FF01F",
    "Ge": "#668C8C", "Y": "#94FFFF", "Mg": "#B5E8FF", "Ca": "#AFFF8F",
    "Ba": "#00C1FF"
}


# ======================================================
# 🔹 2. 渲染晶体结构 + 图例
# ======================================================
def show_structure_3d_with_legend(cif_string, width=720, height=520):

    # ---------- 获取结构 ----------
    structure = Structure.from_str(cif_string, fmt="cif")

    # ---------- 获取唯一元素 ----------
    elements = sorted({str(site.specie) for site in structure})

    # ---------- 渲染晶体结构 ----------
    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_string, "cif")

    view.setStyle({
        "sphere": {"scale": 0.28},
        "stick": {"radius": 0.15}
    })

    view.addUnitCell()
    view.zoomTo()

    html3d = view._make_html()

    # ---------- 渲染 3D 图 ----------
    st.components.v1.html(html3d, height=height + 20, scrolling=False)

    # ---------- 构建图例 HTML ----------
    legend_html = """<div class="legend-container">"""

    for elem in elements:
        color = JMOL_COLORS.get(elem, "#BBBBBB")  # 默认灰色
        legend_html += f"""
        <div class="legend-item">
            <span class="legend-circle" style="background-color:{color};"></span>
            {elem}
        </div>
        """

    legend_html += "</div>"

    st.markdown(legend_html, unsafe_allow_html=True)


# ======================================================
# 🔹 3. 特征工程（保持你的原逻辑）
# ======================================================
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

        # 数值列转为 dict
        for col in df.select_dtypes(include=[np.number]).columns:
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


def filter_selected_features(features, selected, temperature):
    result = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            result[f] = features.get(f, 0.0)
    return result


# ======================================================
# 🔹 4. 主逻辑
# ======================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ================== 显示晶体结构 ==================
    st.subheader("Crystal Structure (from Materials Project)")

    cif_data = get_structure_cif(formula_input)

    if cif_data:
        show_structure_3d_with_legend(cif_data)
    else:
        st.warning("No structure found for this formula in Materials Project.")

    # ================== 特征 + 预测 ==================
    with st.spinner("Processing material and making predictions..."):

        features = calculate_material_features(formula_input)
        st.write(f"Total features extracted: {len(features)}")

        selected = filter_selected_features(
            features, required_descriptors, temperature
        )

        st.subheader("Material Features")
        st.dataframe(pd.DataFrame([selected]))

        # 构建输入
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)

        try:
            predictor = load_predictor()
        except:
            predictor = None
            st.error("Failed to load predictor.")

        if predictor:
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
