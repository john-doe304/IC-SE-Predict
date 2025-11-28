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
import py3Dmol

# ======== MATERIALS PROJECT ========
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

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

# ===================== 用户输入 =====================
formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")
temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=10)
submit_button = st.button("Submit and Predict")

@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

# ====================================================================
# 🔹 FIXED — Materials Project (2023+ API)
# ====================================================================
def get_structure_cif(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:
            results = mpr.materials.summary.search(formula=formula)
            if not results:
                return None

            material_id = results[0].material_id
            struct_dict = mpr.materials.structures.get(material_id)

            if "structure" not in struct_dict:
                return None

            struct = Structure.from_dict(struct_dict["structure"])
            return struct
    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None

# ====================================================================
# 🔥 高级 3D 可视化 —— Stick + Ball + Polyhedra + Auto Color
# ====================================================================
def show_structure_3d_advanced(struct: Structure, width=700, height=520):
    view = py3Dmol.view(width=width, height=height)

    cif_str = struct.to(fmt="cif")
    view.addModel(cif_str, "cif")

    # ======= 自动颜色方案 =======
    color_map = {
        "Li": "0x00ff00",
        "O": "0xff0000",
        "S": "0xffff00",
        "P": "0xff8000",
        "Cl": "0x00ffff",
        "Br": "0x9966ff",
        "I": "0x6600ff",
        "La": "0xffffff",
        "Zr": "0x999999",
        "Ge": "0x00aaff",
    }

    # ====== Stick + Ball ======
    for el in struct.composition.elements:
        sym = el.symbol
        view.setStyle(
            {"elem": sym},
            {
                "stick": {"radius": 0.15},
                "sphere": {
                    "scale": 0.28,
                    "color": color_map.get(sym, "0xaaaaaa"),
                },
            },
        )

    # ========== Polyhedra 自动渲染 ==========
    try:
        sites = struct.sites
        for i, site in enumerate(sites):
            el = site.specie.symbol

            # 自动寻找配位环境
            neighbors = struct.get_neighbors(site, 3.0)

            # 四面体 / 八面体判断
            if 3 <= len(neighbors) <= 4:
                style = "tetrahedron"
            elif 5 <= len(neighbors) <= 6:
                style = "octahedron"
            else:
                continue

            coords = [site.frac_coords.tolist()] + [n[0].frac_coords.tolist() for n in neighbors[:4]]

            view.addPolyhedra(
                {
                    "vertexArr": coords,
                    "color": color_map.get(el, "0xffffff"),
                    "opacity": 0.4,
                }
            )
    except:
        pass

    # 显示 Unit Cell
    view.addUnitCell()

    # 自适应缩放
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=height + 20, scrolling=False)

# ====================================================================
# 🔹 特征工程（保持你之前的功能）
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

def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered = {"Temp": float(temperature)}
    for f in selected_descriptors:
        if f != "Temp":
            filtered[f] = features_dict.get(f, 0.0)
    return filtered

# ====================================================================
# 🔥 主程序
# ====================================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ========== 显示晶体结构 ==========  
    st.subheader("Crystal Structure")
    struct = get_structure_cif(formula_input)

    if struct:
        show_structure_3d_advanced(struct)
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

        input_df = pd.DataFrame([selected_features])

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
