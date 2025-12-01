# app.py — 完整版（formula 查询 + conventional cell + py3Dmol 可视化 + 元素图例 + 特征+预测）
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
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
# 把这里换成你的 API KEY（不要公开分享）
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

# ===================== Streamlit 页面样式 =====================
st.set_page_config(page_title="Solid Electrolyte Predictor", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 16px;
        margin: 24px auto;
        max-width: 980px;
        background-color: #ffffff;
        padding: 18px;
        box-sizing: border-box;
    }
    .legend-row { display:flex; flex-wrap:wrap; gap:12px; justify-content:flex-end; align-items:center; }
    .legend-item { display:flex; align-items:center; gap:8px; font-size:14px; }
    .legend-circle { width:14px; height:14px; border-radius:50%; border:1px solid #222; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom:8px;">
        <h2 style="margin:6px 0;">Predict Ionic Conductivity of Solid Electrolytes</h2>
        <div style="color:#555;">Enter a chemical formula below to fetch structure from Materials Project and predict ionic conductivity.</div>
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
        st.warning(f"Cannot load predictor: {e}")
        return None

# ---------------------------
# 元素颜色映射（可扩展）
# 尽量与 Materials Project / common palettes 近似
# ---------------------------
DEFAULT_ELEMENT_COLORS = {
    "Li": "#CC80FF",
    "La": "#70D4FF",
    "Zr": "#94E0E0",
    "O": "#FF0D0D",
    "Cl": "#1FF01F",
    "Y": "#66CCFF",
    "Ge": "#668C8C",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Mg": "#B5E8FF",
    "Ca": "#AFFF8F",
    "Ba": "#00C1FF",
    # add more if needed; fallback will be grey
}

def get_color_for_element(elem):
    return DEFAULT_ELEMENT_COLORS.get(elem, "#BDBDBD")


# ====================================================================
# 🔹 1. 从 Materials Project 获取晶体结构（优先返回 conventional cell）
# ====================================================================
def get_structure_cif(formula):
    """
    使用 pymatgen 的 MPRester 尝试多种方式获取结构，优先返回 conventional (standard) cell，
    并将其轉換为 CIF 字符串。兼容不同 pymatgen 版本。
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # 1) 新 API: mpr.materials.summary.search
            try:
                if hasattr(mpr, "summary") and hasattr(mpr.summary, "search"):
                    res = mpr.summary.search(formula=formula)
                    if res:
                        first = res[0]
                        # some summary results include structure directly
                        if hasattr(first, "structure") and first.structure is not None:
                            struct = first.structure
                        else:
                            # try material_id then structures endpoint
                            matid = None
                            if isinstance(first, dict) and "material_id" in first:
                                matid = first["material_id"]
                            elif hasattr(first, "material_id"):
                                matid = first.material_id
                            if matid:
                                # request structures endpoint and ask for conventional cell if supported
                                try:
                                    sdict = mpr.materials.structures.get(matid)
                                    if isinstance(sdict, dict) and "structure" in sdict:
                                        struct = Structure.from_dict(sdict["structure"])
                                    else:
                                        # maybe it's already a Structure object
                                        struct = mpr.get_structure_by_material_id(matid)
                                except Exception:
                                    # fallback
                                    struct = mpr.get_structure_by_material_id(matid)
                        # try to get conventional standard cell if available
                        try:
                            struct_conv = struct.get_conventional_standard_structure()
                            struct = struct_conv
                        except Exception:
                            # ignore if method not available
                            pass
                        return struct.to(fmt="cif")
            except Exception:
                pass

            # 2) classic query
            try:
                if hasattr(mpr, "query"):
                    query_res = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                    if query_res:
                        mid = query_res[0].get("material_id")
                        if mid:
                            # try structures endpoint first (newer API)
                            try:
                                sdict = mpr.materials.structures.get(mid)
                                if isinstance(sdict, dict) and "structure" in sdict:
                                    struct = Structure.from_dict(sdict["structure"])
                                else:
                                    struct = mpr.get_structure_by_material_id(mid)
                            except Exception:
                                struct = mpr.get_structure_by_material_id(mid)
                            try:
                                struct_conv = struct.get_conventional_standard_structure()
                                struct = struct_conv
                            except Exception:
                                pass
                            return struct.to(fmt="cif")
            except Exception:
                pass

            # 3) fallback get_entries
            try:
                if hasattr(mpr, "get_entries"):
                    entries = mpr.get_entries(formula)
                    if entries:
                        entry = entries[0]
                        if hasattr(entry, "structure"):
                            struct = entry.structure
                            try:
                                struct_conv = struct.get_conventional_standard_structure()
                                struct = struct_conv
                            except Exception:
                                pass
                            return struct.to(fmt="cif")
            except Exception:
                pass

            # 4) fallback get_structures
            try:
                if hasattr(mpr, "get_structures"):
                    structs = mpr.get_structures(formula)
                    if structs and len(structs) > 0:
                        struct = structs[0]
                        try:
                            struct_conv = struct.get_conventional_standard_structure()
                            struct = struct_conv
                        except Exception:
                            pass
                        return struct.to(fmt="cif")
            except Exception:
                pass

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None

    return None


# ====================================================================
# 🔹 2. 使用 py3Dmol 渲染晶体结构并在右下角显示元素图例
# ====================================================================
def show_structure_with_legend(cif_string, width=820, height=560):
    """
    Render CIF with py3Dmol and show element legend (colors matched with rendering).
    """
    try:
        # parse structure to get element list (pymatgen.Structure)
        struct = Structure.from_str(cif_string, fmt="cif")
        elements = sorted({str(s.specie) for s in struct})

        # Build py3Dmol view
        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, "cif")

        # Apply per-element styles so legend colors match
        # set default style small spheres + sticks
        view.setStyle({}, {"sphere": {"scale": 0.28}, "stick": {"radius": 0.15}})

        # override per-element colors/sizes if present
        for el in elements:
            color = get_color_for_element(el)
            # larger sphere for cations, keep generic for others
            try:
                view.setStyle({"elem": el}, {"sphere": {"scale": 0.35}, "stick": {"radius": 0.15}, "color": color})
            except Exception:
                # fallback to generic style
                pass

        view.addUnitCell()
        view.zoomTo()
        html = view._make_html()

        # show 3D viewer
        st.components.v1.html(html, height=height + 20, scrolling=False)

        # build legend HTML — place it below viewer; style to align right similar to MP
        legend_html = '<div style="display:flex; justify-content:flex-end; margin-top:6px;">'
        legend_html += '<div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">'
        for el in elements:
            color = get_color_for_element(el)
            # circle + element label
            legend_html += f'<div style="display:flex; align-items:center; gap:6px; font-size:14px;">'
            legend_html += f'<span style="width:14px; height:14px; border-radius:50%; display:inline-block; background:{color}; border:1px solid #333;"></span>'
            legend_html += f'<span style="color:#222;">{el}</span>'
            legend_html += '</div>'
        legend_html += '</div></div>'

        st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"3D visualization error: {e}")


# ====================================================================
# 🔹 3. 特征工程（沿用你原来的实现）
# ====================================================================
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
        import traceback
        print(traceback.format_exc())
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


# 自动匹配模型特征（保持你原有逻辑）
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


# ====================================================================
# 🔹 4. 主逻辑（点击提交）
# ====================================================================
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
                # 1) 获取并显示晶体结构（conventional-ish）
                st.subheader("Crystal Structure (from Materials Project)")
                cif_data = get_structure_cif(formula_input)

                if cif_data:
                    show_structure_with_legend(cif_data, width=860, height=540)
                else:
                    st.warning("No structure found for this formula in Materials Project.")

                # 2) 特征提取
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")

                selected_features = filter_selected_features(features, required_descriptors, temperature)
                st.subheader("Material Features")
                st.dataframe(pd.DataFrame([selected_features]))

                # 3) 构建输入并预测
                input_data = {"Formula": [formula_input], "Temp": [temperature]}
                for feature_name in required_descriptors:
                    if feature_name == 'Temp':
                        continue
                    input_data[feature_name] = [features.get(feature_name, 0.0)]
                input_df = pd.DataFrame(input_data)

                predictor = load_predictor()
                if predictor is None:
                    st.error("Predictor unavailable; cannot generate predictions.")
                else:
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
                        except Exception as err:
                            results[m] = f"Error: {err}"

                    st.subheader("Prediction Results")
                    st.dataframe(pd.DataFrame(results).iloc[:1, :])

                    del predictor
                    gc.collect()

            except Exception as e:
                st.error(f"An error occurred: {e}")
