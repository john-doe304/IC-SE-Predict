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

# crystal structure
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# ========= Materials Project API KEY ==========
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


# ===================== 模型缓存 =====================
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")


# ====================================================================
# 🔹 1. 统一 API 获取晶体结构（保证与 Materials Project 一致）
# ====================================================================
def get_structure_cif(formula):
    """
    尝试从 Materials Project 获取结构（多种 fallback）
    返回 cif 字符串
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # 第一优先：使用 get_structure_by_material_id 通过 material_id 获取
            try:
                # 先搜索获取 material_id
                results = mpr.query(
                    criteria={"formula": formula},
                    properties=["material_id", "pretty_formula", "spacegroup.symbol"]
                )
                if results:
                    # 获取第一个匹配的 material_id
                    material_id = results[0]["material_id"]
                    st.info(f"Found material: {results[0]['pretty_formula']} ({material_id}) - Space Group: {results[0]['spacegroup.symbol']}")
                    
                    # 使用 material_id 获取精确结构
                    structure = mpr.get_structure_by_material_id(material_id)
                    return structure.to(fmt="cif"), material_id
            except Exception as e:
                st.warning(f"Method 1 failed: {e}")

            # 第二优先：entries
            try:
                entries = mpr.get_entries(formula)
                if entries:
                    structure = entries[0].structure
                    material_id = entries[0].entry_id
                    st.info(f"Found material via entries: {material_id}")
                    return structure.to(fmt="cif"), material_id
            except Exception as e:
                st.warning(f"Method 2 failed: {e}")

            return None, None

    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
        return None, None


# ====================================================================
# 🔹 2. 渲染晶体结构（Materials Project 风格：球棍模型 + 晶胞）
# ====================================================================
def show_structure_3d(cif_string, material_id=None, width=700, height=520):
    """
    使用 py3Dmol 渲染晶体结构，接近 Materials Project 风格
    """
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, "cif")
        
        # Materials Project 风格的原子颜色映射
        element_colors = {
            'H': 'white', 'He': 'cyan', 'Li': 'violet', 'Be': 'darkgreen', 
            'B': 'salmon', 'C': 'black', 'N': 'blue', 'O': 'red', 
            'F': 'green', 'Ne': 'cyan', 'Na': 'yellow', 'Mg': 'darkgreen', 
            'Al': 'darkgray', 'Si': 'goldenrod', 'P': 'orange', 'S': 'yellow', 
            'Cl': 'green', 'Ar': 'cyan', 'K': 'violet', 'Ca': 'darkgreen', 
            'Sc': 'darkgray', 'Ti': 'darkgray', 'V': 'darkgray', 'Cr': 'darkgray', 
            'Mn': 'darkgray', 'Fe': 'orange', 'Co': 'darkgray', 'Ni': 'darkgreen', 
            'Cu': 'darkorange', 'Zn': 'darkgreen', 'Ga': 'darkgray', 'Ge': 'goldenrod', 
            'As': 'darkgray', 'Se': 'yellow', 'Br': 'darkred', 'Kr': 'cyan', 
            'Rb': 'violet', 'Sr': 'darkgreen', 'Y': 'darkgray', 'Zr': 'darkgray', 
            'Nb': 'darkgray', 'Mo': 'darkgray', 'Tc': 'darkgray', 'Ru': 'darkgray', 
            'Rh': 'darkgray', 'Pd': 'darkgray', 'Ag': 'darkgray', 'Cd': 'darkgreen', 
            'In': 'darkgray', 'Sn': 'darkgray', 'Sb': 'darkgray', 'Te': 'darkgray', 
            'I': 'darkviolet', 'Xe': 'cyan', 'Cs': 'violet', 'Ba': 'darkgreen', 
            'La': 'darkgray', 'Ce': 'darkgray', 'Pr': 'darkgray', 'Nd': 'darkgray', 
            'Pm': 'darkgray', 'Sm': 'darkgray', 'Eu': 'darkgray', 'Gd': 'darkgray', 
            'Tb': 'darkgray', 'Dy': 'darkgray', 'Ho': 'darkgray', 'Er': 'darkgray', 
            'Tm': 'darkgray', 'Yb': 'darkgray', 'Lu': 'darkgray', 'Hf': 'darkgray', 
            'Ta': 'darkgray', 'W': 'darkgray', 'Re': 'darkgray', 'Os': 'darkgray', 
            'Ir': 'darkgray', 'Pt': 'darkgray', 'Au': 'gold', 'Hg': 'darkgray', 
            'Tl': 'darkgray', 'Pb': 'darkgray', 'Bi': 'darkgray', 'Po': 'darkgray', 
            'At': 'darkgray', 'Rn': 'cyan', 'Fr': 'violet', 'Ra': 'darkgreen'
        }
        
        # 设置 Materials Project 风格的渲染
        view.setStyle({}, {
            "sphere": {
                "colorscheme": {
                    # 使用自定义颜色映射
                    "prop": "elem",
                    "map": element_colors
                },
                "scale": 0.3  # 稍微调整球体大小
            },
            "stick": {
                "radius": 0.12,  # 调整键的粗细
                "colorscheme": {
                    "prop": "elem",
                    "map": element_colors
                }
            }
        })
        
        # 添加晶胞边界（Materials Project 风格）
        view.addUnitCell({
            "color": "black",
            "radius": 0.05,
            "dashed": True
        })
        
        # 设置背景和视角
        view.setBackgroundColor(0xffffff)  # 白色背景
        
        # 自动调整视角到最佳位置
        view.zoomTo()
        
        # 稍微旋转以获得更好的3D视角
        view.rotate(90, 'x')
        view.rotate(30, 'y')
        
        # 显示 material_id
        if material_id:
            st.info(f"**Material ID:** {material_id}")
        
        # 渲染
        html = view._make_html()
        st.components.v1.html(html, height=height + 20, scrolling=False)
        
    except Exception as e:
        st.error(f"3D visualization error: {e}")


# ====================================================================
# 🔹 3. 获取材料详细信息
# ====================================================================
def get_material_details(material_id):
    """
    获取材料的详细信息
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            data = mpr.query(
                criteria={"material_id": material_id},
                properties=[
                    "pretty_formula",
                    "spacegroup.symbol",
                    "spacegroup.number",
                    "crystal_system",
                    "volume",
                    "density",
                    "formation_energy_per_atom",
                    "band_gap"
                ]
            )
            if data:
                return data[0]
    except Exception as e:
        st.warning(f"Could not fetch detailed material info: {e}")
    return None


# ====================================================================
# 🔹 4. 特征工程（保持原逻辑）
# ====================================================================
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


def filter_selected_features(features, selected, temperature):
    result = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            result[f] = features.get(f, 0.0)
    return result


# ====================================================================
# 🔹 5. 主逻辑（结构 + 特征 + 预测）
# ====================================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ========== 显示晶体结构 ==========
    st.subheader("Crystal Structure (from Materials Project)")

    with st.spinner("Fetching crystal structure from Materials Project..."):
        cif_data, material_id = get_structure_cif(formula_input)

    if cif_data:
        # 显示材料详细信息
        if material_id:
            details = get_material_details(material_id)
            if details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Formula", details.get("pretty_formula", "N/A"))
                    st.metric("Space Group", details.get("spacegroup.symbol", "N/A"))
                with col2:
                    st.metric("Crystal System", details.get("crystal_system", "N/A"))
                    st.metric("Volume (Å³)", f"{details.get('volume', 0):.2f}")
                with col3:
                    st.metric("Density (g/cm³)", f"{details.get('density', 0):.2f}")
                    st.metric("Band Gap (eV)", f"{details.get('band_gap', 0):.2f}")
        
        # 显示3D结构
        show_structure_3d(cif_data, material_id)
        
        # 提供CIF文件下载
        st.download_button(
            label="📥 Download CIF File",
            data=cif_data,
            file_name=f"{formula_input.replace(' ', '_')}_{material_id or 'structure'}.cif",
            mime="chemical/x-cif"
        )
    else:
        st.warning("No structure found for this formula in Materials Project.")

    # ========== 特征 + 预测 ==========
    with st.spinner("Processing material and making predictions..."):

        features = calculate_material_features(formula_input)
        st.write(f"✅ Total features extracted: {len(features)}")

        selected = filter_selected_features(
            features, required_descriptors, temperature
        )

        st.subheader("Material Features")
        st.dataframe(pd.DataFrame([selected]))

        # prepare ML input
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)

        # 预测部分
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
