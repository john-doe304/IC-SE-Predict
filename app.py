import streamlit as st
import os
import re
import gc
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# --- Matminer ---
from mordred import Calculator, descriptors

# --- Pymatgen & Crystal Toolkit (for structure rendering) ---
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from crystal_toolkit.renderables.structure import StructureMoleculeComponent
from crystal_toolkit.settings import SETTINGS
SETTINGS.DEFAULT_VIEWER = "speck"  # lightweight 3D viewer


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
    /* 减小指标卡片的字体大小 */
    .stMetric {
        font-size: 0.9em;
    }
    /* 减小特征提取成功信息的字体大小 */
    .stWrite {
        font-size: 0.9em;
    }
    /* 减小子标题的字体大小 */
    h3 {
        font-size: 1.2em;
    }
    /* 减小数据框的字体大小 */
    .dataframe {
        font-size: 0.8em;
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





# FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:",placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6", )

# 温度输入
temperature = st.number_input("Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10)

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

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
@st.cache_resource(show_spinner=False, max_entries=1)  # 限制只缓存一个实例
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

def load_from_MP(formula: str):
    """
    Search structure from Materials Project via formula.
    """
    try:
        api_key = st.secrets["MP_API_KEY"]
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula)

            if len(results) == 0:
                return None

            entry = sorted(results, key=lambda x: x.energy_per_atom)[0]
            return entry.structure

    except Exception as e:
        st.warning(f"Materials Project search failed: {e}")
        return None

def load_from_COD(formula: str):
    """
    Search CIF from COD by formula.
    """
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        lines = r.text.strip().split()
        if len(lines) == 0:
            return None

        cod_id = lines[0]
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        cif_bytes = requests.get(cif_url, timeout=10).content

        return Structure.from_str(cif_bytes.decode(), fmt="cif")

    except:
        return None

def load_crystal_structure_public(formula: str):
    """
    Try:
    1. Materials Project
    2. COD
    """
    st.info("Searching public databases for crystal structure...")

    s = load_from_MP(formula)
    if s:
        st.success("Structure found in Materials Project ✓")
        return s

    s = load_from_COD(formula)
    if s:
        st.success("Structure found in COD ✓")
        return s

    st.error("No structure found in Materials Project or COD.")
    return None

def display_crystal_structure(structure):
    """
    Render interactive 3D structure in Streamlit.
    """
    try:
        component = StructureMoleculeComponent(structure, show_compass=True)
        st.write(component)
    except Exception as e:
        st.error(f"Rendering error: {e}")

def mol_to_image(mol, size=(200, 200)):
    """将分子转换为背景颜色为 #f9f9f9f9 的SVG图像"""
    # 创建绘图对象
    d2d = MolDraw2DSVG(size[0], size[1])
    
    # 获取绘图选项
    draw_options = d2d.drawOptions()
    
    # 设置背景颜色为 #f9f9f9f9
    draw_options.background = '#f9f9f9'
    
    # 移除所有边框和填充
    draw_options.padding = 0.0
    draw_options.additionalBondPadding = 0.0
    
    # 移除原子标签的边框
    draw_options.annotationFontScale = 1.0
    draw_options.addAtomIndices = False
    draw_options.addStereoAnnotation = False
    draw_options.bondLineWidth = 1.5
    
    # 禁用所有边框
    draw_options.includeMetadata = False
    
    # 绘制分子
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    
    # 获取SVG内容
    svg = d2d.GetDrawingText()
    
    # 移除SVG中所有可能存在的边框元素
    # 1. 移除黑色边框矩形
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg, flags=re.DOTALL)
    
    # 2. 移除所有空的rect元素
    svg = re.sub(r'<rect[^>]*/>', '', svg, flags=re.DOTALL)
    
    # 3. 确保viewBox正确设置
    if 'viewBox' in svg:
        # 设置新的viewBox以移除边距
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    
    return svg



# 材料特征计算函数
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
    """只显示选定的七个特征"""
    filtered_features = {}
    
    # 添加温度特征
    
    filtered_features['Temp'] = float(temperature)
    
    # 添加选定的七个特征
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            continue
        
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            # 如果特征不存在，设为0
            filtered_features[feature_name] = 0.0
    
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

if submit_button:

    if not formula_input:
        st.error("Please enter a chemical formula.")
    else:

        ###########################################################
        # 1. Load Structure from Public Databases
        ###########################################################
        st.subheader("Crystal Structure")

        structure = load_crystal_structure_public(formula_input)

        if structure:
            display_crystal_structure(structure)

        ###########################################################
        # 2. Calculate Features
        ###########################################################
        st.subheader("Extracting Features...")
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} total features")

        selected_features = filter_selected_features(features, temperature)
        st.dataframe(pd.DataFrame([selected_features]))

        ###########################################################
        # 3. Prediction with AutoGluon
        ###########################################################
        st.subheader("Prediction Results")

        input_df = pd.DataFrame({
            "Formula": [formula_input],
            **{k: [v] for k, v in selected_features.items()}
        })

        try:
            predictor = load_predictor()

            essential_models = [
                "CatBoost", "ExtraTreesMSE", "LightGBM",
                "KNeighborsDist", "WeightedEnsemble_L2", "XGBoost"
            ]

            results = {}
            for model in essential_models:
                try:
                    results[model] = predictor.predict(input_df, model=model)
                except:
                    results[model] = "Error"

            st.dataframe(pd.DataFrame(results).iloc[:1, :])

            del predictor
            gc.collect()

        except Exception as e:
            st.error(f"Prediction failed: {e}")










