[file name]: image.png
[file content begin]
# Crystal Structure Information

Material ID:  
mp-942733  

Formula: L17La3Zr2O12  



Density: 5.01 g/cm³  

Volume: 1112.63 Å³  

Formation Energy: -7.484 eV/atom  

---

## Structure Analysis

Structure Type: Orthorhombic/triclinic  

Symmetry: Low  

---

## Crystal Structure

### Crystal Structure: L17La3Zr2O12

---

## Crystal Structure

L17La3Zr2O12  

View detailed structure on Materials Project  

---

**Total features extracted: 276**


[file content end]

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
import gc  # 添加垃圾回收模块
import re  # 添加正则表达式模块用于处理SVG
from tqdm import tqdm 
import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.ext.matproj import MPRester
import plotly.graph_objects as go
import io
import requests
from PIL import Image
import base64

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
    /* 晶体结构图片样式 */
    .crystal-image-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .crystal-image {
        max-width: 100%;
        border-radius: 8px;
        margin: 10px 0;
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

# Materials Project API 密钥输入
mp_api_key = st.text_input("Materials Project API Key (optional):", 
                          placeholder="Enter your API key to view crystal structure",
                          type="password",
                          value="Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN")

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
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

def validate_chemical_formula(formula):
    """验证化学公式格式"""
    if not formula:
        return False, "Formula cannot be empty"
    
    invalid_chars = set('!@#$%^&*()_+=[]{}|;:,<>?`~')
    if any(char in formula for char in invalid_chars):
        return False, "Formula contains invalid characters"
    
    if not any(c.isalpha() for c in formula):
        return False, "Formula must contain chemical elements"
    
    return True, "Valid formula"

def get_materials_project_structure_simple(formula, api_key):
    """使用简单直接的方法获取晶体结构信息"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format. API key should be 32 alphanumeric characters."
        
        with MPRester(api_key) as mpr:
            # 方法1: 直接使用get_entries获取结构
            try:
                entries = mpr.get_entries(formula, inc_structure=True)
                
                if not entries:
                    return None, f"No materials found for formula: {formula}"
                
                # 选择第一个材料（通常是最稳定的）
                material = entries[0]
                structure = material.structure
                material_id = material.entry_id
                
                # 获取材料的基本信息
                try:
                    # 使用summary.search获取详细信息
                    summary_results = mpr.summary.search(material_id=material_id, fields=[
                        "formula_pretty", "spacegroup", "density", "volume", 
                        "formation_energy_per_atom", "band_gap"
                    ])
                    
                    if summary_results:
                        material_data = summary_results[0]
                        pretty_formula = material_data.formula_pretty
                        spacegroup_data = material_data.spacegroup
                        spacegroup_symbol = spacegroup_data.symbol if spacegroup_data else "N/A"
                        spacegroup_number = spacegroup_data.number if spacegroup_data else "N/A"
                        density = material_data.density
                        volume = material_data.volume
                        formation_energy = material_data.formation_energy_per_atom
                        band_gap = material_data.band_gap
                    else:
                        # 如果summary.search失败，使用基本数据
                        pretty_formula = formula
                        spacegroup_symbol = "N/A"
                        spacegroup_number = "N/A"
                        density = structure.density
                        volume = structure.volume
                        formation_energy = material.energy_per_atom
                        band_gap = "N/A"
                        
                except Exception as detail_error:
                    # 如果获取详细信息失败，使用基本数据
                    pretty_formula = formula
                    spacegroup_symbol = "N/A"
                    spacegroup_number = "N/A"
                    density = structure.density
                    volume = structure.volume
                    formation_energy = material.energy_per_atom
                    band_gap = "N/A"
                
                return {
                    'structure': structure,
                    'material_id': material_id,
                    'spacegroup': {
                        'symbol': spacegroup_symbol,
                        'number': spacegroup_number
                    },
                    'density': density,
                    'volume': volume,
                    'formation_energy_per_atom': formation_energy,
                    'band_gap': band_gap,
                    'formula': formula,
                    'pretty_formula': pretty_formula
                }, None
                
            except Exception as entries_error:
                # 方法2: 使用直接的结构获取
                try:
                    # 搜索材料ID
                    search_results = mpr.summary.search(formula=formula, fields=["material_id"])
                    if not search_results:
                        return None, f"No materials found for formula: {formula}"
                    
                    material_id = search_results[0].material_id
                    
                    # 直接获取结构
                    structure = mpr.get_structure_by_material_id(material_id)
                    
                    return {
                        'structure': structure,
                        'material_id': material_id,
                        'spacegroup': {'symbol': 'N/A', 'number': 'N/A'},
                        'density': structure.density,
                        'volume': structure.volume,
                        'formation_energy_per_atom': 'N/A',
                        'band_gap': 'N/A',
                        'formula': formula,
                        'pretty_formula': formula
                    }, None
                    
                except Exception as direct_error:
                    return None, f"All methods failed: {str(direct_error)}"
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def display_crystal_structure_image(material_id, formula, api_key):
    """显示晶体结构信息，包括直接链接和图片"""
    try:
        st.subheader("🎯 Crystal Structure")
        
        # 清理material_id（去掉-mp后缀）
        clean_material_id = material_id.split('-')[0] if '-' in material_id else material_id
        
        # 尝试多种图片URL格式
        image_urls = [
            f"https://next-gen.materialsproject.org/materials/{clean_material_id}/image",
            f"https://legacy.materialsproject.org/materials/{clean_material_id}/image",
            f"https://materialsproject.org/materials/{clean_material_id}/image"
        ]
        
        image_found = False
        
        for image_url in image_urls:
            try:
                headers = {
                    "X-API-KEY": api_key,
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(image_url, headers=headers, timeout=10)
                
                if response.status_code == 200 and response.content:
                    # 成功获取图片
                    image = Image.open(BytesIO(response.content))
                    st.markdown(f"""
                    <div class="crystal-image-container">
                        <h4>Crystal Structure: {formula}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(image, caption=f"Crystal Structure: {formula}", use_column_width=True)
                    image_found = True
                    break
                    
            except Exception as img_error:
                continue
        
        if not image_found:
            # 如果所有图片URL都失败，显示占位图和链接
            st.markdown(f"""
            <div class="crystal-image-container">
                <h4>Crystal Structure: {formula}</h4>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          padding: 60px 20px; border-radius: 8px; color: white; margin: 20px 0;">
                    <h3>🔬 Crystal Structure</h3>
                    <p><strong>{formula}</strong></p>
                    <p>View detailed structure on Materials Project</p>
                </div>
                <p style="color: #666; margin-top: 10px;">
                    The crystal structure image is available on Materials Project website
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 添加查看详情链接 - 尝试多种URL格式
        material_urls = [
            f"https://next-gen.materialsproject.org/materials/{clean_material_id}",
            f"https://legacy.materialsproject.org/materials/{clean_material_id}",
            f"https://materialsproject.org/materials/{clean_material_id}"
        ]
        
        st.markdown("""
        <div style="text-align: center; margin: 15px 0;">
            <p><strong>View Interactive Structure:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, url in enumerate(material_urls):
            st.markdown(f"""
            <div style="text-align: center; margin: 10px 0;">
                <a href="{url}" target="_blank" style="
                    display: inline-block;
                    padding: 8px 16px;
                    background-color: #1976d2;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 0.9em;
                    margin: 5px;
                ">
                🔍 Link {i+1} - Materials Project
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        return image_found
        
    except Exception as e:
        st.error(f"Error displaying crystal structure: {str(e)}")
        return False

def analyze_structure_features(structure):
    """分析晶体结构特征"""
    try:
        # 计算密度
        density = structure.density
        
        # 判断结构类型
        lattice_type = "unknown"
        symmetry = "low"
        
        # 分析晶格参数判断对称性
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        # 判断立方晶系
        if abs(a - b) < 0.1 and abs(b - c) < 0.1 and all(abs(angle - 90) < 1 for angle in [alpha, beta, gamma]):
            lattice_type = "cubic"
            symmetry = "high"
        # 判断四方晶系
        elif abs(a - b) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1:
            lattice_type = "tetragonal"
            symmetry = "medium"
        # 判断六方晶系
        elif abs(a - b) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 120) < 1:
            lattice_type = "hexagonal"
            symmetry = "medium"
        else:
            lattice_type = "orthorhombic/triclinic"
            symmetry = "low"
        
        return {
            'density': density,
            'structure_type': lattice_type,
            'symmetry': symmetry
        }
        
    except Exception as e:
        return {
            'density': 'N/A',
            'structure_type': 'unknown',
            'symmetry': 'unknown'
        }

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

# 如果点击提交按钮
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        # 验证化学公式
        is_valid, validation_msg = validate_chemical_formula(formula_input)
        
        if not is_valid:
            st.error(f"Invalid chemical formula: {validation_msg}")
            st.info("💡 Please use standard chemical notation like: Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")
        else:
            with st.spinner("Processing material and making predictions..."):
                try:
                    # 首先尝试从Materials Project获取晶体结构
                    if mp_api_key and mp_api_key.strip():
                        with st.spinner("Fetching crystal structure from Materials Project..."):
                            # 修正化学公式
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I').replace('3272', '3Zr2')
                            
                            mp_data, mp_error = get_materials_project_structure_simple(corrected_formula, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                
                                # 显示材料信息
                                st.subheader("📊 Crystal Structure Information")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Material ID:** `{mp_data['material_id']}`")
                                    st.write(f"**Formula:** {mp_data['pretty_formula']}")
                                    st.write(f"**Space Group:** {mp_data['spacegroup'].get('symbol', 'N/A')} ({mp_data['spacegroup'].get('number', 'N/A')})")
                                    
                                with col2:
                                    if mp_data['density'] != 'N/A':
                                        st.write(f"**Density:** {mp_data['density']:.2f} g/cm³")
                                    else:
                                        st.write(f"**Density:** N/A")
                                    if mp_data['volume'] != 'N/A':
                                        st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
                                    else:
                                        st.write(f"**Volume:** N/A")
                                    if mp_data['formation_energy_per_atom'] != 'N/A':
                                        st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
                                    else:
                                        st.write(f"**Formation Energy:** N/A")
                                
                                # 分析结构特征
                                structure_info = analyze_structure_features(mp_data['structure'])
                                
                                # 显示结构分析
                                st.subheader("🔬 Structure Analysis")
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.write(f"**Structure Type:** {structure_info['structure_type'].capitalize()}")
                                with col4:
                                    st.write(f"**Symmetry:** {structure_info['symmetry'].capitalize()}")
                                
                                # 显示晶体结构图片和链接
                                display_crystal_structure_image(
                                    mp_data['material_id'], 
                                    mp_data['pretty_formula'],
                                    mp_api_key
                                )
                                
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                st.info("💡 The material might not exist in Materials Project database, or try a different formula")
                    else:
                        st.info("💡 Enter a Materials Project API key to view crystal structure information")
                    
                    # 计算材料特征
                    features = calculate_material_features(formula_input)
                    st.write(f"✅ Total features extracted: {len(features)}")
                
                    # 只显示选定的七个特征
                    selected_features = filter_selected_features(features, required_descriptors, temperature)
                    feature_df = pd.DataFrame([selected_features])
                
                    st.subheader("Material Features")
                    st.dataframe(feature_df)
            
                    if features:
                        # 创建输入数据
                        input_data = {
                            "Formula": [formula_input],
                            "Temp": [temperature],
                        }
                    
                        # 添加数值特征
                        numeric_features = {}
                        for feature_name in required_descriptors:
                            if feature_name == 'Temp':
                                numeric_features[feature_name] = [temperature]
                            elif feature_name in features:
                                numeric_features[feature_name] = [features[feature_name]]
                            else:
                                numeric_features[feature_name] = [0.0]  # 默认值
                        
                        input_data.update(numeric_features)
                        
                        input_df = pd.DataFrame(input_data)
                  
                    # 加载模型并预测
                    try:
                        # 使用缓存的模型加载方式
                        predictor = load_predictor()
                    
                        # 只使用最关键的模型进行预测，减少内存占用
                        essential_models = ['CatBoost',
                                        'ExtraTreesMSE',
                                        'LightGBM',
                                        'KNeighborsDist',
                                        'WeightedEnsemble_L2',
                                        'XGBoost']
                        predict_df = input_df.copy()
                        predictions_dict = {}
                    
                        for model in essential_models:
                            try:
                                predictions = predictor.predict(predict_df, model=model)
                                predictions_dict[model] = predictions
                            except Exception as model_error:
                                st.warning(f"Model {model} prediction failed: {str(model_error)}")
                                predictions_dict[model] = "Error"

                        # 显示预测结果
                        st.write("Prediction Results (Essential Models):")
                        st.markdown(
                            "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                        results_df = pd.DataFrame(predictions_dict)
                        st.dataframe(results_df.iloc[:1,:])
                    
                        # 主动释放内存
                        del predictor
                        gc.collect()

                    except Exception as e:
                        st.error(f"Model loading failed: {str(e)}")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

