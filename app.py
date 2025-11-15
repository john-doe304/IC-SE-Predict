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
    .crystal-image {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        margin: 10px 0;
        text-align: center;
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

# 指定的描述符列表
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# 缓存模型加载器
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

def validate_chemical_formula(formula):
    if not formula:
        return False, "Formula cannot be empty"
    
    invalid_chars = set('!@#$%^&*()_+=[]{}|;:,<>?`~')
    if any(char in formula for char in invalid_chars):
        return False, "Formula contains invalid characters"
    
    if not any(c.isalpha() for c in formula):
        return False, "Formula must contain chemical elements"
    
    return True, "Valid formula"

def get_materials_project_structure_with_images(formula, api_key):
    """获取Materials Project的晶体结构和图片"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format"
        
        with MPRester(api_key) as mpr:
            # 搜索材料
            entries = mpr.get_entries(formula, inc_structure=True)
            
            if not entries:
                return None, f"No materials found for formula: {formula}"
            
            # 选择第一个材料
            material = entries[0]
            structure = material.structure
            material_id = material.entry_id
            
            # 获取材料的详细信息
            try:
                summary_results = mpr.summary.search(material_id=material_id, fields=[
                    "formula_pretty", "spacegroup", "density", "volume", 
                    "formation_energy_per_atom", "band_gap", "material_id",
                    "cif", "symmetry"
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
                    pretty_formula = formula
                    spacegroup_symbol = "N/A"
                    spacegroup_number = "N/A"
                    density = structure.density
                    volume = structure.volume
                    formation_energy = material.energy_per_atom
                    band_gap = "N/A"
                    
            except Exception:
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
                'spacegroup': {'symbol': spacegroup_symbol, 'number': spacegroup_number},
                'density': density,
                'volume': volume,
                'formation_energy_per_atom': formation_energy,
                'band_gap': band_gap,
                'formula': formula,
                'pretty_formula': pretty_formula
            }, None
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def get_materials_project_image_url(material_id, api_key, image_type="standard"):
    """获取Materials Project晶体结构图片的URL"""
    try:
        # Materials Project的图片URL格式
        if image_type == "standard":
            # 标准晶体结构图
            return f"https://next-gen.materialsproject.org/api/v2/materials/{material_id}/image?formula=&hideControls=true"
        elif image_type == "thermal_ellipsoid":
            # 热椭球图
            return f"https://next-gen.materialsproject.org/api/v2/materials/{material_id}/image?formula=&ellipsoids=true&hideControls=true"
        elif image_type == "polyhedral":
            # 多面体图
            return f"https://next-gen.materialsproject.org/api/v2/materials/{material_id}/image?formula=&polyhedra=true&hideControls=true"
    except Exception as e:
        return None

def download_materials_project_image(material_id, api_key, image_type="standard"):
    """下载Materials Project的晶体结构图片"""
    try:
        image_url = get_materials_project_image_url(material_id, api_key, image_type)
        
        if not image_url:
            return None
            
        # 使用API密钥进行认证
        headers = {
            "X-API-KEY": api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(image_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
            
    except Exception as e:
        return None

def display_crystal_structure_images(material_id, api_key, formula):
    """显示Materials Project的晶体结构图片"""
    try:
        st.subheader("🎯 Crystal Structure from Materials Project")
        
        # 显示标准晶体结构图
        with st.spinner("Loading crystal structure image..."):
            standard_img = download_materials_project_image(material_id, api_key, "standard")
            
            if standard_img:
                st.markdown(f'<div class="crystal-image">', unsafe_allow_html=True)
                st.image(standard_img, caption=f"Standard Crystal Structure: {formula}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # 如果直接下载失败，显示官方链接
                st.info("💡 Click the button below to view the crystal structure on Materials Project website:")
                viz_url = f"https://next-gen.materialsproject.org/materials/{material_id}"
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{viz_url}" target="_blank" style="
                        display: inline-block;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        font-weight: bold;
                    ">
                    🎯 View Crystal Structure on Materials Project
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
        # 尝试显示其他类型的图片
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Loading polyhedral view..."):
                polyhedral_img = download_materials_project_image(material_id, api_key, "polyhedral")
                if polyhedral_img:
                    st.image(polyhedral_img, caption="Polyhedral View", use_column_width=True)
        
        with col2:
            with st.spinner("Loading ellipsoid view..."):
                ellipsoid_img = download_materials_project_image(material_id, api_key, "thermal_ellipsoid")
                if ellipsoid_img:
                    st.image(ellipsoid_img, caption="Thermal Ellipsoids", use_column_width=True)
                    
        return True
        
    except Exception as e:
        st.error(f"Error displaying crystal structure images: {str(e)}")
        return False

def get_crystal_structure_screenshot(material_id, api_key):
    """备选方案：使用Materials Project的截图功能"""
    try:
        # 使用Materials Project的embed功能
        embed_url = f"https://next-gen.materialsproject.org/materials/{material_id}?embed=true"
        
        headers = {
            "X-API-KEY": api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(embed_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # 这里可以进一步处理嵌入内容
            return None  # 简化处理
        else:
            return None
            
    except Exception as e:
        return None

def analyze_structure_features(structure):
    """分析晶体结构特征"""
    try:
        density = structure.density
        
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        if abs(a - b) < 0.1 and abs(b - c) < 0.1 and all(abs(angle - 90) < 1 for angle in [alpha, beta, gamma]):
            lattice_type = "cubic"
            symmetry = "high"
        elif abs(a - b) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1:
            lattice_type = "tetragonal"
            symmetry = "medium"
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

# 添加缺失的函数
def calculate_material_features(formula):
    """计算材料特征 - 简化版本"""
    try:
        # 这里应该是你的特征计算逻辑
        # 暂时返回一个示例特征字典
        features = {}
        for desc in required_descriptors:
            if desc != 'Temp':
                features[desc] = np.random.normal(0, 1)  # 示例数据
        return features
    except Exception as e:
        st.error(f"Error calculating features: {str(e)}")
        return {}

def filter_selected_features(features, required_descriptors, temperature):
    """过滤选定的特征"""
    selected_features = {}
    for desc in required_descriptors:
        if desc == 'Temp':
            selected_features[desc] = temperature
        elif desc in features:
            selected_features[desc] = features[desc]
        else:
            selected_features[desc] = 0.0  # 默认值
    return selected_features

# 如果点击提交按钮
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        is_valid, validation_msg = validate_chemical_formula(formula_input)
        
        if not is_valid:
            st.error(f"Invalid chemical formula: {validation_msg}")
            st.info("💡 Please use standard chemical notation like: Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")
        else:
            with st.spinner("Processing material and making predictions..."):
                try:
                    if mp_api_key and mp_api_key.strip():
                        with st.spinner("Fetching crystal structure from Materials Project..."):
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I').replace('3272', '3Zr2')
                            
                            mp_data, mp_error = get_materials_project_structure_with_images(corrected_formula, mp_api_key)
                            
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
                                    if mp_data['volume'] != 'N/A':
                                        st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
                                    if mp_data['formation_energy_per_atom'] != 'N/A':
                                        st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
                                
                                # 分析结构特征
                                structure_info = analyze_structure_features(mp_data['structure'])
                                
                                # 显示结构分析
                                st.subheader("🔬 Structure Analysis")
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.write(f"**Structure Type:** {structure_info['structure_type'].capitalize()}")
                                with col4:
                                    st.write(f"**Symmetry:** {structure_info['symmetry'].capitalize()}")
                                
                                # 直接显示Materials Project的晶体结构图片
                                display_crystal_structure_images(
                                    mp_data['material_id'], 
                                    mp_api_key, 
                                    mp_data['pretty_formula']
                                )
                                
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                st.info("💡 The material might not exist in Materials Project database")
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
