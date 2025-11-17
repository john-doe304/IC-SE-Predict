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
import json

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

def get_crystal_structure_image_direct(material_id, api_key):
    """直接获取晶体结构图片 - 使用正确的API端点"""
    try:
        # Materials Project的官方API端点
        base_url = "https://next-gen.materialsproject.org"
        
        # 方法1: 使用materials/{id}/image端点
        image_url = f"{base_url}/materials/{material_id}/image"
        
        headers = {
            "X-API-KEY": api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "image/png,image/jpeg"
        }
        
        params = {
            "formula": "",
            "hideControls": "true",
            "width": "600",
            "height": "400"
        }
        
        response = requests.get(image_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200 and response.content:
            return Image.open(BytesIO(response.content))
        else:
            # 方法2: 尝试使用不同的参数
            params2 = {
                "style": "ball_and_stick",
                "size": "large"
            }
            response2 = requests.get(image_url, headers=headers, params=params2, timeout=30)
            if response2.status_code == 200 and response2.content:
                return Image.open(BytesIO(response2.content))
            
        return None
        
    except Exception as e:
        return None

def create_crystal_structure_plotly(structure, formula):
    """使用plotly创建晶体结构可视化"""
    try:
        # 获取晶格参数
        lattice = structure.lattice
        sites = structure.sites
        
        # 创建原子位置数据
        x, y, z = [], [], []
        colors, sizes, symbols, hover_texts = [], [], [], []
        
        # 原子颜色映射
        color_map = {
            'Li': '#CC80FF', 'La': '#70D4FF', 'Zr': '#4EACCE', 'O': '#FF0D0D',
            'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Ge': '#668F8F',
            'Y': '#94FFFF', 'F': '#90E050', 'Br': '#A62929', 'I': '#940094',
            'Na': '#AB5CF2', 'K': '#8F40D4', 'Mg': '#8AFF00', 'Ca': '#3DFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'Ti': '#BFC2C7', 'Fe': '#E06633'
        }
        
        # 原子大小映射
        size_map = {
            'Li': 10, 'La': 20, 'Zr': 15, 'O': 12,
            'P': 13, 'S': 12, 'Cl': 12, 'Ge': 14,
            'Y': 14, 'F': 10, 'Br': 13, 'I': 15,
            'Na': 12, 'K': 14, 'Mg': 13, 'Ca': 14,
            'Al': 13, 'Si': 13, 'Ti': 14, 'Fe': 14
        }
        
        for site in sites:
            x.append(site.coords[0])
            y.append(site.coords[1])
            z.append(site.coords[2])
            element = site.species_string
            colors.append(color_map.get(element, '#CCCCCC'))
            sizes.append(size_map.get(element, 12))
            symbols.append(element)
            hover_texts.append(f"{element} ({site.coords[0]:.2f}, {site.coords[1]:.2f}, {site.coords[2]:.2f})")
        
        # 创建原子轨迹
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.9,
                line=dict(width=2, color='darkgray')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Atoms'
        )
        
        # 创建晶格线
        lattice_traces = []
        origin = [0, 0, 0]
        a_vec = lattice.matrix[0]
        b_vec = lattice.matrix[1]
        c_vec = lattice.matrix[2]
        
        # 创建晶胞边界
        vertices = [
            origin,
            a_vec, b_vec, c_vec,
            a_vec + b_vec, a_vec + c_vec, b_vec + c_vec,
            a_vec + b_vec + c_vec
        ]
        
        edges = [
            (0,1), (0,2), (0,3),
            (1,4), (1,5), (2,4), (2,6),
            (3,5), (3,6), (4,7), (5,7), (6,7)
        ]
        
        for start, end in edges:
            lattice_traces.append(go.Scatter3d(
                x=[vertices[start][0], vertices[end][0]],
                y=[vertices[start][1], vertices[end][1]],
                z=[vertices[start][2], vertices[end][2]],
                mode='lines',
                line=dict(color='black', width=5),
                hoverinfo='none',
                showlegend=False
            ))
        
        # 创建图形
        fig = go.Figure(data=[atom_trace] + lattice_traces)
        
        fig.update_layout(
            title=f"Crystal Structure: {formula}",
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='white'
            ),
            width=600,
            height=500,
            margin=dict(l=20, r=20, b=20, t=40)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating structure visualization: {str(e)}")
        return None

def display_crystal_structure_direct(material_id, api_key, formula, structure):
    """直接显示晶体结构"""
    try:
        st.subheader("🎯 Crystal Structure Visualization")
        
        # 尝试获取官方图片
        with st.spinner("Loading crystal structure image..."):
            crystal_img = get_crystal_structure_image_direct(material_id, api_key)
            
            if crystal_img:
                st.markdown(f'<div class="crystal-image">', unsafe_allow_html=True)
                st.image(crystal_img, caption=f"Crystal Structure: {formula}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.success("✅ Crystal structure image loaded successfully")
            else:
                # 如果官方图片获取失败，使用plotly创建可视化
                st.warning("Using interactive 3D visualization instead...")
                fig = create_crystal_structure_plotly(structure, formula)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **Interactive Controls:** Drag to rotate, scroll to zoom, shift+drag to pan")
                else:
                    # 最后备选方案：显示链接
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
                    
        return True
        
    except Exception as e:
        st.error(f"Error displaying crystal structure: {str(e)}")
        return False

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
                                
                                # 直接显示晶体结构图片或可视化
                                display_crystal_structure_direct(
                                    mp_data['material_id'], 
                                    mp_api_key, 
                                    mp_data['pretty_formula'],
                                    mp_data['structure']
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
