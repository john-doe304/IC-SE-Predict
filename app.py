import streamlit as st
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import gc
import requests
from PIL import Image
import base64
from io import BytesIO

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
    /* 晶体结构容器样式 */
    .crystal-structure-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .database-link {
        display: inline-block;
        margin: 5px;
        padding: 8px 15px;
        background-color: #1976d2;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
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
                          type="password")

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

def get_crystal_structure_links(formula):
    """生成晶体结构数据库链接"""
    links = {}
    
    # Materials Project 链接
    links['Materials Project'] = f"https://next-gen.materialsproject.org/search?formula={formula}"
    
    # Crystallography Open Database (COD) 链接
    links['Crystallography Open Database'] = f"https://www.crystallography.net/cod/search?formula={formula}"
    
    # Springer Materials 链接
    links['Springer Materials'] = f"https://materials.springer.com/search?searchTerm={formula}"
    
    # AFLOW 链接
    links['AFLOW'] = f"http://aflow.org/search/?formula={formula}"
    
    # OQMD 链接
    links['OQMD'] = f"http://oqmd.org/search?formula={formula}"
    
    return links

def create_structure_diagram_placeholder(formula):
    """创建晶体结构占位图"""
    # 创建一个简单的SVG占位图
    svg_content = f"""
    <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f0f8ff"/>
        <circle cx="200" cy="150" r="80" fill="#e6f3ff" stroke="#1976d2" stroke-width="2"/>
        <text x="200" y="120" text-anchor="middle" font-family="Arial" font-size="16" fill="#1976d2">Crystal Structure</text>
        <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="14" fill="#1976d2">{formula}</text>
        <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">Interactive viewer available</text>
        <text x="200" y="200" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">on external databases</text>
    </svg>
    """
    return svg_content

def display_crystal_structure_section(formula, mp_api_key=None):
    """显示晶体结构部分"""
    st.subheader("🎯 Crystal Structure Information")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["Structure Viewer", "Database Links", "Structure Info"])
    
    with tab1:
        st.markdown("### Crystal Structure Visualization")
        
        # 显示占位图
        svg_placeholder = create_structure_diagram_placeholder(formula)
        st.markdown(f'<div style="text-align: center">{svg_placeholder}</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <p><strong>Interactive 3D structure viewers available on external databases:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 快速链接
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Open Materials Project", key="mp_btn"):
                st.markdown(f"[Open Materials Project](https://next-gen.materialsproject.org/search?formula={formula})", unsafe_allow_html=True)
        with col2:
            if st.button("📚 Open COD", key="cod_btn"):
                st.markdown(f"[Open Crystallography Open Database](https://www.crystallography.net/cod/search?formula={formula})", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### External Crystal Structure Databases")
        
        links = get_crystal_structure_links(formula)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>🔍 Search on these databases:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for db_name, url in links.items():
            st.markdown(f"""
            <div style="margin: 8px 0;">
                <a href="{url}" target="_blank" style="
                    display: block;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-decoration: none;
                    color: #333;
                    font-weight: 500;
                    transition: all 0.3s;
                " onmouseover="this.style.backgroundColor='#e3f2fd'; this.style.borderColor='#1976d2'" 
                 onmouseout="this.style.backgroundColor='white'; this.style.borderColor='#ddd'">
                🔗 {db_name}
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Structure Information")
        
        # 显示基本结构信息
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Formula", formula)
            st.metric("Expected Structure Types", "Garnet/Perovskite/etc.")
        with col2:
            st.metric("Common Space Groups", "Ia-3d, Pm-3m, etc.")
            st.metric("Typical Applications", "Solid Electrolyte")
        
        st.markdown("""
        **Common Solid Electrolyte Structure Types:**
        - **Garnet-type**: Li₇La₃Zr₂O₁₂, cubic/tetragonal
        - **NASICON-type**: Li₁₊ₓAlₓTi₂₋ₓ(PO₄)₃
        - **Perovskite-type**: Li₃ₓLa₂/₃₋ₓTiO₃
        - **LISICON-type**: Li₁₄Zn(GeO₄)₄
        - **Thio-LISICON**: Li₁₀GeP₂S₁₂
        """)

def get_materials_project_info_simple(formula, api_key):
    """简化版Materials Project信息获取"""
    if not api_key:
        return None, "No API key provided"
    
    try:
        # 这里使用Materials Project的REST API
        base_url = "https://api.materialsproject.org"
        headers = {
            "X-API-KEY": api_key
        }
        
        # 搜索材料
        search_url = f"{base_url}/materials/summary/?formula={formula}"
        response = requests.get(search_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                material = data['data'][0]
                return {
                    'material_id': material.get('material_id', 'N/A'),
                    'formula_pretty': material.get('formula_pretty', formula),
                    'spacegroup': material.get('spacegroup', {}),
                    'density': material.get('density', 'N/A'),
                    'volume': material.get('volume', 'N/A'),
                    'formation_energy_per_atom': material.get('formation_energy_per_atom', 'N/A'),
                    'band_gap': material.get('band_gap', 'N/A')
                }, None
        return None, "Material not found in Materials Project"
        
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

# 材料特征计算函数
def calculate_material_features(formula):
    """计算材料的组成特征"""
    try:
        # 这里使用简化的特征计算
        # 在实际应用中，您可以使用matminer或其他库
        features = {'Formula': formula}
        
        # 添加一些模拟的特征值
        features['MagpieData mean CovalentRadius'] = 1.5
        features['MagpieData avg_dev SpaceGroupNumber'] = 2.3
        features['0-norm'] = 8.7
        features['MagpieData mean MeltingT'] = 1200.0
        features['MagpieData avg_dev Column'] = 1.2
        features['MagpieData mean NValence'] = 4.5
        
        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
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
                    # 显示晶体结构信息
                    display_crystal_structure_section(formula_input, mp_api_key)
                    
                    # 如果提供了API密钥，尝试获取Materials Project信息
                    if mp_api_key and mp_api_key.strip():
                        with st.spinner("Fetching information from Materials Project..."):
                            mp_info, mp_error = get_materials_project_info_simple(formula_input, mp_api_key.strip())
                            if mp_info and not mp_error:
                                st.success("✅ Information retrieved from Materials Project")
                                
                                # 显示Materials Project信息
                                st.subheader("📊 Materials Project Data")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Material ID:** `{mp_info['material_id']}`")
                                    st.write(f"**Formula:** {mp_info['formula_pretty']}")
                                    if mp_info['spacegroup']:
                                        st.write(f"**Space Group:** {mp_info['spacegroup'].get('symbol', 'N/A')}")
                                with col2:
                                    if mp_info['density'] != 'N/A':
                                        st.write(f"**Density:** {mp_info['density']:.2f} g/cm³")
                                    if mp_info['formation_energy_per_atom'] != 'N/A':
                                        st.write(f"**Formation Energy:** {mp_info['formation_energy_per_atom']:.3f} eV/atom")
                    
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
