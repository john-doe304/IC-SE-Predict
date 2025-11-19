import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import requests
from PIL import Image
import base64
import tempfile
import gc
import sys
import os

# 禁用有问题的库，使用替代方案
_3D_AVAILABLE = False

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
    .crystal-structure-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
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
             2. Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:", placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")

# 温度输入
temperature = st.number_input("Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10)

# Materials Project API 密钥输入
mp_api_key = st.text_input("Materials Project API Key (optional):", 
                          placeholder="Enter your API key to view crystal structure",
                          type="password")

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
    """缓存模型加载"""
    try:
        from autogluon.tabular import TabularPredictor
        return TabularPredictor.load("./ag-20251024_075719")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

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

def get_materials_project_structure(formula, api_key):
    """获取晶体结构信息 - 简化版本"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        from pymatgen.ext.matproj import MPRester
        
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format"
        
        with MPRester(api_key) as mpr:
            try:
                # 搜索材料
                results = mpr.summary.search(formula=formula, fields=[
                    "material_id", "formula_pretty", "spacegroup", 
                    "density", "volume", "formation_energy_per_atom"
                ])
                
                if not results:
                    return None, f"No materials found for formula: {formula}"
                
                material = results[0]
                
                return {
                    'material_id': material.material_id,
                    'pretty_formula': material.formula_pretty,
                    'spacegroup': {
                        'symbol': material.spacegroup.symbol if material.spacegroup else "N/A",
                        'number': material.spacegroup.number if material.spacegroup else "N/A"
                    },
                    'density': material.density,
                    'volume': material.volume,
                    'formation_energy_per_atom': material.formation_energy_per_atom,
                    'formula': formula
                }, None
                
            except Exception as e:
                return None, f"Error retrieving material: {str(e)}"
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def display_structure_info(mp_data):
    """显示结构信息"""
    st.markdown("### Crystal Structure Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Material ID:** `{mp_data['material_id']}`")
        st.write(f"**Formula:** {mp_data['pretty_formula']}")
        st.write(f"**Space Group:** {mp_data['spacegroup']['symbol']} ({mp_data['spacegroup']['number']})")
        
    with col2:
        st.write(f"**Density:** {mp_data['density']:.2f} g/cm³")
        st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
        st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
    
    # 显示晶格参数可视化
    st.markdown("### Crystal Structure Visualization")
    
    # 创建结构示意图
    fig = go.Figure()
    
    # 添加晶体结构示意图
    fig.add_annotation(
        text=f"<b>{mp_data['pretty_formula']}</b><br>"
             f"Space Group: {mp_data['spacegroup']['symbol']}<br>"
             f"Density: {mp_data['density']:.2f} g/cm³<br>"
             f"Volume: {mp_data['volume']:.2f} Å³<br><br>"
             f"<i>View interactive 3D structure on Materials Project</i>",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="white",
        bordercolor="blue",
        borderwidth=2,
        borderpad=10,
        align="center"
    )
    
    fig.update_layout(
        title="Crystal Structure Information",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='lightblue'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加外部链接
    st.markdown("### View Interactive 3D Structure")
    material_id = mp_data['material_id']
    clean_material_id = material_id.split('-')[0] if '-' in material_id else material_id
    mp_url = f"https://next-gen.materialsproject.org/materials/{clean_material_id}"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <a href="{mp_url}" target="_blank" style="
            display: inline-block;
            padding: 12px 24px;
            background-color: #1976d2;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1em;
        ">
        🔍 View Interactive 3D Structure on Materials Project
        </a>
    </div>
    """, unsafe_allow_html=True)

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
                            mp_data, mp_error = get_materials_project_structure(formula_input, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                display_structure_info(mp_data)
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                st.info("💡 The material might not exist in Materials Project database")
                    else:
                        st.info("💡 Enter a Materials Project API key to view crystal structure information")
                    
                    # 计算材料特征
                    with st.spinner("Calculating material features..."):
                        features = calculate_material_features(formula_input)
                    
                    st.write(f"✅ Features prepared for prediction")
                    
                    # 显示选定的特征
                    selected_features = filter_selected_features(features, required_descriptors, temperature)
                    feature_df = pd.DataFrame([selected_features])
                    
                    st.subheader("Material Features")
                    st.dataframe(feature_df)
                
                    # 创建输入数据
                    input_data = {feature: [value] for feature, value in selected_features.items()}
                    input_data['Formula'] = [formula_input]
                    input_df = pd.DataFrame(input_data)
                  
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
                                        
                       results_data = []
                       for model, prediction in  essential_models.items():
                           results_data.append({
                                'Model': model,
                                'Predicted Ionic Conductivity (S/cm)': prediction
                            })
                            
                       results_df = pd.DataFrame(results_data)
                       st.dataframe(results_df)     
               
                       else:
                           st.error("Model not available in current environment")
                            
                   except Exception as e:
                       st.error(f"Prediction failed: {str(e)}")
                        
                    
               except Exception as e:
                   st.error(f"An error occurred: {str(e)}")










