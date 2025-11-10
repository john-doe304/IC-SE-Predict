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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 39%; /* 设置最大宽度 */
        background-color: #f9f9f9f9;
        padding: 20px; /* 增加内边距 */
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
    a {
        color: #0000EE;
        text-decoration: underline;
    }
    .process-text, .molecular-weight {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 0px !important;
    }
    .molecule-container {
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: transparent; /* 透明背景 */
    }
    .crystal-structure-info {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .crystal-visualization {
        background-color: #fff8f0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF6B00;
    }
    .prediction-results {
        background-color: #f8fff0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF6B00;
    }
     /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px; /* 减少内边距 */
        }
        .rounded-container blockquote {
            font-size: 0.9em; /* 缩小字体 */
        }
        .rounded-container h2 {
            font-size: 1.2em; /* 调整标题字体大小 */
        }
        .stApp {
            padding: 1px !important; /* 减少内边距 */
            max-width: 99%; /* 设置最大宽度 */
        }
        .process-text, .molecular-weight {
            font-size: 0.9em; /* 缩小文本字体 */
        }
        .molecule-container {
            max-width: 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2> Predict Ionic Conductivity(Cond) of Solid Electrolytes</h2>
        <blockquote>
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
            2. Supports various solid electrolyte materials including oxides, sulfides, and halides.<br>
            3. Code and data available at <a href='https://github.com/john-doe304/IC-SE-Predict' target='_blank'>GitHub Repository</a>.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 材料体系选择
material_systems = {
    "LLZO": {"Type": "Garnet Oxide", "Typical Composition": "Li7La3Zr2O12", "Temperature Range": "25-500°C"},
    "LGPS": {"Type": "Crystalline Sulfide", "Typical Composition": "Li10GeP2S12", "Temperature Range": "25-300°C"},
    "NASICON": {"Type": "NASICON Oxide", "Typical Composition": "Li1+xAlxTi2-x(PO4)3", "Temperature Range": "25-400°C"},
    "Perovskite": {"Type": "Perovskite Oxide", "Typical Composition": "Li3xLa2/3-xTiO3", "Temperature Range": "25-600°C"},
    "Anti-Perovskite": {"Type": "Anti-Perovskite Halide", "Typical Composition": "Li3OCl", "Temperature Range": "25-300°C"},
    "Sulfide Glass": {"Type": "Amorphous Sulfide", "Typical Composition": "Li2S-P2S5", "Temperature Range": "25-200°C"},
    "Polymer": {"Type": "Polymer Electrolyte", "Typical Composition": "PEO-LiTFSI", "Temperature Range": "40-100°C"},
    "Halide": {"Type": "Halide Electrolyte", "Typical Composition": "Li3YCl6", "Temperature Range": "25-300°C"}
}

# 材料体系选择下拉菜单
material_system = st.selectbox("Select Material Type:", list(material_systems.keys()))

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

def create_crystal_structure_visualization(crystal_system, lattice_params, formula):
    """
    创建晶体结构可视化
    """
    fig = go.Figure()
    
    # 根据晶体系统设置不同的可视化
    if "Cubic" in crystal_system:
        # 立方晶系
        x = [0, 1, 1, 0, 0, 1, 1, 0]
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        z = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # 绘制立方体边
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面
            [0,4], [1,5], [2,6], [3,7]   # 侧面
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=False
            ))
        
        # 添加原子位置
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Atoms'
        ))
        
    elif "Tetragonal" in crystal_system:
        # 四方晶系
        a, c = 1.0, 1.5  # 不同的a和c参数
        x = [0, a, a, 0, 0, a, a, 0]
        y = [0, 0, a, a, 0, 0, a, a]
        z = [0, 0, 0, 0, c, c, c, c]
        
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='green', width=4),
                showlegend=False
            ))
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='orange'),
            name='Atoms'
        ))
        
    elif "Trigonal" in crystal_system or "Rhombohedral" in crystal_system:
        # 三角/菱方晶系
        import math
        angles = [0, 2*math.pi/3, 4*math.pi/3]
        x = [math.cos(angle) for angle in angles] + [math.cos(angle) for angle in angles]
        y = [math.sin(angle) for angle in angles] + [math.sin(angle) for angle in angles]
        z = [0,0,0,1,1,1]
        
        # 绘制三角棱柱
        edges = [
            [0,1], [1,2], [2,0],  # 底面三角形
            [3,4], [4,5], [5,3],  # 顶面三角形
            [0,3], [1,4], [2,5]   # 侧面
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='purple', width=4),
                showlegend=False
            ))
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='magenta'),
            name='Atoms'
        ))
        
    else:
        # 默认立方晶系
        x = [0, 1, 1, 0, 0, 1, 1, 0]
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        z = [0, 0, 0, 0, 1, 1, 1, 1]
        
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='gray', width=4),
                showlegend=False
            ))
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Atoms'
        ))
    
    fig.update_layout(
        title=f"Crystal Structure: {crystal_system} - {formula}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=500,
        height=400,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_unit_cell_diagram(crystal_system, lattice_params):
    """
    创建晶胞示意图
    """
    fig = go.Figure()
    
    # 解析晶格参数
    a_match = re.search(r'a\s*=\s*([\d.]+)', lattice_params)
    c_match = re.search(r'c\s*=\s*([\d.]+)', lattice_params)
    
    a_val = float(a_match.group(1)) if a_match else 1.0
    c_val = float(c_match.group(1)) if c_match else (1.5 if "Tetragonal" in crystal_system or "Trigonal" in crystal_system else 1.0)
    
    # 根据晶体系统绘制不同的晶胞
    if "Cubic" in crystal_system:
        # 立方晶胞
        fig.add_trace(go.Mesh3d(
            x=[0, a_val, a_val, 0, 0, a_val, a_val, 0],
            y=[0, 0, a_val, a_val, 0, 0, a_val, a_val],
            z=[0, 0, 0, 0, a_val, a_val, a_val, a_val],
            i=[0, 0, 0, 2],
            j=[1, 2, 3, 3],
            k=[2, 3, 7, 7],
            opacity=0.3,
            color='lightblue'
        ))
        
    elif "Tetragonal" in crystal_system:
        # 四方晶胞
        fig.add_trace(go.Mesh3d(
            x=[0, a_val, a_val, 0, 0, a_val, a_val, 0],
            y=[0, 0, a_val, a_val, 0, 0, a_val, a_val],
            z=[0, 0, 0, 0, c_val, c_val, c_val, c_val],
            i=[0, 0, 0, 2],
            j=[1, 2, 3, 3],
            k=[2, 3, 7, 7],
            opacity=0.3,
            color='lightgreen'
        ))
        
    elif "Trigonal" in crystal_system:
        # 三角晶胞
        import math
        # 简化的三角晶胞表示
        fig.add_trace(go.Mesh3d(
            x=[0, a_val, a_val/2, 0, a_val, a_val/2],
            y=[0, 0, a_val*math.sqrt(3)/2, 0, 0, a_val*math.sqrt(3)/2],
            z=[0, 0, 0, c_val, c_val, c_val],
            i=[0, 0, 1],
            j=[1, 2, 2],
            k=[2, 4, 5],
            opacity=0.3,
            color='lavender'
        ))
    
    # 添加晶胞边界
    fig.update_layout(
        title=f"Unit Cell - {crystal_system}",
        scene=dict(
            xaxis_title='a (Å)',
            yaxis_title='b (Å)',
            zaxis_title='c (Å)',
            aspectmode='data'
        ),
        width=400,
        height=300
    )
    
    return fig

# 晶体结构数据库
crystal_structures = {
    "Li7La3Zr2O12": {
        "crystal_system": "Cubic",
        "space_group": "Ia-3d",
        "lattice_parameters": "a = 12.97 Å",
        "density": "5.08 g/cm³",
        "reference": "Murugan et al., Angew. Chem. Int. Ed. (2007)",
        "color": "#FF6B6B"
    },
    "Li10GeP2S12": {
        "crystal_system": "Tetragonal", 
        "space_group": "P4_2/nmc",
        "lattice_parameters": "a = 8.72 Å, c = 12.54 Å",
        "density": "2.04 g/cm³",
        "reference": "Kamaya et al., Nat. Mater. (2011)",
        "color": "#4ECDC4"
    },
    "Li3YCl6": {
        "crystal_system": "Trigonal",
        "space_group": "R-3m", 
        "lattice_parameters": "a = 6.62 Å, c = 18.24 Å",
        "density": "2.67 g/cm³",
        "reference": "Asano et al., Adv. Mater. (2018)",
        "color": "#45B7D1"
    },
    "Li3OCl": {
        "crystal_system": "Cubic",
        "space_group": "Pm-3m",
        "lattice_parameters": "a = 3.92 Å",
        "density": "2.41 g/cm³", 
        "reference": "Zhao et al., Nat. Commun. (2016)",
        "color": "#96CEB4"
    },
    "Li1+xAlxTi2-x(PO4)3": {
        "crystal_system": "Rhombohedral",
        "space_group": "R-3c",
        "lattice_parameters": "a = 8.51 Å, c = 20.84 Å",
        "density": "2.94 g/cm³",
        "reference": "Aono et al., J. Electrochem. Soc. (1990)",
        "color": "#FECA57"
    }
}

def get_crystal_structure_info(formula):
    """获取晶体结构信息"""
    # 直接匹配
    if formula in crystal_structures:
        return crystal_structures[formula]
    
    # 模糊匹配（包含关系）
    for key in crystal_structures:
        if formula in key or key in formula:
            return crystal_structures[key]
    
    # 根据材料类型推断
    if "Li" in formula and ("La" in formula or "Zr" in formula):
        return {
            "crystal_system": "Cubic/Tetragonal",
            "space_group": "Ia-3d/P4_2/nmc",
            "lattice_parameters": "~12.9-13.0 Å",
            "density": "~4.5-5.5 g/cm³",
            "reference": "Typical Garnet Structure",
            "color": "#FF9FF3"
        }
    elif "Li" in formula and ("S" in formula or "P" in formula):
        return {
            "crystal_system": "Tetragonal/Orthorhombic", 
            "space_group": "P4_2/nmc/Pnma",
            "lattice_parameters": "a~8.7 Å, c~12.5 Å",
            "density": "~2.0-2.5 g/cm³",
            "reference": "Typical Sulfide Structure",
            "color": "#54A0FF"
        }
    elif "Li" in formula and ("Cl" in formula or "Br" in formula or "I" in formula):
        return {
            "crystal_system": "Trigonal/Hexagonal",
            "space_group": "R-3m/P6_3/mmc", 
            "lattice_parameters": "a~6.6 Å, c~18.2 Å",
            "density": "~2.5-3.0 g/cm³",
            "reference": "Typical Halide Structure",
            "color": "#00D2D3"
        }
    else:
        return {
            "crystal_system": "Unknown",
            "space_group": "Unknown", 
            "lattice_parameters": "Unknown",
            "density": "Unknown",
            "reference": "Structure data not available",
            "color": "#C8D6E5"
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
    filtered_features['Temperature_K'] = temperature
    filtered_features['Temp'] = temperature
    
    # 添加选定的七个特征
    for feature_name in selected_descriptors:
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            # 如果特征不存在，设为0
            filtered_features[feature_name] = 0.0
    
    return filtered_features

# 自动匹配模型特征
def align_features_with_model(features_dict, predictor, temperature, formula, material_system):
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
        elif f_low in ['material_type']:
            aligned[feat] = material_system
        else:
            aligned[feat] = 0.0

    return pd.DataFrame([aligned])

def preprocess_material_data(formula, material_system, temperature, crystal_info):
    """
    预处理材料数据，确保晶体结构信息完整和温度有效
    """
    processed = {
        'formula': formula,
        'material_type': material_system,
        'temperature': temperature,
        'crystal_info': crystal_info
    }
    
    # 验证和设置温度
    if temperature == 0:
        processed['temperature'] = 298
        st.warning("警告：温度值为0，已使用默认值298K")
    
    # 确保晶体结构信息完整
    if not crystal_info:
        processed['crystal_info'] = get_crystal_structure_info(formula)
    
    return processed

def format_prediction_output(prediction_results, crystal_info, temperature, formula, material_system):
    """
    格式化预测输出，确保晶体结构信息清晰显示
    """
    output_lines = []
    
    # 标题
    output_lines.append("=" * 60)
    output_lines.append("           MATERIAL PROPERTY PREDICTION RESULTS")
    output_lines.append("=" * 60)
    
    # 晶体结构信息部分
    output_lines.append("\n📐 CRYSTAL STRUCTURE INFORMATION")
    output_lines.append("-" * 40)
    output_lines.append(f"Material: {formula}")
    output_lines.append(f"Type: {material_system}")
    output_lines.append(f"Crystal System: {crystal_info.get('crystal_system', 'N/A')}")
    output_lines.append(f"Space Group: {crystal_info.get('space_group', 'N/A')}")
    output_lines.append(f"Lattice Parameters: {crystal_info.get('lattice_parameters', 'N/A')}")
    output_lines.append(f"Density: {crystal_info.get('density', 'N/A')}")
    output_lines.append(f"Reference: {crystal_info.get('reference', 'N/A')}")
    
    # 实验条件
    output_lines.append("\n🌡️ EXPERIMENTAL CONDITIONS")
    output_lines.append("-" * 40)
    output_lines.append(f"Temperature: {temperature} K")
    
    # 预测结果
    if prediction_results and len(prediction_results) > 0:
        output_lines.append("\n📊 PREDICTION RESULTS")
        output_lines.append("-" * 40)
        
        # 显示每个模型的预测结果
        for model_name, prediction in prediction_results.items():
            if model_name != "status" and prediction != "Error":
                output_lines.append(f"{model_name}: {prediction:.6f} S/cm")
    
    output_lines.append("\n" + "=" * 60)
    
    return "\n".join(output_lines)

# 如果点击提交按钮
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
                # 显示材料信息
                material_info = material_systems[material_system]
                    
                col1, col2, col3 = st.columns(3)
                col1.metric("Material Type", material_system)
                col2.metric("Crystal Structure", material_info["Type"])
                col3.metric("Temperature", f"{temperature} K")
                
                # 获取晶体结构信息
                crystal_info = get_crystal_structure_info(formula_input)
                
                # 预处理数据（包含温度验证）
                processed_data = preprocess_material_data(
                    formula_input, material_system, temperature, crystal_info
                )
                
                # 使用处理后的温度
                actual_temperature = processed_data['temperature']
                if temperature != actual_temperature:
                    st.info(f"Temperature adjusted from {temperature}K to {actual_temperature}K for prediction")
                
                # 显示晶体结构信息
                st.subheader("📐 Crystal Structure Information")
                with st.container():
                    st.markdown(f"""
                    <div class='crystal-structure-info'>
                    <h4>Crystal Structure Details for {formula_input}</h4>
                    <p><strong>Crystal System:</strong> {crystal_info['crystal_system']}</p>
                    <p><strong>Space Group:</strong> {crystal_info['space_group']}</p>
                    <p><strong>Lattice Parameters:</strong> {crystal_info['lattice_parameters']}</p>
                    <p><strong>Density:</strong> {crystal_info['density']}</p>
                    <p><strong>Reference:</strong> <em>{crystal_info['reference']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 显示晶体结构可视化
                st.subheader("🔬 Crystal Structure Visualization")
                with st.container():
                    st.markdown(f"""
                    <div class='crystal-visualization'>
                    <h4>3D Crystal Structure Model</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 创建晶体结构可视化
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 3D晶体结构图
                        crystal_fig = create_crystal_structure_visualization(
                            crystal_info['crystal_system'],
                            crystal_info['lattice_parameters'],
                            formula_input
                        )
                        st.plotly_chart(crystal_fig, use_container_width=True)
                    
                    with col2:
                        # 晶胞示意图
                        unit_cell_fig = create_unit_cell_diagram(
                            crystal_info['crystal_system'],
                            crystal_info['lattice_parameters']
                        )
                        st.plotly_chart(unit_cell_fig, use_container_width=True)
                        
                # 显示晶体系统说明
                st.info(f"""
                **Crystal System Explanation:** 
                - **{crystal_info['crystal_system']}** crystal system
                - Space group: **{crystal_info['space_group']}**
                - Characterized by: {crystal_info['lattice_parameters']}
                """)
                        
                # 计算材料特征
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")
                
                # 只显示选定的七个特征（使用实际温度）
                selected_features = filter_selected_features(features, required_descriptors, actual_temperature)
                feature_df = pd.DataFrame([selected_features])
                
                st.subheader("Selected Material Features")
                st.dataframe(feature_df)
            
                if features:
                    # 创建输入数据（使用实际温度）
                    input_data = {
                        "Formula": [formula_input],
                        "Material_Type": [material_system],
                        "Temperature_K": [actual_temperature],
                    }
                    
                    # 添加数值特征
                    numeric_features = {}
                    for feature_name in required_descriptors:
                        if feature_name in features:
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
                            predictions_dict[model] = predictions.iloc[0] if hasattr(predictions, 'iloc') else predictions[0]
                        except Exception as model_error:
                            st.warning(f"Model {model} prediction failed: {str(model_error)}")
                            predictions_dict[model] = "Error"

                    # 显示预测结果
                    st.subheader("🎯 Prediction Results")
                    st.markdown(
                        "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    
                    # 创建预测结果表格
                    results_data = []
                    for model_name, prediction in predictions_dict.items():
                        if prediction != "Error":
                            results_data.append({
                                "Model": model_name,
                                "Ionic Conductivity (S/cm)": f"{prediction:.6f}"
                            })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df)
                    
                    # 显示格式化的完整输出
                    st.subheader("📋 Complete Prediction Report")
                    formatted_output = format_prediction_output(
                        predictions_dict, crystal_info, actual_temperature, formula_input, material_system
                    )
                    st.markdown(f"```\n{formatted_output}\n```")
                    
                    # 主动释放内存
                    del predictor
                    gc.collect()

                except Exception as e:
                    st.error(f"Model loading failed: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
