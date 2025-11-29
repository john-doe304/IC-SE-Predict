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
import requests
import json
from pymatgen.core import Structure
import plotly.graph_objects as go
from crystal_toolkit.components.structure import StructureMoleculeComponent
import io

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

# 新增加：是否获取晶体结构
get_crystal_structure = st.checkbox("Get Crystal Structure (if available)", value=True)

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

def get_crystal_structure_from_materials_project(formula):
    """
    从Materials Project获取晶体结构信息
    注意：你需要注册Materials Project账号并获取API key
    """
    try:
        # 这里需要你的Materials Project API key
        # 你可以在 https://materialsproject.org/open 注册获取
        API_KEY = "your_materials_project_api_key_here"  # 替换为你的API key
        
        # 搜索材料
        base_url = "https://materialsproject.org/rest/v2"
        search_url = f"{base_url}/materials/summary/{formula}/?API_KEY={API_KEY}"
        
        response = requests.get(search_url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('response'):
                material = data['response'][0]
                
                # 获取结构信息
                structure_url = f"{base_url}/materials/{material['material_id']}/structure/?API_KEY={API_KEY}"
                structure_response = requests.get(structure_url)
                
                if structure_response.status_code == 200:
                    structure_data = structure_response.json()
                    return {
                        'success': True,
                        'material_id': material['material_id'],
                        'formula': material['full_formula'],
                        'space_group': material['spacegroup']['symbol'],
                        'crystal_system': material['spacegroup']['crystal_system'],
                        'volume': material['volume'],
                        'density': material['density'],
                        'structure': structure_data
                    }
        
        return {'success': False, 'error': 'Material not found in Materials Project'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_crystal_structure_from_cod(formula):
    """
    从Crystallography Open Database获取晶体结构
    """
    try:
        # COD的REST API端点
        cod_url = f"http://www.crystallography.net/cod/result.php?formula={formula}&format=json"
        
        response = requests.get(cod_url)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                # 获取第一个匹配的结构
                cod_id = data[0]['file']
                cif_url = f"http://www.crystallography.net/cod/{cod_id}"
                
                cif_response = requests.get(cif_url)
                if cif_response.status_code == 200:
                    return {
                        'success': True,
                        'source': 'COD',
                        'cod_id': cod_id,
                        'cif_data': cif_response.text,
                        'formula': data[0]['formula']
                    }
        
        return {'success': False, 'error': 'Material not found in COD'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def visualize_crystal_structure_plotly(structure_info):
    """
    使用plotly可视化晶体结构
    """
    try:
        if structure_info.get('source') == 'COD' and 'cif_data' in structure_info:
            # 从CIF数据创建结构
            from pymatgen.io.cif import CifParser
            from io import StringIO
            
            cif_parser = CifParser(StringIO(structure_info['cif_data']))
            structure = cif_parser.get_structures()[0]
        else:
            return None
        
        # 获取晶格参数
        lattice = structure.lattice
        
        # 创建3D散点图
        fig = go.Figure()
        
        # 添加原子
        for site in structure:
            fig.add_trace(go.Scatter3d(
                x=[site.coords[0]],
                y=[site.coords[1]],
                z=[site.coords[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',  # 可以根据元素类型设置不同颜色
                    opacity=0.8
                ),
                name=str(site.specie)
            ))
        
        # 添加晶格向量
        origin = [0, 0, 0]
        for i, vector in enumerate([lattice.matrix[0], lattice.matrix[1], lattice.matrix[2]]):
            fig.add_trace(go.Scatter3d(
                x=[origin[0], vector[0]],
                y=[origin[1], vector[1]],
                z=[origin[2], vector[2]],
                mode='lines',
                line=dict(color='red', width=4),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Crystal Structure: {structure_info.get('formula', 'Unknown')}",
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=600,
            height=500
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Crystal structure visualization failed: {str(e)}")
        return None

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

# 如果点击提交按钮
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
                # 如果用户选择获取晶体结构
                crystal_structure_info = None
                if get_crystal_structure:
                    with st.spinner("Searching for crystal structure..."):
                        # 首先尝试从COD获取
                        crystal_structure_info = get_crystal_structure_from_cod(formula_input)
                        
                        if not crystal_structure_info.get('success'):
                            st.info("Crystal structure not found in open databases. Using composition features only.")
                        else:
                            st.success("Crystal structure found!")
                
                # 计算材料特征
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")
                
                # 只显示选定的七个特征
                selected_features = filter_selected_features(features, required_descriptors, temperature)
                feature_df = pd.DataFrame([selected_features])
                
                st.subheader("Material Features")
                st.dataframe(feature_df)
                
                # 显示晶体结构信息
                if crystal_structure_info and crystal_structure_info.get('success'):
                    st.subheader("Crystal Structure Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Structure Details:**")
                        st.write(f"Source: {crystal_structure_info.get('source', 'Unknown')}")
                        if crystal_structure_info.get('formula'):
                            st.write(f"Formula: {crystal_structure_info['formula']}")
                        if crystal_structure_info.get('cod_id'):
                            st.write(f"COD ID: {crystal_structure_info['cod_id']}")
                    
                    with col2:
                        # 可视化晶体结构
                        fig = visualize_crystal_structure_plotly(crystal_structure_info)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 提供CIF文件下载
                    if crystal_structure_info.get('cif_data'):
                        st.download_button(
                            label="Download CIF File",
                            data=crystal_structure_info['cif_data'],
                            file_name=f"{formula_input.replace(' ', '_')}.cif",
                            mime="chemical/x-cif"
                        )
            
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
