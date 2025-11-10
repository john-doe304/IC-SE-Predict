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
from pymatgen.core import Composition
from pymatgen.ext.matproj import MPRester
import plotly.graph_objects as go
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
    /* 晶体结构显示样式 */
    .crystal-structure {
        margin: 20px 0;
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
@st.cache_resource(show_spinner=False, max_entries=1)  # 限制只缓存一个实例
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

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

def get_materials_project_structure(formula, api_key):
    """从Materials Project获取晶体结构信息"""
    if not api_key:
        return None, "No API key provided"
    
    try:
        with MPRester(Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN) as mpr:
            # 搜索材料
            materials = mpr.get_entries(formula)
            
            if not materials:
                return None, f"No materials found for formula: {formula}"
            
            # 获取第一个材料的完整信息
            material_id = materials[0].entry_id
            structure = mpr.get_structure_by_material_id(material_id.split("-")[1])
            
            # 获取材料详情
            material_data = mpr.get_doc(material_id.split("-")[1])
            
            return {
                'structure': structure,
                'material_id': material_id,
                'spacegroup': material_data.get('spacegroup', {}),
                'density': material_data.get('density', 'N/A'),
                'volume': material_data.get('volume', 'N/A'),
                'formation_energy_per_atom': material_data.get('formation_energy_per_atom', 'N/A'),
                'band_gap': material_data.get('band_gap', 'N/A')
            }, None
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def plot_crystal_structure_plotly(structure):
    """使用plotly绘制3D晶体结构"""
    try:
        # 获取晶格参数
        lattice = structure.lattice
        sites = structure.sites
        
        # 创建原子位置数据
        x, y, z = [], [], []
        colors, sizes, symbols = [], [], []
        
        # 原子颜色映射
        color_map = {
            'Li': 'purple', 'La': 'green', 'Zr': 'blue', 'O': 'red',
            'P': 'orange', 'S': 'yellow', 'Cl': 'green', 'Ge': 'gray',
            'Y': 'cyan', 'F': 'blue', 'Br': 'brown', 'I': 'purple',
            'Na': 'blue', 'K': 'violet', 'Mg': 'green', 'Ca': 'orange',
            'Al': 'gray', 'Si': 'yellow', 'Ti': 'gray', 'Fe': 'brown'
        }
        
        # 原子大小映射
        size_map = {
            'Li': 8, 'La': 15, 'Zr': 12, 'O': 10,
            'P': 11, 'S': 10, 'Cl': 10, 'Ge': 12,
            'Y': 13, 'F': 8, 'Br': 12, 'I': 14,
            'Na': 10, 'K': 12, 'Mg': 11, 'Ca': 12,
            'Al': 11, 'Si': 11, 'Ti': 12, 'Fe': 12
        }
        
        for site in sites:
            x.append(site.coords[0])
            y.append(site.coords[1])
            z.append(site.coords[2])
            element = site.species_string
            colors.append(color_map.get(element, 'gray'))
            sizes.append(size_map.get(element, 10))
            symbols.append(element)
        
        # 创建原子轨迹
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='darkgray')
            ),
            text=symbols,
            textposition="middle center",
            hoverinfo='text',
            name='Atoms'
        )
        
        # 创建晶格线
        lines_x, lines_y, lines_z = [], [], []
        
        # 绘制晶格向量
        origin = [0, 0, 0]
        a_vec = lattice.matrix[0]
        b_vec = lattice.matrix[1]
        c_vec = lattice.matrix[2]
        
        # a轴
        lines_x += [origin[0], a_vec[0], None]
        lines_y += [origin[1], a_vec[1], None]
        lines_z += [origin[2], a_vec[2], None]
        
        # b轴
        lines_x += [origin[0], b_vec[0], None]
        lines_y += [origin[1], b_vec[1], None]
        lines_z += [origin[2], b_vec[2], None]
        
        # c轴
        lines_x += [origin[0], c_vec[0], None]
        lines_y += [origin[1], c_vec[1], None]
        lines_z += [origin[2], c_vec[2], None]
        
        lattice_trace = go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='black', width=4),
            name='Lattice'
        )
        
        # 创建图形
        fig = go.Figure(data=[atom_trace, lattice_trace])
        
        # 更新布局
        fig.update_layout(
            title="Crystal Structure",
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=600,
            height=500,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting crystal structure: {str(e)}")
        return None

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
                # 首先尝试从Materials Project获取晶体结构
                if mp_api_key:
                    with st.spinner("Fetching crystal structure from Materials Project..."):
                        mp_data, mp_error = get_materials_project_structure(formula_input, mp_api_key)
                        
                        if mp_data and mp_error is None:
                            st.success("✅ Crystal structure retrieved from Materials Project")
                            
                            # 显示材料信息
                            st.subheader("Crystal Structure Information")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Material ID:** {mp_data['material_id']}")
                                st.write(f"**Space Group:** {mp_data['spacegroup'].get('symbol', 'N/A')} ({mp_data['spacegroup'].get('number', 'N/A')})")
                                st.write(f"**Density:** {mp_data['density']:.2f} g/cm³")
                                
                            with col2:
                                st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
                                st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
                                st.write(f"**Band Gap:** {mp_data['band_gap']:.3f} eV")
                            
                            # 绘制晶体结构
                            st.subheader("3D Crystal Structure")
                            fig = plot_crystal_structure_plotly(mp_data['structure'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Could not retrieve crystal structure: {mp_error}")
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

