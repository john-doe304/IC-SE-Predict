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
    .error-message {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .structure-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: white;
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

def get_materials_project_structure_updated(formula, api_key):
    """使用新的MPRester API获取晶体结构信息"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format. API key should be 32 alphanumeric characters."
        
        with MPRester(api_key) as mpr:
            # 使用新的summary.search方法
            try:
                # 方法1: 使用summary.search
                results = mpr.summary.search(formula=formula, fields=[
                    "material_id", "formula_pretty", "spacegroup", 
                    "density", "volume", "formation_energy_per_atom", 
                    "band_gap", "structure"
                ])
                
                if not results:
                    return None, f"No materials found for formula: {formula}"
                
                # 选择第一个结果
                material_data = results[0]
                material_id = material_data.material_id
                pretty_formula = material_data.formula_pretty
                structure = material_data.structure
                
                # 获取空间群信息
                spacegroup_data = material_data.spacegroup
                spacegroup_symbol = spacegroup_data.symbol if spacegroup_data else "N/A"
                spacegroup_number = spacegroup_data.number if spacegroup_data else "N/A"
                
                return {
                    'structure': structure,
                    'material_id': material_id,
                    'spacegroup': {
                        'symbol': spacegroup_symbol,
                        'number': spacegroup_number
                    },
                    'density': getattr(material_data, 'density', 'N/A'),
                    'volume': getattr(material_data, 'volume', 'N/A'),
                    'formation_energy_per_atom': getattr(material_data, 'formation_energy_per_atom', 'N/A'),
                    'band_gap': getattr(material_data, 'band_gap', 'N/A'),
                    'formula': formula,
                    'pretty_formula': pretty_formula
                }, None
                
            except Exception as search_error:
                # 方法2: 备用方法 - 使用get_structure_by_material_id
                try:
                    # 首先获取材料ID
                    entries = mpr.get_entries(formula)
                    if not entries:
                        return None, f"No materials found for formula: {formula}"
                    
                    # 选择第一个材料
                    material = entries[0]
                    material_id = material.entry_id
                    structure = mpr.get_structure_by_material_id(material_id.split("-")[1])
                    
                    # 获取详细数据
                    try:
                        doc = mpr.get_doc(material_id.split("-")[1])
                        spacegroup_data = doc.get("spacegroup", {})
                        spacegroup_symbol = spacegroup_data.get("symbol", "N/A") if isinstance(spacegroup_data, dict) else "N/A"
                        spacegroup_number = spacegroup_data.get("number", "N/A") if isinstance(spacegroup_data, dict) else "N/A"
                    except:
                        spacegroup_symbol = "N/A"
                        spacegroup_number = "N/A"
                        doc = {}
                    
                    return {
                        'structure': structure,
                        'material_id': material_id,
                        'spacegroup': {
                            'symbol': spacegroup_symbol,
                            'number': spacegroup_number
                        },
                        'density': doc.get('density', structure.density if structure else 'N/A'),
                        'volume': doc.get('volume', structure.volume if structure else 'N/A'),
                        'formation_energy_per_atom': material.energy_per_atom,
                        'band_gap': doc.get('band_gap', 'N/A'),
                        'formula': formula,
                        'pretty_formula': formula
                    }, None
                    
                except Exception as fallback_error:
                    return None, f"Both search methods failed: {str(fallback_error)}"
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def create_enhanced_structure_plot(structure, formula, material_id):
    """创建增强的晶体结构3D图"""
    try:
        # 获取晶格参数
        lattice = structure.lattice
        sites = structure.sites
        
        # 创建原子位置数据
        x, y, z = [], [], []
        colors, sizes, symbols, hover_texts = [], [], [], []
        
        # 改进的原子颜色映射（更接近MP的颜色）
        color_map = {
            'Li': '#CC80FF', 'La': '#70D4FF', 'Zr': '#4EACCE', 'O': '#FF0D0D',
            'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Ge': '#668F8F',
            'Y': '#94FFFF', 'F': '#90E050', 'Br': '#A62929', 'I': '#940094',
            'Na': '#AB5CF2', 'K': '#8F40D4', 'Mg': '#8AFF00', 'Ca': '#3DFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'Ti': '#BFC2C7', 'Fe': '#E06633'
        }
        
        # 原子大小映射
        size_map = {
            'Li': 10, 'La': 18, 'Zr': 14, 'O': 12,
            'P': 13, 'S': 12, 'Cl': 12, 'Ge': 14,
            'Y': 14, 'F': 10, 'Br': 13, 'I': 15,
            'Na': 12, 'K': 14, 'Mg': 13, 'Ca': 14,
            'Al': 13, 'Si': 13, 'Ti': 14, 'Fe': 14
        }
        
        for i, site in enumerate(sites):
            x.append(site.coords[0])
            y.append(site.coords[1])
            z.append(site.coords[2])
            element = site.species_string
            colors.append(color_map.get(element, '#CCCCCC'))
            sizes.append(size_map.get(element, 12))
            symbols.append(element)
            hover_texts.append(f"{element} atom<br>Position: ({site.coords[0]:.2f}, {site.coords[1]:.2f}, {site.coords[2]:.2f})")
        
        # 创建原子轨迹
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.95,
                line=dict(width=3, color='white')
            ),
            text=symbols,
            textposition="middle center",
            textfont=dict(size=14, color='black'),
            hoverinfo='text',
            hovertext=hover_texts,
            name='Atoms'
        )
        
        # 创建晶格线
        lines_x, lines_y, lines_z = [], [], []
        line_colors = []
        
        # 绘制晶格向量
        origin = [0, 0, 0]
        a_vec = lattice.matrix[0]
        b_vec = lattice.matrix[1]
        c_vec = lattice.matrix[2]
        
        # a轴 - 红色
        lines_x += [origin[0], a_vec[0], None]
        lines_y += [origin[1], a_vec[1], None]
        lines_z += [origin[2], a_vec[2], None]
        line_colors += ['red', 'red', None]
        
        # b轴 - 绿色
        lines_x += [origin[0], b_vec[0], None]
        lines_y += [origin[1], b_vec[1], None]
        lines_z += [origin[2], b_vec[2], None]
        line_colors += ['green', 'green', None]
        
        # c轴 - 蓝色
        lines_x += [origin[0], c_vec[0], None]
        lines_y += [origin[1], c_vec[1], None]
        lines_z += [origin[2], c_vec[2], None]
        line_colors += ['blue', 'blue', None]
        
        lattice_trace = go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color=line_colors, width=8),
            name='Lattice Vectors',
            hoverinfo='none'
        )
        
        # 创建图形
        fig = go.Figure(data=[atom_trace, lattice_trace])
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"Crystal Structure: {formula}<br><sub>Material ID: {material_id}</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8)
                ),
                bgcolor='white'
            ),
            width=800,
            height=700,
            margin=dict(l=20, r=20, b=50, t=80),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        # 添加坐标轴样式
        fig.update_scenes(
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating structure plot: {str(e)}")
        return None

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
                            # 修正化学公式 - 注意这里你输入的是 Li7La3272O12，应该是 Li7La3Zr2O12
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I').replace('3272', '3Zr2')
                            
                            mp_data, mp_error = get_materials_project_structure_updated(corrected_formula, mp_api_key)
                            
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
                                
                                # 创建并显示增强的3D结构图
                                st.subheader("🎯 3D Crystal Structure Visualization")
                                fig = create_enhanced_structure_plot(
                                    mp_data['structure'], 
                                    mp_data['pretty_formula'], 
                                    mp_data['material_id']
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 添加交互说明
                                    st.info("""
                                    **💡 Interactive Controls:**
                                    - **Rotate:** Click and drag to rotate the structure
                                    - **Zoom:** Use mouse wheel to zoom in/out
                                    - **Pan:** Hold Shift and drag to pan
                                    - **Reset:** Double-click to reset view
                                    - **Hover:** Hover over atoms to see details
                                    """)
                                
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
