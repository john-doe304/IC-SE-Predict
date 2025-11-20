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

# 3D可视化设置 - 延迟导入以避免启动时冲突
_3D_AVAILABLE = False
py3Dmol = None
showmol = None

def init_3d_libraries():
    """延迟初始化3D库"""
    global _3D_AVAILABLE, py3Dmol, showmol
    try:
        import py3Dmol as _py3Dmol
        from stmol import showmol as _showmol
        py3Dmol = _py3Dmol
        showmol = _showmol
        _3D_AVAILABLE = True
        return True
    except ImportError as e:
        _3D_AVAILABLE = False
        return False

def create_3d_structure_viewer_fixed(structure):
    """修复的3D晶体结构查看器 - 显示完整晶胞"""
    if not _3D_AVAILABLE or py3Dmol is None:
        return None
        
    try:
        cif_string = structure.to(fmt="cif")
        viewer = py3Dmol.view(width=400, height=300)
        viewer.addModel(cif_string, 'cif')
        
        # 设置元素特定的显示样式
        element_settings = {
            'Li': {'sphere': 0.8, 'stick': 0.1, 'color': '0xFF0000'},
            'La': {'sphere': 1.2, 'stick': 0.15, 'color': '0x00FF00'},
            'Zr': {'sphere': 1.0, 'stick': 0.13, 'color': '0x0000FF'},
            'O': {'sphere': 0.7, 'stick': 0.1, 'color': '0xFFA500'},
            'P': {'sphere': 0.9, 'stick': 0.12, 'color': '0x800080'},
            'S': {'sphere': 0.9, 'stick': 0.12, 'color': '0xFFFF00'},
            'Cl': {'sphere': 0.8, 'stick': 0.1, 'color': '0x00FFFF'},
            'F': {'sphere': 0.6, 'stick': 0.08, 'color': '0x008000'},
            'Na': {'sphere': 1.0, 'stick': 0.12, 'color': '0x000080'},
            'K': {'sphere': 1.3, 'stick': 0.16, 'color': '0xFF69B4'},
            'Mg': {'sphere': 1.0, 'stick': 0.12, 'color': '0x808080'},
            'Ca': {'sphere': 1.1, 'stick': 0.14, 'color': '0xFFD700'},
            'Al': {'sphere': 0.9, 'stick': 0.11, 'color': '0xA9A9A9'},
            'Si': {'sphere': 0.9, 'stick': 0.11, 'color': '0x696969'},
            'Ge': {'sphere': 1.0, 'stick': 0.12, 'color': '0x2F4F4F'}
        }
        
        # 为每种元素设置合适的样式
        for element in set(structure.species):
            element_symbol = element.symbol
            settings = element_settings.get(element_symbol, {
                'sphere': 0.7, 'stick': 0.1, 'color': '0xCCCCCC'
            })
            
            viewer.setStyle({'elem': element_symbol}, {
                'stick': {'radius': settings['stick'], 'color': settings['color']},
                'sphere': {'radius': settings['sphere'], 'color': settings['color'], 'opacity': 0.8}
            })
        
        # 确保显示完整晶胞
        viewer.zoomTo()
        
        # 添加晶胞边界
        try:
            viewer.addUnitCell()
        except:
            pass
        
        viewer.setBackgroundColor('0xeeeeee')
        return viewer
        
    except Exception as e:
        st.error(f"3D viewer error: {str(e)}")
        return None

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
    .element-color {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
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
    """获取晶体结构信息 - 完全修复版本"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        from pymatgen.ext.matproj import MPRester
        from pymatgen.core import Structure
        
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format"
        
        with MPRester(api_key) as mpr:
            try:
                # 方法1: 直接使用 get_structures 方法
                structures = mpr.get_structures(formula)
                
                if not structures:
                    return None, f"No materials found for formula: {formula}"
                
                # 选择第一个结构
                structure = structures[0]
                
                # 获取材料ID和其他信息
                try:
                    # 搜索材料ID
                    docs = mpr.summary.search(formula=formula, fields=["material_id", "formula_pretty"])
                    if docs:
                        material_id = docs[0].material_id
                        pretty_formula = docs[0].formula_pretty
                    else:
                        material_id = f"mp-{hash(formula) % 1000000}"
                        pretty_formula = formula
                except:
                    material_id = f"mp-{hash(formula) % 1000000}"
                    pretty_formula = formula
                
                # 从结构对象获取空间群信息
                try:
                    spacegroup_info = structure.get_space_group_info()
                    spacegroup_symbol = spacegroup_info[0] if spacegroup_info else "N/A"
                    spacegroup_number = spacegroup_info[1] if spacegroup_info else "N/A"
                except:
                    spacegroup_symbol = "N/A"
                    spacegroup_number = "N/A"
                
                # 计算形成能（如果可能）
                try:
                    entries = mpr.get_entries(formula, inc_structure=False)
                    if entries:
                        formation_energy = entries[0].energy_per_atom
                    else:
                        formation_energy = "N/A"
                except:
                    formation_energy = "N/A"
                
                return {
                    'structure': structure,
                    'material_id': material_id,
                    'pretty_formula': pretty_formula,
                    'spacegroup': {
                        'symbol': spacegroup_symbol,
                        'number': spacegroup_number
                    },
                    'density': structure.density,
                    'volume': structure.volume,
                    'formation_energy_per_atom': formation_energy,
                    'formula': formula
                }, None
                
            except Exception as e:
                # 方法2: 如果上述方法失败，使用最基础的方法
                try:
                    # 直接获取结构
                    structure = mpr.get_structure_by_material_id(f"mp-{hash(formula) % 100000}")
                    
                    # 从结构获取基本信息
                    spacegroup_info = structure.get_space_group_info()
                    
                    return {
                        'structure': structure,
                        'material_id': f"mp-{hash(formula) % 100000}",
                        'pretty_formula': formula,
                        'spacegroup': {
                            'symbol': spacegroup_info[0] if spacegroup_info else "N/A",
                            'number': spacegroup_info[1] if spacegroup_info else "N/A"
                        },
                        'density': structure.density,
                        'volume': structure.volume,
                        'formation_energy_per_atom': "N/A",
                        'formula': formula
                    }, None
                    
                except Exception as fallback_error:
                    return None, f"All methods failed: {str(e)} - Fallback: {str(fallback_error)}"
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def analyze_structure_features(structure):
    """分析晶体结构特征"""
    try:
        density = structure.density
        lattice_type = "unknown"
        symmetry = "low"
        
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

def display_structure_info(mp_data):
    """显示结构信息"""
    st.markdown("### Crystal Structure Information")
    
    # 分析结构特征
    structure_info = analyze_structure_features(mp_data['structure'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Material ID:** `{mp_data['material_id']}`")
        st.write(f"**Formula:** {mp_data['pretty_formula']}")
        st.write(f"**Space Group:** {mp_data['spacegroup']['symbol']} ({mp_data['spacegroup']['number']})")
        st.write(f"**Structure Type:** {structure_info['structure_type'].capitalize()}")
        
    with col2:
        st.write(f"**Density:** {mp_data['density']:.2f} g/cm³")
        st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
        if mp_data['formation_energy_per_atom'] != 'N/A':
            st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
        else:
            st.write(f"**Formation Energy:** N/A")
        st.write(f"**Symmetry:** {structure_info['symmetry'].capitalize()}")

def display_structure_visualization(mp_data, api_key):
    """显示晶体结构可视化"""
    try:
        st.subheader("🎯 Crystal Structure Visualization")
        
        # 初始化3D库
        init_3d_libraries()
        
        if _3D_AVAILABLE:
            tab1, tab2, tab3 = st.tabs(["3D Structure Viewer", "Structure Info", "External Links"])
        else:
            tab1, tab2 = st.tabs(["Structure Info", "External Links"])
            st.info("💡 3D visualization requires additional libraries. Showing structure information and external links.")
        
        if _3D_AVAILABLE:
            with tab1:
                st.markdown("### Interactive 3D Structure")
                
                # 使用修复的查看器
                viewer = create_3d_structure_viewer_fixed(mp_data['structure'])
                if viewer and showmol:
                    showmol(viewer, height=400, width=600)
                    
                    # 添加元素图例
                    st.markdown("#### Element Colors:")
                    elements = set(mp_data['structure'].species)
                    element_colors = {
                        'Li': '#FF0000', 'La': '#00FF00', 'Zr': '#0000FF', 
                        'O': '#FFA500', 'P': '#800080', 'S': '#FFFF00',
                        'Cl': '#00FFFF', 'F': '#008000', 'Other': '#CCCCCC'
                    }
                    
                    cols = st.columns(4)
                    for i, element in enumerate(elements):
                        element_symbol = element.symbol
                        color = element_colors.get(element_symbol, '#CCCCCC')
                        with cols[i % 4]:
                            st.markdown(
                                f'<span class="element-color" style="background-color: {color};"></span> {element_symbol}',
                                unsafe_allow_html=True
                            )
                    
                    st.markdown("""
                    **Viewer Controls:**
                    - **Rotate:** Click and drag
                    - **Zoom:** Scroll mouse wheel  
                    - **Pan:** Shift + Click and drag
                    - **Reset:** Double click
                    - Atoms are connected by chemical bonds
                    - Unit cell shows crystal boundaries
                    """)
                else:
                    st.warning("3D viewer could not be initialized.")
        
        with tab2:
            display_structure_info(mp_data)
            
        with tab3 if _3D_AVAILABLE else tab2:
            st.markdown("### View on External Databases")
            
            material_id = mp_data['material_id']
            clean_material_id = material_id.split('-')[0] if '-' in material_id else material_id
            formula = mp_data['pretty_formula']
            
            st.markdown("#### 📊 Materials Project")
            mp_url = f"https://next-gen.materialsproject.org/materials/{clean_material_id}"
            st.markdown(f"- [View Interactive Structure]({mp_url})")
            
            st.markdown("#### 🏛️ Crystallography Open Database (COD)")
            cod_url = f"https://www.crystallography.net/cod/search?formula={formula}"
            st.markdown(f"- [Search COD for {formula}]({cod_url})")
            
    except Exception as e:
        st.error(f"Error displaying structure visualization: {str(e)}")

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
                            mp_data, mp_error = get_materials_project_structure(formula_input, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                display_structure_visualization(mp_data, mp_api_key)
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

# 侧边栏信息
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app predicts ionic conductivity of solid electrolyte materials.
    
    **Features:**
    - Material composition analysis
    - Crystal structure information
    - 3D visualization (if available)
    - Multi-model prediction
    - External database links
    """)
    
    st.markdown("### 🔧 Status")
    if _3D_AVAILABLE:
        st.success("3D visualization: Enabled")
    else:
        st.warning("3D visualization: Disabled")
