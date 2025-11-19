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
import gc
import re
from tqdm import tqdm 
import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.ext.matproj import MPRester
import plotly.graph_objects as go
import io
import requests
from PIL import Image
import base64
import py3Dmol
from stmol import showmol
import plotly.express as px

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
    /* 3D查看器样式 */
    .mol-container {
        width: 100%;
        height: 400px;
        position: relative;
        border: 1px solid #ccc;
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

def create_3d_structure_viewer(structure):
    """使用py3Dmol创建3D晶体结构查看器"""
    try:
        # 将结构转换为CIF格式
        cif_string = structure.to(fmt="cif")
        
        # 创建3D查看器
        viewer = py3Dmol.view(width=400, height=300)
        viewer.addModel(cif_string, 'cif')
        viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.5}})
        viewer.zoomTo()
        viewer.setBackgroundColor('0xeeeeee')
        
        return viewer
        
    except Exception as e:
        st.error(f"Error creating 3D viewer: {str(e)}")
        return None

def display_structure_visualization(mp_data, api_key):
    """显示晶体结构可视化"""
    try:
        st.subheader("🎯 Crystal Structure Visualization")
        
        # 创建选项卡：3D查看器、外部链接、结构信息
        tab1, tab2, tab3 = st.tabs(["3D Structure Viewer", "External Links", "Structure Info"])
        
        with tab1:
            st.markdown("### Interactive 3D Structure")
            
            # 创建3D查看器
            viewer = create_3d_structure_viewer(mp_data['structure'])
            if viewer:
                # 在Streamlit中显示3D查看器
                showmol(viewer, height=400, width=600)
                
                # 添加查看控制说明
                st.markdown("""
                **Viewer Controls:**
                - **Rotate:** Click and drag
                - **Zoom:** Scroll mouse wheel
                - **Pan:** Shift + Click and drag
                - **Reset:** Double click
                """)
            else:
                st.warning("3D viewer could not be initialized. Showing structure information instead.")
                display_structure_info(mp_data)
        
        with tab2:
            st.markdown("### View on External Databases")
            
            material_id = mp_data['material_id']
            clean_material_id = material_id.split('-')[0] if '-' in material_id else material_id
            formula = mp_data['pretty_formula']
            
            # Materials Project 链接
            st.markdown("#### Materials Project")
            mp_urls = [
                f"https://next-gen.materialsproject.org/materials/{clean_material_id}",
                f"https://legacy.materialsproject.org/materials/{clean_material_id}",
            ]
            
            for i, url in enumerate(mp_urls):
                st.markdown(f"[🔗 Materials Project Link {i+1}]({url})")
            
            # Crystallography Open Database 链接
            st.markdown("#### Crystallography Open Database (COD)")
            cod_url = f"https://www.crystallography.net/cod/search?formula={formula}"
            st.markdown(f"[🔗 Search COD for {formula}]({cod_url})")
            
            # Springer Materials 链接
            st.markdown("#### Springer Materials")
            springer_url = f"https://materials.springer.com/search?searchTerm={formula}"
            st.markdown(f"[🔗 Search Springer Materials]({springer_url})")
            
        with tab3:
            display_structure_info(mp_data)
            
    except Exception as e:
        st.error(f"Error displaying structure visualization: {str(e)}")

def display_structure_info(mp_data):
    """显示结构信息"""
    st.markdown("### Structure Information")
    
    # 分析结构特征
    structure_info = analyze_structure_features(mp_data['structure'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Material ID:** `{mp_data['material_id']}`")
        st.write(f"**Formula:** {mp_data['pretty_formula']}")
        st.write(f"**Space Group:** {mp_data['spacegroup'].get('symbol', 'N/A')} ({mp_data['spacegroup'].get('number', 'N/A')})")
        st.write(f"**Structure Type:** {structure_info['structure_type'].capitalize()}")
        
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
        st.write(f"**Symmetry:** {structure_info['symmetry'].capitalize()}")

def create_static_structure_plot(structure):
    """创建静态晶体结构图"""
    try:
        # 使用plotly创建晶格参数可视化
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        fig = go.Figure()
        
        # 添加晶格参数信息
        fig.add_annotation(
            text=f"Lattice Parameters:<br>a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å<br>"
                 f"α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°<br>"
                 f"Volume: {structure.volume:.2f}Å³<br>"
                 f"Density: {structure.density:.2f} g/cm³",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Crystal Lattice Parameters",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='lightgray'
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
                            # 修正化学公式
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I').replace('3272', '3Zr2')
                            
                            mp_data, mp_error = get_materials_project_structure_simple(corrected_formula, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                
                                # 显示结构可视化
                                display_structure_visualization(mp_data, mp_api_key)
                                
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                # 即使没有获取到结构，也显示基本信息
                                st.info("💡 The material might not exist in Materials Project database")
                                # 显示外部数据库链接
                                st.subheader("🔗 Search on External Databases")
                                formula = formula_input
                                cod_url = f"https://www.crystallography.net/cod/search?formula={formula}"
                                springer_url = f"https://materials.springer.com/search?searchTerm={formula}"
                                
                                st.markdown(f"- [Crystallography Open Database (COD)]({cod_url})")
                                st.markdown(f"- [Springer Materials]({springer_url})")
                                st.markdown(f"- [Materials Project Search](https://next-gen.materialsproject.org/search)")
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
