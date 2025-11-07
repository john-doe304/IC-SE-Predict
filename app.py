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
            2. Code and data available at <a href='https://github.com/john-doe304/IC-SE-Predict' target='_blank'>GitHub Repository</a>.
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

# 缓存模型加载器以避免重复加载
@st.cache_resource(show_spinner=False, max_entries=1)  # 限制只缓存一个实例
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

def mol_to_image(mol, size=(300, 300)):
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


# 晶体结构数据库
crystal_structures = {
    "Li7La3Zr2O12": {
        "crystal_system": "Cubic",
        "space_group": "Ia-3d",
        "lattice_parameters": "a = 12.97 Å",
        "density": "5.08 g/cm³",
        "reference": "Murugan et al., Angew. Chem. Int. Ed. (2007)"
    },
    "Li10GeP2S12": {
        "crystal_system": "Tetragonal", 
        "space_group": "P4_2/nmc",
        "lattice_parameters": "a = 8.72 Å, c = 12.54 Å",
        "density": "2.04 g/cm³",
        "reference": "Kamaya et al., Nat. Mater. (2011)"
    },
    "Li3YCl6": {
        "crystal_system": "Trigonal",
        "space_group": "R-3m", 
        "lattice_parameters": "a = 6.62 Å, c = 18.24 Å",
        "density": "2.67 g/cm³",
        "reference": "Asano et al., Adv. Mater. (2018)"
    },
    "Li3OCl": {
        "crystal_system": "Cubic",
        "space_group": "Pm-3m",
        "lattice_parameters": "a = 3.92 Å",
        "density": "2.41 g/cm³", 
        "reference": "Zhao et al., Nat. Commun. (2016)"
    },
	"Li3OCl": {
        "crystal_system": "Cubic",
        "space_group": "Pm-3m",
        "lattice_parameters": "a = 3.92 Å",
        "density": "2.41 g/cm³", 
        "reference": "Zhao et al., Nat. Commun. (2016)"
    },
    "Li1+xAlxTi2-x(PO4)3": {
        "crystal_system": "Rhombohedral",
        "space_group": "R-3c",
        "lattice_parameters": "a = 8.51 Å, c = 20.84 Å",
        "density": "2.94 g/cm³",
        "reference": "Aono et al., J. Electrochem. Soc. (1990)"
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
            "reference": "Typical Garnet Structure"
        }
    elif "Li" in formula and ("S" in formula or "P" in formula):
        return {
            "crystal_system": "Tetragonal/Orthorhombic", 
            "space_group": "P4_2/nmc/Pnma",
            "lattice_parameters": "a~8.7 Å, c~12.5 Å",
            "density": "~2.0-2.5 g/cm³",
            "reference": "Typical Sulfide Structure"
        }
		elif "Li" in formula and ("Cl" in formula or "Br" in formula or "I" in formula):
        return {
            "crystal_system": "Trigonal/Hexagonal",
            "space_group": "R-3m/P6_3/mmc", 
            "lattice_parameters": "a~6.6 Å, c~18.2 Å",
            "density": "~2.5-3.0 g/cm³",
            "reference": "Typical Halide Structure"
        }
    else:
        return {
            "crystal_system": "Unknown",
            "space_group": "Unknown", 
            "lattice_parameters": "Unknown",
            "density": "Unknown",
            "reference": "Structure data not available"
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
		
#过滤特征（仅展示非零数值列）
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



#自动匹配模型特征
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

				# 显示晶体结构信息
                st.subheader("📐 Crystal Structure Information")
                crystal_info = get_crystal_structure_info(formula_input)
                
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
						
                # 计算材料特征
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")
                feature_df = pd.DataFrame([features])
                filtered_df = filter_features(feature_df)
                st.subheader("Extracted Material Features (non-zero numeric columns)")
                st.dataframe(filtered_df)
			
                if features:
                    # 显示特征信息
                   
                    feature_df = pd.DataFrame([features])
                    filtered_features = filter_features(feature_df)
                        
                   
                    
					
                    # 创建输入数据
                    input_data = {
                        "Formula": [formula_input],
                         "Material_Type": [material_system],
                         "Temperature_K": [temperature],
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
                
                    # 显示输入数据
                    #st.write("Input Data for Prediction:")
                    #st.dataframe(input_df)

                
                
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












