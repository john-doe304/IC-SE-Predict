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
        <h2>IC-SE Predict - Solid State Ionic Conductivity Prediction Platform</h2>
        <blockquote>
            1. This platform predicts ionic conductivity of solid-state electrolytes based on material composition and structural features.<br>
            2. Supports various solid electrolyte materials including oxides, sulfides, and halides.<br>
            3. Code and data available at <a href='https://github.com/john-doe304/IC-SE-Predict' target='_blank'>GitHub Repository</a>.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 材料体系选择
material_systems = {
    "LLZO": {"Type": "Garnet", "Typical Composition": "Li7La3Zr2O12", "Temperature Range": "25-500°C"},
    "LGPS": {"Type": "Sulfide", "Typical Composition": "Li10GeP2S12", "Temperature Range": "25-300°C"},
    "NASICON": {"Type": "NASICON", "Typical Composition": "Li1+xAlxTi2-x(PO4)3", "Temperature Range": "25-400°C"},
    "Perovskite": {"Type": "Perovskite", "Typical Composition": "Li3xLa2/3-xTiO3", "Temperature Range": "25-600°C"},
    "Anti-Perovskite": {"Type": "Anti-Perovskite", "Typical Composition": "Li3OCl", "Temperature Range": "25-300°C"},
    "Sulfide": {"Type": "Sulfide Glass", "Typical Composition": "Li2S-P2S5", "Temperature Range": "25-200°C"},
    "Polymer": {"Type": "Polymer", "Typical Composition": "PEO-LiTFSI", "Temperature Range": "40-100°C"},
    "Halide": {"Type": "Halide", "Typical Composition": "Li3YCl6", "Temperature Range": "25-300°C"}
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
	
# 材料特征计算函数
def calculate_material_features_debug(formula):
    """带调试信息的特征计算函数"""
    try:
        from pymatgen.core import Composition
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, ValenceOrbital, IonProperty
        )
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
        
        print(f"开始处理化学式: {formula}")
        
        # 创建DataFrame用于特征计算
        df = pd.DataFrame({'Formula': [formula]})
        
        # 将字符串转换为composition对象
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)
        print(f"转换后DataFrame列: {df.columns.tolist()}")
        
        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            print("错误: 无法将化学式转换为composition对象")
            return {'Formula': formula}
        
        features = {'Formula': formula}

        # 1. 元素属性特征 (Magpie)
        print("计算元素属性特征...")
        ep_featurizer = ElementProperty.from_preset('magpie')
        df = ep_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        print(f"元素属性特征后列数: {len(df.columns)}")
        
        # 2. Meredig特征
        print("计算Meredig特征...")
        meredig_featurizer = Meredig()
        df = meredig_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        print(f"Meredig特征后列数: {len(df.columns)}")
        
        # 3. 化学计量特征
        print("计算化学计量特征...")
        stoichiometry_featurizer = Stoichiometry()
        df = stoichiometry_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        print(f"化学计量特征后列数: {len(df.columns)}")
        
        # 4. 离子特性特征
        print("计算离子特性特征...")
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
        ion_featurizer = IonProperty()
        df = ion_featurizer.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)
        print(f"离子特性特征后列数: {len(df.columns)}")

        # 提取数值特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns  # 现在np已经导入
        print(f"找到数值列: {len(numeric_columns)} 个")
        
        for col in numeric_columns:
            if col != 'Formula' and not col.startswith('composition'):
                value = df[col].iloc[0]
                features[col] = value if not pd.isna(value) else 0.0
        
        print(f"最终生成特征: {len(features)} 个")
        return features

    except Exception as e:
        print(f"特征计算失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return {'Formula': formula}
		
		
# 如果点击提交按钮
if submit_button:
    if not formula input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
               # 显示材料信息
                material_info = material_systems[material_system]
                    
                col1, col2, col3 = st.columns(3)
                with col1:
                        st.metric("Material Type", material_system)
                with col2:
                        st.metric("Crystal Structure", material_info["Type"])
                with col3:
                        st.metric("Temperature", f"{temperature} K")
						
                # 计算材料特征
                features = calculate_material_features(formula_input)

                if features:
                    # 显示特征信息
                    st.subheader("Material Features")
                    feature_df = pd.DataFrame([features])
                    filtered_features = filter_features(feature_df)
                        
                    st.write(f"Total features calculated: {len(features)}")
                    st.dataframe(filtered_features)
					
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
                    st.write("Input Data for Prediction:")
                    st.dataframe(input_df)

                
                
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
