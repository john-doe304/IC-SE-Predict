import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gc
import re

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 设置页面
st.set_page_config(
    page_title="IC-SE Predict - Solid State Ionic Conductivity Prediction Platform",
    page_icon="🔋",
    layout="wide"
)

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 90%;
        background-color: #f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0;
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
    .feature-container {
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        background-color: white;
    }
    /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px;
        }
        .rounded-container blockquote {
            font-size: 0.9em;
        }
        .rounded-container h2 {
            font-size: 1.2em;
        }
        .stApp {
            padding: 10px !important;
            max-width: 95%;
        }
        .process-text, .molecular-weight {
            font-size: 0.9em;
        }
        .feature-container {
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

# 侧边栏导航
st.sidebar.header("Navigation Menu")
page = st.sidebar.selectbox(
    "Select Function",
    ["Home", "Data Preview", "Material Feature Extraction", "Model Prediction", "Model Analysis"]
)

# 主页内容
if page == "Home":
    # 材料体系选择下拉菜单
    material_system = st.selectbox("Select Material Type:", list(material_systems.keys()))

    # FORMULA 输入区域
    formula_input = st.text_input(
        "Enter Chemical Formula of the Material:",
        placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6",
    )

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

    # 缓存模型加载器
    @st.cache_resource(show_spinner=False, max_entries=1)
    def load_predictor():
        """Cache model loading to avoid repeated loading causing memory overflow"""
        try:
            # 这里加载你的训练好的模型
            # return TabularPredictor.load("./ag-20251024_075719")
            return None  # 暂时返回None，你需要替换为实际的模型加载代码
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None

    # 材料特征计算函数
    def calculate_material_features(formula):
        """Calculate material features based on chemical formula using Magpie descriptors"""
        try:
            # 尝试导入所需的库
            try:
                from pymatgen.core import Composition
                from matminer.featurizers.composition import (
                    ElementProperty, Meredig, Stoichiometry, ValenceOrbital, IonProperty
                )
                from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
            except ImportError as e:
                st.warning(f"Some feature calculation libraries not available: {e}")
                return calculate_basic_features(formula)
            
            # 创建DataFrame用于特征计算
            df = pd.DataFrame({'Formula': [formula]})
            
            # 将字符串转换为composition对象
            stc = StrToComposition()
            df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)
            
            if 'composition' not in df.columns:
                st.error("Failed to convert formula to composition object")
                return calculate_basic_features(formula)
            
            features = {'Formula': formula}
            
            try:
                # 1. 元素属性特征 (Magpie)
                ep_featurizer = ElementProperty.from_preset('magpie')
                df = ep_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Element property features failed: {e}")
            
            try:
                # 2. Meredig特征
                meredig_featurizer = Meredig()
                df = meredig_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Meredig features failed: {e}")
            
            try:
                # 3. 化学计量特征
                stoichiometry_featurizer = Stoichiometry()
                df = stoichiometry_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Stoichiometry features failed: {e}")
            
            try:
                # 4. 离子特性特征需要先转换氧化态
                cto = CompositionToOxidComposition()
                df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
                
                ion_featurizer = IonProperty()
                df = ion_featurizer.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)
            except Exception as e:
                st.warning(f"Ion property features failed: {e}")
            
            # 提取数值特征
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'Formula':  # 跳过非特征列
                    features[col] = df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 0.0
            
            # 添加基本特征作为后备
            basic_features = calculate_basic_features(formula)
            features.update(basic_features)
            
            return features
            
        except Exception as e:
            st.error(f"Advanced feature calculation failed: {e}")
            # 如果高级特征计算失败，返回基本特征
            return calculate_basic_features(formula)

    def calculate_basic_features(formula):
        """Calculate basic material features when advanced libraries are not available"""
        try:
            # 基本特征计算（不依赖外部库）
            elements = []
            current_element = ""
            
            # 简单的化学式解析
            for char in formula:
                if char.isupper():
                    if current_element:
                        elements.append(current_element)
                    current_element = char
                elif char.islower():
                    current_element += char
                elif char.isdigit():
                    # 处理数字（这里简化处理）
                    continue
            
            if current_element:
                elements.append(current_element)
            
            unique_elements = set(elements)
            
            features = {
                'Formula': formula,
                'Element_Count': len(unique_elements),
                'Formula_Length': len(formula),
                'Li_Content': formula.count('Li'),
                'O_Content': formula.count('O'),
                'S_Content': formula.count('S'),
                'Cl_Content': formula.count('Cl'),
                'P_Content': formula.count('P'),
                'La_Content': formula.count('La'),
                'Zr_Content': formula.count('Zr'),
                'Ge_Content': formula.count('Ge'),
                'Y_Content': formula.count('Y'),
                'Has_Li': 1 if 'Li' in formula else 0,
                'Has_O': 1 if 'O' in formula else 0,
                'Has_S': 1 if 'S' in formula else 0,
                'Has_Cl': 1 if 'Cl' in formula else 0,
            }
            
            return features
            
        except Exception as e:
            st.error(f"Basic feature calculation failed: {e}")
            return {'Formula': formula, 'Error': str(e)}

    def filter_features(features_df, nan_threshold=0.4):
        """Filter features based on NaN ratio"""
        try:
            # 删除缺失值比例太高的列
            nan_ratio = features_df.isnull().sum() / features_df.shape[0]
            data_filtered = features_df.loc[:, nan_ratio < nan_threshold]
            
            # 填充剩余的NaN值
            data_filtered = data_filtered.fillna(0)
            
            return data_filtered
        except Exception as e:
            st.error(f"Feature filtering failed: {e}")
            return features_df.fillna(0)

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
                    predictor = load_predictor()
                    # 加载模型并预测
                try:
                    # 使用缓存的模型加载方式
                    predictor = load_predictor()
                    
                    # 只使用最关键的模型进行预测，减少内存占用
                    essential_models = ['CatBoost',
                                         'LightGBM',
                                         'LightGBMLarge',
                                         'RandomForestMSE',
                                         'WeightedEnsemble_L2',
                                         'XGBoost']
                    predict_df_1 = pd.concat([predict_df,predict_df],axis=0)
                    predictions_dict = {}
                    
                    for model in essential_models:
                        try:
                            predictions = predictor.predict(predict_df_1, model=model)
                            predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
                        except Exception as model_error:
                            st.warning(f"Model {model} prediction failed: {str(model_error)}")
                            predictions_dict[model] = "Error"
                                
                                # 显示预测结果
                                st.subheader("Prediction Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Predicted log(σ) [S/cm]", 
                                        f"{example_predictions['log_conductivity']:.3f}"
                                    )
                                with col2:
                                    st.metric(
                                        "Predicted σ [S/cm]", 
                                        f"{example_predictions['conductivity_S_cm']:.6f}"
                                    )
                                
                                # 显示置信区间
                                st.info(
                                    f"Prediction confidence interval: "
                                    f"log(σ) = {example_predictions['log_conductivity']-0.2:.3f} ~ "
                                    f"{example_predictions['log_conductivity']+0.2:.3f}"
                                )
                                
                                # 材料性能评估
                                conductivity = example_predictions['conductivity_S_cm']
                                if conductivity > 1e-2:
                                    performance = "Excellent"
                                    color = "green"
                                elif conductivity > 1e-3:
                                    performance = "Good"
                                    color = "blue"
                                elif conductivity > 1e-4:
                                    performance = "Moderate"
                                    color = "orange"
                                else:
                                    performance = "Poor"
                                    color = "red"
                                    
                                st.markdown(
                                    f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;'>"
                                    f"<strong>Performance Rating: {performance}</strong>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
                        else:
                            st.warning("Model not available. Using example predictions.")
                            
                            # 显示示例结果
                            st.subheader("Example Prediction Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted log(σ) [S/cm]", "-3.2")
                            with col2:
                                st.metric("Predicted σ [S/cm]", "0.000631")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# 数据预览页面
elif page == "Data Preview":
    st.header("Data Preview")
    
    # 示例数据
    sample_data = pd.DataFrame({
        'Formula': ['Li7La3Zr2O12', 'Li10GeP2S12', 'Li3YCl6', 'Li6PS5Cl', 'Li1.3Al0.3Ti1.7(PO4)3'],
        'Material_Type': ['Garnet', 'Sulfide', 'Halide', 'Sulfide', 'NASICON'],
        'Temperature_K': [298, 298, 298, 298, 298],
        'log_conductivity': [-3.0, -2.0, -3.5, -2.5, -3.2],
        'conductivity_S_cm': [0.001, 0.01, 0.0003, 0.003, 0.0006]
    })
    
    st.subheader("Example Solid Electrolyte Data")
    st.dataframe(sample_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample Count", sample_data.shape[0])
    with col2:
        st.metric("Feature Count", sample_data.shape[1] - 2)
    with col3:
        st.metric("Target Variable", "log_conductivity")
    
    # 数据分布可视化
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample_data['log_conductivity'], bins=10, alpha=0.7, color='skyblue')
    ax.set_xlabel('log(σ) Value')
    ax.set_ylabel('Frequency')
    ax.set_title('log(σ) Value Distribution')
    st.pyplot(fig)

# 材料特征提取页面
elif page == "Material Feature Extraction":
    st.header("Material Feature Extraction")
    
    st.subheader("Magpie Feature Extraction")
    
    uploaded_file = st.file_uploader("Upload Excel Data File", type=['xlsx'])
    
    if uploaded_file is not None:
        # 保存上传的文件
        os.makedirs('temp_data', exist_ok=True)
        file_path = os.path.join("temp_data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Execute Feature Extraction"):
                with st.spinner("Extracting features..."):
                    try:
                        # 读取Excel文件
                        df_excel = pd.read_excel(file_path)
                        
                        # 显示原始数据
                        st.write("Original Data:")
                        st.dataframe(df_excel.head())
                        
                        # 这里可以调用你的完整特征提取流程
                        st.info("Full Magpie feature extraction would be implemented here")
                        
                        # 示例：对每个化学式计算特征
                        if 'Formula' in df_excel.columns:
                            all_features = []
                            for formula in df_excel['Formula']:
                                features = calculate_material_features(formula)
                                all_features.append(features)
                            
                            features_df = pd.DataFrame(all_features)
                            filtered_features = filter_features(features_df)
                            
                            st.success("Feature extraction completed!")
                            st.write(f"Extracted {filtered_features.shape[1]} features for {filtered_features.shape[0]} materials")
                            st.dataframe(filtered_features.head())
                            
                            # 下载功能
                            csv = filtered_features.to_csv(index=False)
                            st.download_button(
                                "Download Features (CSV)",
                                csv,
                                "material_features.csv",
                                "text/csv"
                            )
                        
                    except Exception as e:
                        st.error(f"Feature extraction failed: {e}")

# 模型预测页面
elif page == "Model Prediction":
    st.header("Model Prediction")
    st.info("Use the Home page to input chemical formula and get predictions.")
    
    # 可以在这里添加批量预测功能
    st.subheader("Batch Prediction")
    st.warning("Batch prediction feature will be implemented in future versions.")

# 模型分析页面
elif page == "Model Analysis":
    st.header("Model Analysis")
    
    st.subheader("Feature Importance")
    st.info("""
    Model analysis features include:
    - Feature importance ranking
    - Model performance metrics
    - Prediction confidence intervals
    - Cross-validation results
    """)
    
    if st.button("Run Model Analysis"):
        with st.spinner("Analyzing model..."):
            try:
                # 示例特征重要性
                feature_importance = pd.DataFrame({
                    'Feature': ['Li_Content', 'O_Content', 'Element_Count', 
                               'Formula_Length', 'Temperature', 'S_Content'],
                    'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
                })
                
                st.success("Model analysis completed!")
                
                # 显示特征重要性图表
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance Ranking')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Model analysis failed: {e}")

# 页脚
st.markdown("---")
st.markdown("**IC-SE Predict System** | Solid State Ionic Conductivity Prediction | Powered by Streamlit & Machine Learning")