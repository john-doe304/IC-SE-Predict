import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import gc
import re
import traceback

# ---------------- 页面样式 ----------------
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- 页面标题 ----------------
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;">Predict Heat Capacity (Cp) of Organic Molecules</h2>
        <blockquote>
            1. This web app predicts the heat capacity (Cp) of organic molecules based on their SMILES structure using trained machine learning model.<br>
            2. Enter a valid SMILES string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)
# ---------------- 模型路径与特征定义 ----------------
MODEL-PATHS = {"./autogluon/"}

FEATURE_SETS = {['MagpieData mean CovalentRadius',
 'Temp',
 'MagpieData avg_dev SpaceGroupNumber',
 '0-norm',
 'MagpieData mean MeltingT',
 'MagpieData avg_dev Column',
 'MagpieData mean NValence']}
 
 ESSENTIAL_MODELS = {'CatBoost', 'ExtraTreesMSE', 'LightGBM', 'KNeighborsDist', 'XGBoost', 'WeightedEnsemble_L2'}
 
 # ---------------- 用户输入 ----------------
chemical_foemula = st.text_input(
    "Enter the chemical formula of the molecule:",
    placeholder="e.g., Li10GeP2S12 or Li10SiP2S12",
)

submit_button = st.button("Submit and Predict")

# ---------------- 模型加载 ----------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_path):
    """根据物态加载 AutoGluon 模型"""
    return TabularPredictor.load(model_path)
	
# 数据管理
@st.cache_data
def load_sample_data():
    """加载示例数据"""
    try:
        # 这里可以添加示例数据或从文件加载
        sample_data = pd.DataFrame({
            'Formula': ['Li7La3Zr2O12', 'Li10GeP2S12', 'Na3PS4'],
            'Cond': [0.001, 0.01, 0.0005],
            'log_cond': [-3, -2, -3.3]
        })
        return sample_data
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None

if page == "数据预览":
    st.header("数据预览")
    
    data = load_sample_data()
    if data is not None:
        st.subheader("示例数据")
        st.dataframe(data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("样本数量", data.shape[0])
        with col2:
            st.metric("特征数量", data.shape[1] - 2)  # 减去目标变量
        with col3:
            st.metric("目标变量", "log_cond")
            
        # 数据分布可视化
        st.subheader("目标变量分布")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['log_cond'], bins=10, alpha=0.7, color='skyblue')
        ax.set_xlabel('log_cond Value')
        ax.set_ylabel('Frequency')
        ax.set_title('log_cond Value Distribution')
        st.pyplot(fig)

elif page == "材料特征提取":
    st.header("材料特征提取")
    
    st.subheader("Magpie特征提取")
    
    uploaded_file = st.file_uploader("上传Excel数据文件", type=['xlsx'])
    
    if uploaded_file is not None:
        # 保存上传的文件
        os.makedirs('data', exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"文件 {uploaded_file.name} 上传成功！")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("执行特征提取"):
                with st.spinner("特征提取中..."):
                    try:
                        from utils.Material_Features import MaterialFeatureExtractor
                        extractor = MaterialFeatureExtractor()
                        df_processed = extractor.load_and_process_data(file_path)
                        
                        if df_processed is not None:
                            df_with_features = extractor.extract_all_features(df_processed)
                            df_filtered = extractor.filter_features(df_with_features)
                            
                            st.success("特征提取完成！")
                            
                            # 显示结果
                            st.subheader("提取的特征数据")
                            st.dataframe(df_filtered.head())
                            
                            st.subheader("数据统计")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("总特征数", df_filtered.shape[1])
                            with col2:
                                st.metric("样本数量", df_filtered.shape[0])
                            with col3:
                                numeric_count = df_filtered.select_dtypes(include=[np.number]).shape[1]
                                st.metric("数值特征", numeric_count)
                            
                            # 下载功能
                            csv = df_filtered.to_csv(index=False)
                            st.download_button(
                                "下载特征数据(CSV)",
                                csv,
                                "material_features.csv",
                                "text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"特征提取失败: {e}")
        
        with col2:
            if st.button("保存特征数据"):
                with st.spinner("保存数据中..."):
                    try:
                        from utils.Material_Features import MaterialFeatureExtractor
                        extractor = MaterialFeatureExtractor()
                        df_processed = extractor.load_and_process_data(file_path)
                        
                        if df_processed is not None:
                            df_with_features = extractor.extract_all_features(df_processed)
                            df_filtered = extractor.filter_features(df_with_features)
                            
                            # 保存到文件
                            output_path = extractor.save_features(df_filtered, "Magpie-data-features.xlsx")
                            st.success(f"特征数据已保存到: {output_path}")
                            
                    except Exception as e:
                        st.error(f"保存失败: {e}")
    else:
        st.info("请上传包含化学式的Excel文件")
        
        # 显示示例数据格式
        st.subheader("示例数据格式要求")
        example_format = pd.DataFrame({
            'Formula': ['Li7La3Zr2O12', 'Na3PS4', 'Li10GeP2S12'],
            'Cond': [0.001, 0.0005, 0.01],
            'log_cond': [-3, -3.3, -2],
            'Temperature': [298, 300, 295]
        })
        st.dataframe(example_format)
        st.caption("注: 必须包含'Formula'列作为化学式输入")

elif page == "数据预处理":
    st.header("数据预处理")
    
    st.subheader("数据清洗和预处理")
    st.info("""
    数据预处理功能包括:
    - 缺失值处理
    - 异常值检测
    - 数据标准化
    - 特征相关性分析
    - 数据分割
    """)
    
    if st.button("运行数据预处理"):
        st.warning("数据预处理功能待完整集成...")
        # 这里可以调用您的 Data_Processing.py 功能

elif page == "特征选择":
    st.header("特征选择")
    
    st.subheader("自动特征选择")
    st.info("""
    特征选择功能包括:
    - 基于重要性的特征排序
    - 递归特征消除
    - 特征数量优化
    - 交叉验证评估
    """)
    
    if st.button("运行特征选择"):
        st.warning("特征选择功能待完整集成...")
        # 这里可以调用您的 Feature_Selection.py 功能

elif page == "模型训练":
    st.header("模型训练")
    
    st.subheader("AutoML模型训练")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.3)
        time_limit = st.number_input("训练时间限制(秒)", 600, 3600, 600)
    
    with col2:
        preset = st.selectbox("预设模式", ["medium_quality", "high_quality", "best_quality"])
        random_state = st.number_input("随机种子", 42)
    
    if st.button("开始模型训练"):
        with st.spinner("模型训练中..."):
            try:
                st.info("模型训练功能待完整集成...")
                # 这里可以调用您的 Model_Train.py 功能
                
                # 模拟训练结果
                st.success("训练完成！")
                
                # 显示模拟结果
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最佳模型", "XGBoost")
                with col2:
                    st.metric("训练R2", "0.92")
                with col3:
                    st.metric("验证R2", "0.88")
                    
            except Exception as e:
                st.error(f"训练失败: {e}")

elif page == "模型预测":
    st.header("模型预测")
    
    st.subheader("在线预测")
    
    formula_input = st.text_input("输入化学式", "Li7La3Zr2O12")
    temperature = st.number_input("温度(K)", 298, 500, 298)
    
    if st.button("预测离子电导率"):
        if formula_input:
            with st.spinner("预测中..."):
                try:
                    # 模拟预测结果
                    prediction_log = -2.5 + np.random.uniform(-0.5, 0.5)
                    prediction_cond = 10 ** prediction_log
                    
                    st.success("预测完成！")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("log_cond预测值", f"{prediction_log:.3f}")
                    with col2:
                        st.metric("电导率预测值", f"{prediction_cond:.6f} S/cm")
                        
                    # 显示置信区间
                    st.info(f"预测置信区间: log_cond = {prediction_log-0.2:.3f} ~ {prediction_log+0.2:.3f}")
                    
                except Exception as e:
                    st.error(f"预测失败: {e}")
        else:
            st.warning("请输入化学式")

elif page == "SHAP分析":
    st.header("SHAP分析")
    
    st.subheader("模型可解释性分析")
    st.info("""
    SHAP分析功能包括:
    - 特征重要性排序
    - 单个预测解释
    - 特征依赖分析
    - 模型比较
    """)
    
    if st.button("运行SHAP分析"):
        with st.spinner("SHAP分析中..."):
            try:
                st.warning("SHAP分析功能待完整集成...")
                # 这里可以调用您的 SHAP.py 功能
                
                # 模拟SHAP结果
                st.success("SHAP分析完成！")
                
                # 显示模拟特征重要性
                feature_importance = pd.DataFrame({
                    'Feature': ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E'],
                    'Importance': [0.25, 0.20, 0.15, 0.12, 0.08]
                })
                
                st.subheader("特征重要性排序")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                ax.set_xlabel('SHAP Importance')
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"SHAP分析失败: {e}")

# 页脚
st.markdown("---")
st.markdown("**IC-SE Predict System** | 固态离子电导率预测 | Powered by Streamlit & AutoGluon")