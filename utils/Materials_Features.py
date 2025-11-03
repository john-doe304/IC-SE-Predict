import os
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from matminer.featurizers.composition import (
    ElementProperty, Meredig, Stoichiometry, ValenceOrbital, IonProperty
)
from matminer.featurizers.structure import (
    DensityFeatures, GlobalSymmetryFeatures, StructuralHeterogeneity,
    MaximumPackingEfficiency, ChemicalOrdering, StructureComposition
)
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

class MaterialFeatureExtractor:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
    
    def load_and_process_data(self, file_path):
        """加载并处理数据"""
        try:
            df = pd.read_excel(file_path)
            print("原始Excel数据列名:", df.columns.tolist())
            print("\n前5行数据:")
            print(df.head())
            
            return self.standardize_composition_column(df)
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def standardize_composition_column(self, df):
        """标准化化学式列名"""
        df_excel = df.copy()
        
        # 标准化列名（确保化学式列名为'composition'）
        if 'Formula' in df_excel.columns:
            df_excel['composition_obj'] = df_excel['Formula']
        elif 'formula' in df_excel.columns:
            df_excel['composition_obj'] = df_excel['formula']
        else:
            # 假设第一列是化学式
            df_excel['composition_obj'] = df_excel.iloc[:, 0]
        
        # 将字符串转换为composition对象
        print("\n正在转换化学式为composition对象...")
        stc = StrToComposition()
        df_excel = stc.featurize_dataframe(df_excel, 'composition_obj', ignore_errors=True)
        
        return df_excel
    
    def extract_all_features(self, df):
        """提取所有特征"""
        print("\n开始特征提取...")
        
        # 元素属性特征
        print("1. 元素属性特征...")
        ep_featurizer = ElementProperty.from_preset('magpie')
        df = ep_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        
        # Meredig特征
        print("2. Meredig特征...")
        meredig_featurizer = Meredig()
        df = meredig_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        
        # 化学计量特征
        print("3. 化学计量特征...")
        stoichiometry_featurizer = Stoichiometry()
        df = stoichiometry_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        
        # 离子特性特征
        print("4. 离子特性特征...")
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
        
        return df
    
    def filter_features(self, df, nan_threshold=0.4, zero_threshold=0.5):
        """过滤特征"""
        print("\n开始特征过滤...")
        
        # 删除缺失值比例太高的列
        nan_ratio = df.isnull().sum() / df.shape[0]
        data_filtered = df.loc[:, nan_ratio < nan_threshold]
        
        print(f"保留的特征数: {data_filtered.shape[1]}")
        return data_filtered
    
    def save_features(self, df, filename):
        """保存特征到文件"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_excel(filepath, index=False)
        print(f"特征已保存到: {filepath}")
        return filepath

# 使用示例
if __name__ == "__main__":
    extractor = MaterialFeatureExtractor()
    df_processed = extractor.load_and_process_data("DATA/data.xlsx")
    
    if df_processed is not None:
        df_with_features = extractor.extract_all_features(df_processed)
        df_filtered = extractor.filter_features(df_with_features)
        extractor.save_features(df_filtered, "Magpie-data-features.xlsx")