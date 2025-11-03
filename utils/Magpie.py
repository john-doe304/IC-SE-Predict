#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.composition import (
    ElementProperty, Meredig, Stoichiometry, ValenceOrbital, IonProperty
)
from matminer.featurizers.structure import (
    DensityFeatures, GlobalSymmetryFeatures, StructuralHeterogeneity,
    MaximumPackingEfficiency, ChemicalOrdering, StructureComposition
)
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition


# In[2]:


# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# In[3]:


df = pd.read_excel("data/data.xlsx")
df


# In[4]:


# 读取Excel文件

df_excel = df

# 检查数据
print("原始Excel数据列名:", df_excel.columns.tolist())
print("\n前5行数据:")
display(df_excel.head())

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

# 检查转换结果
print("\n转换后的数据:")
display(df_excel.head())


# In[5]:


#元素属性特征
ep_featurizer = ElementProperty.from_preset('magpie')
print("\n1. 元素属性特征...")
df_excel = ep_featurizer.featurize_dataframe(df_excel, 'composition', ignore_errors=True)


# In[6]:


#Meredig特征（从化学式中提取物理化学特征）
meredig_featurizer = Meredig()
print("\n2. Meredig特征...")
df_excel = meredig_featurizer.featurize_dataframe(df_excel, 'composition', ignore_errors=True)


# In[7]:


#化学计量特征
stoichiometry_featurizer = Stoichiometry()
print("\n3. 化学计量特征...")
df_excel = stoichiometry_featurizer.featurize_dataframe(df_excel, 'composition', ignore_errors=True)


# In[8]:


# 离子特性特征需要先转换氧化态
print("\n4. 离子特性特征...")
cto = CompositionToOxidComposition()
df_excel = cto.featurize_dataframe(df_excel, 'composition', ignore_errors=True)


# In[9]:


# 检查生成的特征
print("\n生成的特征列名:", [col for col in df_excel.columns if col not in ['composition', 'Formula']])
print("\n最终数据预览:")
display(df_excel.head())


# In[10]:


df_excel


# In[11]:


# 设置保留阈值（允许少量NaN和0）
nan_threshold = 0.4   # 缺失率 < 40%
zero_threshold = 0.5  # 0值比例 < 50%

# 第一步：删除缺失值比例太高的列
nan_ratio = df_excel.isnull().sum() / df_excel.shape[0]
data_filtered = df_excel.loc[:, nan_ratio < nan_threshold]



# 打印结果
print("保留的特征数：", data_filtered.shape[1])
data_filtered.head()


# In[12]:


data_filtered.to_excel("./data/Magpie-data-features.xlsx")


# In[ ]:




