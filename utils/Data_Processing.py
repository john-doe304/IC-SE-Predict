#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rdkit
import pandas as pd
import numpy as np
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


data = pd.read_excel("data/Magpie-data-features.xlsx",index_col=0)
data


# In[3]:


columns_to_drop = ['composition_obj', 'composition']


# In[4]:


data =  data.drop(columns=columns_to_drop,errors="ignore")
data


# In[5]:


# 更保守的特征保留策略
import pandas as pd
import numpy as np

# 1. 首先只删除完全为空值的列
consolidated_data = data.dropna(axis=1, how='all')
print(f"删除全空列后形状: {consolidated_data.shape}")

# 2. 删除缺失值比例过高的列（阈值从50%调整为80%）
missing_ratio_threshold = 0.8  # 只有当80%以上为缺失值时才删除
missing_stats = consolidated_data.isnull().mean()  # 每列的缺失值比例

high_missing_cols = missing_stats[missing_stats >= missing_ratio_threshold].index
if len(high_missing_cols) > 0:
    print(f"删除缺失值比例超过{missing_ratio_threshold*100}%的列: {high_missing_cols.tolist()}")
    consolidated_data = consolidated_data.drop(columns=high_missing_cols)

# 3. 更保守的零值处理：只删除全为零的列
zero_cols = (consolidated_data == 0).all(axis=0)
zero_cols_to_drop = zero_cols[zero_cols].index
if len(zero_cols_to_drop) > 0:
    print(f"删除全为零的列: {zero_cols_to_drop.tolist()}")
    consolidated_data = consolidated_data.drop(columns=zero_cols_to_drop)

# 4. 对于有较多零值但不全为零的列，进行标记而不是删除
zero_count = (consolidated_data == 0).sum(axis=0)
high_zero_cols = zero_count[zero_count > 0].index

print(f"\n零值统计（不删除，仅供参考）:")
for col in high_zero_cols:
    zero_pct = (zero_count[col] / len(consolidated_data)) * 100
    print(f"  {col}: {zero_count[col]}个零值 ({zero_pct:.1f}%)")

# 5. 处理剩余缺失值（用中位数填充而不是删除列）
print("\n处理剩余缺失值...")
for col in consolidated_data.columns:
    if consolidated_data[col].isnull().any():
        missing_count = consolidated_data[col].isnull().sum()
        # 对于数值列，用中位数填充
        if pd.api.types.is_numeric_dtype(consolidated_data[col]):
            median_val = consolidated_data[col].median()
            consolidated_data[col] = consolidated_data[col].fillna(median_val)
            print(f"  数值列 '{col}': 用中位数 {median_val:.4f} 填充 {missing_count} 个缺失值")
        else:
            # 对于非数值列，用众数填充或保留原缺失值
            mode_val = consolidated_data[col].mode()[0] if not consolidated_data[col].mode().empty else np.nan
            consolidated_data[col] = consolidated_data[col].fillna(mode_val)
            print(f"  非数值列 '{col}': 用众数 '{mode_val}' 填充 {missing_count} 个缺失值")

print(f"\n最终数据形状: {consolidated_data.shape}")
print(f"保留的列数: {len(consolidated_data.columns)}")
print("保留的列:", consolidated_data.columns.tolist())

# 显示数据前几行
consolidated_data.head()


# In[6]:


# 删除低方差的值
var_data = consolidated_data.iloc[:,3:].var()
del_col = var_data[var_data<0.1].index
consolidated_data = consolidated_data.drop(labels=del_col,axis=1)
print(consolidated_data.shape)
consolidated_data.head()


# In[7]:


correlation = consolidated_data.iloc[:,3:].corr('spearman')
correlation


# In[8]:


df_bool = (abs(correlation) > 0.85)
correlation[df_bool]


# In[9]:


DN_correlation = consolidated_data.iloc[:,2:].corr('spearman')["log_cond"]
DN_correlation[DN_correlation>0.1]


# In[10]:


bb = []
col_index = correlation.index
for i in range(0,len(col_index)):
    for j in range(i+1,len(col_index)):
        bb.append([col_index[i],col_index[j]])


# In[11]:


import math
k = 0
del_list = []
for i in bb:
    if not math.isnan(correlation[df_bool].loc[i[0],i[1]]) and ('log_cond'not in i):
        k+=1
        if abs(DN_correlation[i[0]])>abs(DN_correlation[i[1]]):
            if i[1] not in del_list:
                del_list.append(i[1])
        else:
            if i[0] not in del_list:
                del_list.append(i[0])
            
print(del_list)
k 


# In[12]:


Test_data = consolidated_data.drop(labels=del_list,axis=1)
Test_data


# In[13]:


Test_data.to_excel("data/Data_Procesing.xlsx")


# In[ ]:




