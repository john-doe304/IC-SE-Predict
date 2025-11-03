#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import shap
import warnings
from tqdm import tqdm


# In[2]:


# 禁用警告
warnings.filterwarnings('ignore')


# In[14]:


# 创建保存SHAP图片的目录
os.makedirs('./data/shap_plots', exist_ok=True)


# In[15]:


# 读取数据
train_data = pd.read_excel("./data/train_data.xlsx", index_col=0)
train_data
#test_data = pd.read_excel("./data/test_data.xlsx", index_col=0)


# In[16]:


# 准备特征数据
feature_names = train_data.drop(columns=["log_cond","Cond"]).iloc[:, 1:].columns
train_features = train_data.drop(columns=["log_cond","Cond"]).iloc[:, 1:].copy()


# In[17]:


feature_names


# In[18]:


# 加载Autogluon predictor
predictor = TabularPredictor.load("./AutogluonModels/ag-20251024_075719")


# In[19]:


train_features


# In[20]:


# 选择需要解释的数据
shap_input = train_features.sample(709, random_state=42)


# In[21]:


model_weights = {'CatBoost': 0.56, 'ExtraTreesMSE': 0.28, 'LightGBM': 0.08, 'KNeighborsDist': 0.04, 'XGBoost': 0.04}


# In[22]:


# 用于保存所有模型的 SHAP 值
all_shap_values = []


# In[23]:


# 设置字体为新罗马字体，并将 X 轴和 Y 轴标签设置为加粗
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelweight'] = 'bold'


# In[24]:


print("使用指定的模型权重:")
for name, weight in model_weights.items():
    print(f"{name}: {weight:.3f}")

    # 加载每个子模型
    model_obj = predictor._trainer.load_model(name)

    # 仅处理支持 SHAP 的模型
    if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'predict'):
        booster = model_obj.model

        try:
            # 计算SHAP值
            explainer = shap.TreeExplainer(booster)
            shap_vals = explainer.shap_values(shap_input)
            
            # ==== 关键修复1：使用显式图形对象 ====
            # 创建独立图形对象并设置后端
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            
            # ==== 关键修复2：确保图形完整生命周期 ====
            # SHAP Summary Plot
            fig_summary = plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_vals, 
                shap_input,
                feature_names=feature_names,
                show=False  # 阻止内部调用plt.show()
            )
            #plt.title(f"SHAP Summary Plot - {name}")
            
            fig_summary.savefig(
                f"./data//shap_plots/{name}_summary_plot.png", 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.close(fig_summary)  # 显式关闭图形
            
            # Mean SHAP条形图
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_mean_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean |SHAP|': mean_abs_shap
            }).sort_values(by='Mean |SHAP|', ascending=False)

            fig_bar = plt.figure(figsize=(10, 6))
            bars = plt.barh(shap_mean_df['Feature'], shap_mean_df['Mean |SHAP|'])
            plt.gca().invert_yaxis()
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Mean SHAP Values - {name}')
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.001,  # 避免重叠
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', 
                    ha='left', 
                    va='center'
                )
            fig_bar.savefig(
                f"./data/shap_plots/{name}_mean_shap.png", 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.close(fig_bar)  # 显式关闭图形
            
            # 加权SHAP值
            shap_vals_weighted = np.array(shap_vals) * weight
            all_shap_values.append(shap_vals_weighted)
            
            # 保存特征重要性
            with open(f"./data/shap_plots/{name}_feature_importance.txt", "w") as f:
                f.write("Feature Importance for model: {}\n\n".format(name))
                for idx, feature in enumerate(shap_mean_df['Feature']):
                    f.write(f"{feature}: {shap_mean_df['Mean |SHAP|'].iloc[idx]:.5f}\n")

        except Exception as e:
            print(f"模型 {name} 无法解释: {e}")


# In[25]:


total_weight = sum(model_weights.values())
# 加权平均 SHAP 值（总和除以权重之和）
if all_shap_values:
    combined_shap_values = np.sum(all_shap_values, axis=0) / total_weight

    # Summary plot for combined SHAP
    fig_summary = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        combined_shap_values, shap_input, feature_names=feature_names, show=False
    )
    #plt.title("Weighted SHAP Summary Plot (Custom Weights)")
    fig_summary.savefig(
        "./data/shap_plots/weighted_summary_plot.png", 
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig_summary)

    # Mean SHAP bar plot for combined
    mean_abs_shap_combined = np.abs(combined_shap_values).mean(axis=0)
    shap_mean_combined_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap_combined
    }).sort_values(by='Mean |SHAP|', ascending=False)

    fig_bar = plt.figure(figsize=(10, 6))
    bars = plt.barh(shap_mean_combined_df['Feature'], shap_mean_combined_df['Mean |SHAP|'])
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value|')
    plt.title('Weighted SHAP Feature Importance (Custom Weights)')
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001 * max(shap_mean_combined_df['Mean |SHAP|']),
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left', va='center', fontsize=8
        )
    plt.tight_layout()
    fig_bar.savefig(
        "./data/shap_plots/weighted_mean_shap.png",
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig_bar)

    # 保存加权特征重要性具体数值
    with open("./data/shap_plots/weighted_feature_importance.txt", "w") as f:
        f.write("Weighted Feature Importance\n\n")
        for idx, feature in enumerate(shap_mean_combined_df['Feature']):
            f.write(f"{feature}: {shap_mean_combined_df['Mean |SHAP|'].iloc[idx]:.5f}\n")
else:
    print("没有成功解释的模型 SHAP 值。")


# In[ ]:




