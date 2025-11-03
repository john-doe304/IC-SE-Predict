#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[3]:


data = pd.read_excel("Data/Data_Procesing.xlsx",index_col=0)
data


# In[4]:


data.iloc[:,2:]


# In[5]:


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='log_cond',eval_metric="r2").fit(
                                                               train_data=data.iloc[:,2:],
                                                               #presets='best_quality',
                                                               #hyperparameters=hyperparameters,
                                                               #feature_metadata=feature_metadata,
                                
)


# In[6]:


predictor.model_best


# In[7]:


predictor.path


# In[8]:


data.iloc[:,3:]


# In[9]:


predictor.leaderboard(data)


# In[10]:


feature_importance = predictor.feature_importance(data.iloc[:,2:])
feature_importance


# In[11]:


data.iloc[:,2]


# In[12]:


# 根据特征重要性的排名重新训练数据，并绘制得分图
scores = []
for i in range(1,35):
    # 选择前i个特征
    features = feature_importance.index[:i].tolist()
    #数据集进行链接
    df1 = pd.DataFrame(data.iloc[:,2])
    df2=pd.DataFrame(data[features])
    data_1= pd.concat([df1, df2],axis=1)
    data_1
    # 训练模型
    predictor = TabularPredictor(label="log_cond",eval_metric="r2").fit(data_1,
                                                                          #hyperparameters='multimodal',
                                                                          #presets='good_quality',
                                                                         )
    
    # 计算得分
    score = predictor.evaluate(data_1)["r2"]
    
    #y_pred = predictor.predict(test_data)
    #score = r2_score(test_data[label_column], y_pred)
    scores.append(score)
    
    # 显示进度
    print(f'Training with top {i} features, score = {score:.4f}')


# In[15]:


from pylab import xticks,yticks,np
# 绘制得分图
import matplotlib.pyplot as plt
# 查看前20个特征的得分
plt.figure(figsize=(11,4))
plt.title("Features-R2")
#  选择的特征数量
plt.xlabel("Number of features selected")
# 交叉验证得分
plt.ylabel("R2_Score")

# 修改横轴坐标刻度
xticks(np.linspace(1,34,34,endpoint=True))
# yticks(np.linspace(0.7,1,7,endpoint=True))

# 画出各个特征的得分zzzzz
plt.plot(range(1, len(scores[:34])+1),scores[:34])
plt.grid()
plt.show()


# In[16]:


df = pd.DataFrame(scores)


# In[17]:


df.to_excel("data/scores_34.xlsx")


# In[19]:


# 选择前i个特征
features = feature_importance.index[:7].tolist()
features


# In[20]:


df1 = pd.DataFrame(data.iloc[:,:3])
df2=pd.DataFrame(data[features])
data_1= pd.concat([df1, df2],axis=1)
data_1


# In[21]:


data_1.to_excel("./data/7_feature_selection.xlsx")


# In[ ]:




