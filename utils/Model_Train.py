#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autogluon as ag
import pandas as pd
import numpy as np
import os,urllib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_excel("./data/7_feature_selection.xlsx",index_col=0)
data


# In[3]:


train_data,test_data = train_test_split(data,test_size=0.3,random_state=42)


# In[4]:


train_data


# In[5]:


train_data.to_excel("./data/train_data.xlsx")


# In[6]:


test_data


# In[7]:


test_data.to_excel("./data/test_data.xlsx")


# In[8]:


test_data.iloc[:,2:]


# In[9]:


predictor = TabularPredictor(label="log_cond",eval_metric="r2",problem_type="regression").fit(train_data.iloc[:,2:], 
                                                                                                tuning_data=test_data.iloc[:,2:],
                                                                                                #presets='best_quality',
                                                                                                #use_bag_holdout=True
                                                                                               )


# In[10]:


#保存最佳模型
predictor.save(silent=True)


# In[11]:


predictor.model_best


# In[12]:


predictor.path


# In[13]:


predictor.leaderboard(train_data,extra_metrics=['mae', 'rmse',  'pearsonr'])


# In[14]:


predictor.leaderboard(test_data,extra_metrics=['mae','rmse','pearsonr'])


# In[15]:


predictor.model_names()


# In[16]:


feature_importance = predictor.feature_importance(train_data)


# In[17]:


feature_importance.to_excel("./data/7_feature_import.xlsx")


# In[ ]:




