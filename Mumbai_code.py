#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd
import matplotlib as plt
# from pandas_profiling import ProfileReport


# In[73]:


mumbai=pd.read_csv('Mumbai House Prices.csv')


# In[74]:


mumbai.head()


# In[75]:


mumbai['price_unit']=mumbai['price_unit'].replace(["Cr","L"],[10000000,100000])
mumbai['price']=mumbai['price']*mumbai['price_unit']
mumbai=mumbai[['bhk','type','area','region','locality','price']]
mumbai.head()


# In[76]:


mumbai['bhk'].value_counts()


# In[77]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[78]:


mumbai.head()


# In[79]:


one_hot_cols = ["type", "region", "locality"]
# ordinal_cols=["status", "age"]
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'), one_hot_cols)
#         ('ordinal', OrdinalEncoder(), ordinal_cols),
    ],
    remainder='passthrough'
)


# In[80]:


# for scaling
tr3=ColumnTransformer([
    ('scaler',StandardScaler(with_mean=False),slice(0,9235))
],remainder='passthrough')
tr3


# In[81]:


reg=LinearRegression()


# In[82]:


pipe = make_pipeline(preprocessor,tr3,reg)
pipe


# In[83]:


x_train,x_test,y_train,y_test = train_test_split(mumbai.drop('price',axis=1),mumbai['price'],test_size=0.2,random_state=5)


# In[84]:


pipe = make_pipeline(preprocessor,tr3,reg)


# In[85]:


pipe.fit(x_train,y_train)
r=pipe.predict(x_test)


# In[86]:


r2_score(y_test,r)


# In[87]:


r=Ridge()


# In[88]:


y_predict_reg=pipe.predict(x_test)


# In[89]:


r2_score(y_test,y_predict_reg)


# In[90]:


import pickle


# In[91]:


pickle.dump(pipe,open('ridge_model.pkl','wb'))


# In[92]:


ridge_model=pickle.load(open('ridge_model.pkl','rb'))


# In[ ]:




