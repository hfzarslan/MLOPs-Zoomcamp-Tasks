#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# In[20]:


def read_dataframe(filename, data_t):
    df = pq.read_table(filename)
    df = df.to_pandas()
    print(data_t, df.shape)
    df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    print(data_t, df.duration.mean())
    
    
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df.PUlocationID.fillna(-1, inplace=True)
    df.DOlocationID.fillna(-1, inplace=True)
    
    print(data_t, df.loc[df['PUlocationID'] == -1].shape[0]/df.shape[0]*100)
    
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


# In[21]:


df_train = read_dataframe('fhv_tripdata_2021-01.parquet', data_t='januray')
df_val = read_dataframe('fhv_tripdata_2021-02.parquet', data_t='feb')


# In[22]:


df_train['PUlocationID'].dtype


# In[25]:


df_train = pd.concat([pd.get_dummies(df_train[['PUlocationID', 'DOlocationID']]), df_train[['duration']]], axis=1)


# In[32]:


df_train.columns


# In[27]:


df_val = pd.concat([pd.get_dummies(df_val[['PUlocationID', 'DOlocationID']]), df_val[['duration']]], axis=1)


# In[33]:


df_val.columns


# In[34]:


[x for x in df_val if x not in df_train]


# In[35]:


df_val.drop(columns=['DOlocationID_110.0'], inplace=True)


# In[37]:


df_train.shape


# In[38]:


df_val.shape


# In[39]:


y_train =df_train['duration'].values
X_train = df_train.drop(columns=['duration'])


# In[40]:


y_val =df_val['duration'].values
X_val = df_val.drop(columns=['duration'])


# In[41]:


X_train.shape


# In[42]:


X_val.shape


# In[43]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[44]:


y_pred = lr.predict(X_train)


# In[45]:


rms = mean_squared_error(y_train, y_pred, squared=False)
print('Train RMSE', rms)


# In[ ]:





# In[46]:


y_pred = lr.predict(X_val)


# In[47]:


rms = mean_squared_error(y_val, y_pred, squared=False)
print('validation RMSE', rms)


# In[ ]:




