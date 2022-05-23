#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# In[2]:


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


# In[3]:


df_train = read_dataframe('fhv_tripdata_2021-01.parquet', data_t='januray')
df_val = read_dataframe('fhv_tripdata_2021-02.parquet', data_t='feb')


# In[4]:


df_train['PUlocationID'].dtype


# In[5]:


dv = DictVectorizer()

train_dicts = df_train[['PUlocationID', 'DOlocationID']].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[['PUlocationID', 'DOlocationID']].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[6]:


X_train.shape


# In[25]:


# df_train = pd.concat([pd.get_dummies(df_train[['PUlocationID', 'DOlocationID']]), df_train[['duration']]], axis=1)


# In[27]:


# df_val = pd.concat([pd.get_dummies(df_val[['PUlocationID', 'DOlocationID']]), df_val[['duration']]], axis=1)


# In[8]:


X_val.shape


# In[9]:


y_train =df_train['duration'].values


# In[10]:


y_val =df_val['duration'].values


# In[11]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[12]:


y_pred = lr.predict(X_train)


# In[13]:


rms = mean_squared_error(y_train, y_pred, squared=False)
print('Train RMSE', rms)


# In[ ]:





# In[14]:


y_pred = lr.predict(X_val)


# In[15]:


rms = mean_squared_error(y_val, y_pred, squared=False)
print('validation RMSE', rms)


# In[ ]:




