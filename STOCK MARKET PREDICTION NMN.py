#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data_set = pd.read_csv('NSE-TATAGLOBAL.csv')
data_set


# In[3]:


data_set.head(5)


# In[4]:


data_set.tail(5)


# In[5]:


data_set.dtypes


# In[6]:


data_set['High'].hist()


# In[7]:


training_set = data_set.iloc[:, 1:2].values


# In[8]:


training_set


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_training_scaled = scaler.fit_transform(training_set)


# In[10]:


features_set = []
labels = []
for i in range(60, 586):
    features_set.append(data_training_scaled[i-60:i, 0])
    labels.append(data_training_scaled[i, 0])
    


# In[11]:


features_set, labels = np.array(features_set), np.array(labels)


# In[12]:


features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape


# In[14]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM


# In[15]:


m = Sequential()


# In[17]:


m.compile(optimizer = 'adam', loss = 'mean_squared_error')

m.fit(features_set, labels, epochs = 50, batch_size = 20)


# In[18]:


dataset_test = pd.read_csv('NSE-TATAGLOBAL.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# In[19]:


real_stock_price 


# In[24]:



data_total = pd.concat((data_set['Open'], data_set['Open']), axis=0)
test_inputs = data_total[len(data_total) - len(data_set) - 60:].values
test_inputs.shape
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(60, 89):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
test_features.shape
predictions = m.predict(test_features)
predictions


# In[29]:



plt.figure(figsize=(24,9))
plt.title("STOCK PREDICTED")
plt.plot(dataset_test['Close'])
plt.xlabel('PRICE',fontsize=14)
plt.ylabel('Quantity',fontsize=14)
plt.show()


# In[ ]:




