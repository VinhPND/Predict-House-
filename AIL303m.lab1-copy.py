#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


df = pd.read_csv('kc_house_price.txt')


# In[30]:


df.shape


# In[31]:


df.head()


# In[32]:


features = ['bedrooms','bathrooms','sqft_living','floors']


# In[33]:


X = df[features][:10000]
Y = df['price'][:10000]


# In[34]:


X.head()


# NORMALIZATION DATA

# In[35]:


X_norm = (X - X.mean())/X.std()


# In[36]:


X_norm.head()


# In[37]:


X_norm.shape


# In[38]:


Y.shape


# ### ML MODEL

# In[40]:


#import nescessary library
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X_norm,Y, test_size=0.3, random_state=42)


# In[42]:


max_degree = 4
err_train = np.zeros(max_degree)    
err_test= np.zeros(max_degree) 

for k in range(max_degree):


    poly_features = PolynomialFeatures(degree=k+1)
    X_poly_train = poly_features.fit_transform(X_train)


    poly_features2 = PolynomialFeatures(degree=k+1)
    X_poly_test = poly_features2.fit_transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    poly_predict_train = poly_reg.predict(X_poly_train)

    poly_mse_train = mean_squared_error(y_train, poly_predict_train)
    poly_rmse_train = np.sqrt(poly_mse_train)
    
    err_train[k] = poly_rmse_train

    poly_predict_test = poly_reg.predict(X_poly_test)
    poly_mse_test = mean_squared_error(y_test, poly_predict_test)
    poly_rmse_test = np.sqrt(poly_mse_test)

    err_test[k] = poly_rmse_test


# In[43]:


err_train


# In[44]:


err_test


# In[45]:


plt.plot(err_train, label = "err_train")
plt.plot(err_test, label = "err_test")
plt.xlabel('Degree')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### DEEP LEARNING MODEL

# In[46]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[47]:


def regression_model():
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape = (4,)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    return model


# In[48]:


model = regression_model()


# In[49]:


results_train = []
results_test = []
for i in range(10):
    print(i)
    history = model.fit(X_train, y_train, validation_split=0.3, epochs = 20, batch_size=100, verbose = 0)
    results_train.append(model.evaluate(X_train, y_train))
    results_test.append(model.evaluate(X_test, y_test))


# In[51]:


plt.plot(np.sqrt(results_train), label = 'loss_train')
plt.plot(np.sqrt(results_test), label = 'loss_test')
plt.legend()


# In[ ]:




