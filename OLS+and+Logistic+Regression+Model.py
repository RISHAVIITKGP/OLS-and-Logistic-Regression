
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[13]:


df = pd.read_csv('train_new.csv')


# In[14]:


df.describe()


# In[26]:


df_1 = df.drop(['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Faults'],axis = 1)


# In[27]:


df_1.describe()


# In[18]:


from sklearn import preprocessing


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[22]:


from sklearn.cross_validation import train_test_split


# In[23]:


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))


# In[24]:


df_1 = np.array(df_1)


# In[33]:


df_y = df[['Faults']]


# In[37]:


df_y = df[['Faults']]


# In[39]:


df_y = np.array(df_y)


# In[40]:


df_y


# In[53]:


# TO CONVERT STRINGS TO LEBELS
df_y_new = np.asarray(df_y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_y_new)
df_y_new = le.transform(df_y_new)
df_y_new


# In[54]:


# convert to dictionary, since we are using more than one column theredore we cant use np.asarray
df_1_new = df_1.to_dict(orient = 'record')


# In[55]:


# need to convert this directory to vector for operations 
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
df_2 = vec.fit_transform(df_1_new).toarray()


# In[60]:


clf = LogisticRegression()


# In[62]:


X_train


# In[63]:


Y_train


# In[68]:


np.any(np.isnan(X_train))


# In[70]:


X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])


# In[71]:


np.any(np.isnan(X_train))


# In[72]:


X_train_new = min_max_scaler.fit_transform(X_train)


# In[73]:


X_train_new


# In[104]:


clf = LogisticRegression()
d = clf.fit(X_train_new,Y_train)


# In[76]:


clf.score(X_train_new,Y_train)


# In[89]:


clf.score(X_test,Y_test)


# In[94]:


clf.intercept_


# In[96]:


clf.coef_


# In[105]:


import statsmodels.api as sm


# In[106]:


from scipy import stats


# In[108]:


import statsmodels.api as sm
from scipy import stats



# In[109]:


est = sm.OLS(Y_train, X_train)


# In[110]:


est2 = est.fit()


# In[111]:


est2.summary()

