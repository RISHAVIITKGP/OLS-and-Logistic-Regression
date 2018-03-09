
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('train_new.csv')

df.describe()

df_1 = df.drop(['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Faults'],axis = 1)

df_1.describe()

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

df_1 = np.array(df_1)

df_y = df[['Faults']]

df_y = df[['Faults']]

df_y = np.array(df_y)

df_y

# TO CONVERT STRINGS TO LEBELS
df_y_new = np.asarray(df_y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_y_new)
df_y_new = le.transform(df_y_new)
df_y_new

# convert to dictionary, since we are using more than one column theredore we cant use np.asarray
df_1_new = df_1.to_dict(orient = 'record')

# need to convert this directory to vector for operations 
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
df_2 = vec.fit_transform(df_1_new).toarray()

clf = LogisticRegression()

X_train

Y_train

np.any(np.isnan(X_train))

X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])

np.any(np.isnan(X_train))

X_train_new = min_max_scaler.fit_transform(X_train)

X_train_new

clf = LogisticRegression()
d = clf.fit(X_train_new,Y_train)

clf.score(X_train_new,Y_train)

clf.score(X_test,Y_test)

clf.intercept_

clf.coef_

import statsmodels.api as sm

from scipy import stats

import statsmodels.api as sm
from scipy import stats

est = sm.OLS(Y_train, X_train)

est2 = est.fit()

est2.summary()

