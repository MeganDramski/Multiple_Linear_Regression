#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:23:37 2018

@author: meganpolak
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

# Encoding categotrical data
#encoding the Independant Variable 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,3]= labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Voidnig the  Dummy variable trap 

X = X[:,1:]

# Spliting the dataset into the Traing set and Test set 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.2, random_state=0)


# Fitting Multiple Linear Regression to the Traing set 

from sklearn.linear_model import LinearRegression 
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set result 

y_pred=regressor.predict(X_test)
