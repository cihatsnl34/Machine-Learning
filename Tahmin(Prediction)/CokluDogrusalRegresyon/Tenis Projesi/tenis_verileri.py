# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 21:58:36 2021

@author: cihat
"""

#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#2.veriOnisleme
#2.1 veri yükleme
veriler= pd.read_csv("odev_tenis.csv")
#veriler= pd.read_csv("veriler.csv")


Yas = veriler.iloc[:,1:3].values

print(Yas)

#Encoder:Kategorik -> Numeric
#
outlook=veriler.iloc[:,0:1].values
print(outlook)
windy=veriler.iloc[:,3:4].values
print(windy)
play=veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
c = veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
#1-0-0 :OneHotEncoder() 



#numpy dizileri dataframe donusumu
havadurumu=pd.DataFrame(data=c, index=range(14),columns=['o','s','r'])
print(havadurumu)
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)


#Elemeden boy tahmini için eğitiyoruz

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_ped=regressor.predict(x_test)

#Eleme yaparak boy tahmini için eğitiyoruz
import statsmodels.api as sm

X=np.append(arr= np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1)

#Backward Elimination
#OLS Raporuna göre eleme yapıyoruz P.values 0.05 den büyükse eliyoruz 
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l, dtype=float)
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler=sonveriler.iloc[:,1:]

import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

regressor.fit(x_train,y_train)
y_ped=regressor.predict(x_test)

