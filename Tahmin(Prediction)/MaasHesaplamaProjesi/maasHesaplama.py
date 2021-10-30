# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:29:05 2021

@author: cihat
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
x=veriler.iloc[:,2:-1]
y=veriler.iloc[:,-1:]
X=x.values
Y=y.values
Sabit=np.append(arr= np.ones((30,1)).astype(int), values=x,axis=1)
#OLS Raporuna göre eleme yapıyoruz P.values 0.05 den büyükse eliyoruz 
#kıdem puan
X_l=x.iloc[:,[0,1,2]].values
X_l=np.array(X_l, dtype=float)

#linear regression
#doğrusal model oluşturma

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))