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

print("Linear OLS")
print(model.fit().summary())

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

#doğrusal olmayan (nonlinear model) oluşturma


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#Polinom Derece
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
print('POLY OLS')
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print('POLY R2')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

#SVR PREDİCT
#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli= sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli= sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR OLS")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print("SVR R2")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

#KARAR AGACI ALGORİTMASI (DECİSİON TREE)
from sklearn.tree import DecisionTreeRegressor
dc_reg=DecisionTreeRegressor(random_state=0)
dc_reg.fit(X,Y)
print("DECİSİON TREE OLS")
model4=sm.OLS(dc_reg.predict(X),X)
print(model4.fit().summary())
print("DECISION TREE R2")
print(r2_score(Y,dc_reg.predict(X)))

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
print("RANDOM FOREST OLS")
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
print("RANDOM FOREST R2")
print(r2_score(Y,rf_reg.predict(X)))

#R2 DEGERLERİ
print ("--------------------")
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('POLY R2')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("SVR R2")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("DECISION TREE R2")
print(r2_score(Y,dc_reg.predict(X)))

print("RANDOM FOREST R2")
print(r2_score(Y,rf_reg.predict(X)))

print(lin_reg.predict([[10,10,100]]))

