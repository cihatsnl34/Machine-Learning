# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:29:05 2021

@author: cihat
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#Data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY dizi(array) dönüşümü
X = x.values
Y = y.values


#linear regression
#doğrusal model oluşturma

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)




#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2.Dereceden polinom

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)#Polinom Derece
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)



#4. dereceden polinom

poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#Görselleştirme

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg3.predict(poly_reg3.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg3.fit_transform([[11]])))


#ZORUNLULUK
#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli= sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli= sc2.fit_transform(Y)


#SVR ALGORİTMASI
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


plt.scatter(x_olcekli,y_olcekli, color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

#KARAR AGACI ALGORİTMASI (DECİSİON TREE)

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='yellow')

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))