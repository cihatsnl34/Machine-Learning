# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#2.veriOnisleme
#2.1 veri yükleme
veriler= pd.read_csv("veriler.csv")


#2.2 Eksik veriler
#sci - kit learn
from sklearn.impute import SimpleImputer
Yas = veriler.iloc[:,1:4].values

print(Yas)

#Encoder:Kategorik -> Numeric
#
ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
#1,2,3 :LabelEncoder() 
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

#1-0-0 :OneHotEncoder() 
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#Encoder:Kategorik -> Numeric
#
c=veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing
#1,2,3 :LabelEncoder() 
le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])





#numpy dizileri dataframe donusumu
sonuc=pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)
print(veriler)
sonuc2=pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=c,index=range(22),columns=["cinsiyet"])
print(sonuc3)


#dataframe birleştirme işlemi axis:yanına yerleştiriyor default olarak altına verileri yerleştiriyor
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#boyun tahmini
boy=s2.iloc[:,3:4]
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1)

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

#Elemeden boy tahmini için eğitiyoruz
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_ped=regressor.predict(x_test)

#Eleme yaparak boy tahmini için eğitiyoruz
import statsmodels.api as sm

X=np.append(arr= np.ones((22,1)).astype(int), values=veri,axis=1)
#OLS Raporuna göre eleme yapıyoruz P.values 0.05 den büyükse eliyoruz 
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l, dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l, dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l, dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_l,boy,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_ped=regressor.predict(x_test)
